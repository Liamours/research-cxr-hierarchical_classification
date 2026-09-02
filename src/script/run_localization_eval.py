"""A9: localization evaluation against the corrected NIH boxes (A8).
Rebuilds Table IV (main.tex tab:loc) from scratch -- the scripts that
produced it were never committed. Grad-CAM at the last DenseNet121 block,
predicted box = largest connected component above the heatmap's own 90th
percentile, against the 6 NIH bbox categories that map to canonical
findings, restricted to images in our test split. CPU-only by default
(no training, single forward+backward per image; kept off GPU on
purpose when the GPU is in use for something else).

    uv run python src/script/run_localization_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from src.config.experiment_config import ExperimentConfig
from src.data.label_space import CANONICAL_LABELS
from src.data.transforms import build_transform, norm_kind_for_backbone
from src.model.classifier import build_model_from_cfg
from src.xai.gradcam import GradCAM, resolve_target_layer
from src.xai.localization_metrics import (
    average_precision_at_iou, box_iobb, box_iou, largest_component_box,
    pointing_game_hit,
)

CORRECTED_BOXES = Path(
    r"C:\rifqi\research-cxr-hierarchical_classification\dataset\nih-cxr14\preprocessed\labels\nih_bboxes_224_corrected.csv"
)
COMBINED_CSV = Path(r"C:\rifqi\research-cxr-hierarchical_classification\dataset\combined\combined.csv")

# NIH BBox_List_2017.csv's 8 raw categories -> canonical label. Mass and
# Nodule both collapse to Solitary_Pulmonary_Nodule (configs/label_equivalence.json),
# so 7 of 8 raw categories map but only 6 distinct canonical findings result.
# Infiltration has no canonical target and is dropped.
NIH_BBOX_TO_CANONICAL = {
    "Atelectasis": "Atelectasis",
    "Cardiomegaly": "Cardiomegaly",
    "Effusion": "Pleural_Effusion",
    "Mass": "Solitary_Pulmonary_Nodule",
    "Nodule": "Solitary_Pulmonary_Nodule",
    "Pneumonia": "Pneumonia",
    "Pneumothorax": "Pneumothorax",
}

CONDITIONS = {
    "flat": r"C:\rifqi\research-cxr-hierarchical_classification\weights\classification-densenet121_xrv-flat-260719",
    "hierarchical": r"C:\rifqi\research-cxr-hierarchical_classification\weights\classification-densenet121_xrv-hierarchical-260718",
}
OUT_DIR = Path(r"C:\rifqi\research-cxr-hierarchical_classification\analyses\localization_eval-260829")
IOU_THRESHOLDS = (0.1, 0.25, 0.5)


def load_test_boxes() -> pd.DataFrame:
    boxes = pd.read_csv(CORRECTED_BOXES)
    boxes["canonical"] = boxes["label"].map(NIH_BBOX_TO_CANONICAL)
    boxes = boxes.dropna(subset=["canonical"]).copy()
    boxes["nih_image_id"] = boxes["image_id"].str.replace(".png", "", regex=False)

    combined = pd.read_csv(COMBINED_CSV, usecols=["image_id", "image_path", "dataset", "split"])
    nih = combined[combined["dataset"] == "nih-cxr14"]
    merged = boxes.merge(nih, left_on="nih_image_id", right_on="image_id", suffixes=("", "_c"))
    test_boxes = merged[merged["split"] == "test"].copy()
    return test_boxes


def load_condition_model(run_dir: str, device: torch.device):
    cfg = ExperimentConfig.from_yaml(str(Path(run_dir) / "config.yaml"))
    model = build_model_from_cfg(cfg, pretrained=False)
    ckpt = Path(run_dir) / "checkpoints" / "best_val_auroc_macro.pt"
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    norm_kind = norm_kind_for_backbone(cfg.model.backbone)
    transform = build_transform(cfg.data.image_size, augment=False, norm_kind=norm_kind)
    return model, transform


def evaluate_condition(name: str, run_dir: str, test_boxes: pd.DataFrame, device: torch.device) -> dict:
    model, transform = load_condition_model(run_dir, device)
    cam = GradCAM(model, resolve_target_layer(model))

    per_finding = {c: {"scores": [], "ious": [], "iobbs": [], "hits": []} for c in
                   sorted(set(NIH_BBOX_TO_CANONICAL.values()))}
    pred_area_fracs = []
    records = []  # per-box detail, for cross-condition bucketing (A9b) -- not in the JSON summary

    for row in tqdm(test_boxes.itertuples(), total=len(test_boxes), desc=f"grad-cam [{name}]"):
        canonical = row.canonical
        class_idx = CANONICAL_LABELS.index(canonical)
        img = Image.open(row.image_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)

        heatmap = cam(x, class_idx=class_idx)[0].cpu().numpy()
        with torch.no_grad():
            score = torch.sigmoid(model(x))[0, class_idx].item()

        gt_box = (row.x1, row.y1, row.x2, row.y2)
        pred_box = largest_component_box(heatmap, percentile=90.0)
        iou = box_iou(pred_box, gt_box) if pred_box is not None else 0.0
        iobb = box_iobb(pred_box, gt_box) if pred_box is not None else 0.0
        hit = pointing_game_hit(heatmap, gt_box)
        if pred_box is not None:
            px1, py1, px2, py2 = pred_box
            h, w = heatmap.shape
            pred_area_fracs.append((px2 - px1) * (py2 - py1) / (w * h))

        d = per_finding[canonical]
        d["scores"].append(score)
        d["ious"].append(iou)
        d["iobbs"].append(iobb)
        d["hits"].append(hit)
        records.append({
            "image_id": row.image_id, "canonical": canonical, "image_path": row.image_path,
            "gt_box": gt_box, "pred_box": pred_box, "iou": iou, "iobb": iobb,
        })

    cam.remove()

    per_finding_summary = {}
    for c, d in per_finding.items():
        n = len(d["scores"])
        if n == 0:
            continue
        ap = {f"ap@{t}": average_precision_at_iou(d["scores"], [i >= t for i in d["ious"]])
              for t in IOU_THRESHOLDS}
        per_finding_summary[c] = {
            "n": n,
            "iou": float(np.mean(d["ious"])),
            "iobb": float(np.mean(d["iobbs"])),
            "pointing": float(np.mean(d["hits"])),
            **ap,
        }

    macro = {}
    for key in ("iou", "iobb", "pointing", *[f"ap@{t}" for t in IOU_THRESHOLDS]):
        vals = [v[key] for v in per_finding_summary.values()]
        macro[key] = float(np.mean(vals)) if vals else float("nan")
    macro["n_boxes"] = int(sum(v["n"] for v in per_finding_summary.values()))
    macro["pred_box_area_frac_mean"] = float(np.mean(pred_area_fracs)) if pred_area_fracs else float("nan")
    macro["pred_box_area_frac_p90"] = float(np.percentile(pred_area_fracs, 90)) if pred_area_fracs else float("nan")

    return {"per_finding": per_finding_summary, "macro": macro, "records": records}


BUCKET_ORDER = ("both_miss", "both_hit", "flat_only", "hier_only")


def bucket_agreement_patterns(flat_records: list[dict], hier_records: list[dict]) -> dict:
    """Classify each box by (flat IoU>0, hierarchical IoU>0) into the 4
    agreement patterns Fig. 5 visualizes. Both lists come from
    evaluate_condition() calls over the same test_boxes rows in the same
    order, so zipping by index is safe (checked, not just assumed)."""
    buckets = {k: [] for k in BUCKET_ORDER}
    for f, h in zip(flat_records, hier_records):
        assert f["image_id"] == h["image_id"] and f["canonical"] == h["canonical"], (
            "flat/hierarchical records out of sync -- did they iterate the same test_boxes?"
        )
        combined = {
            "image_id": f["image_id"], "canonical": f["canonical"], "image_path": f["image_path"],
            "gt_box": f["gt_box"],
            "flat_pred_box": f["pred_box"], "flat_iou": f["iou"],
            "hier_pred_box": h["pred_box"], "hier_iou": h["iou"],
        }
        flat_hit, hier_hit = f["iou"] > 0, h["iou"] > 0
        if flat_hit and hier_hit:
            buckets["both_hit"].append(combined)
        elif not flat_hit and not hier_hit:
            buckets["both_miss"].append(combined)
        elif flat_hit:
            buckets["flat_only"].append(combined)
        else:
            buckets["hier_only"].append(combined)
    return buckets


def select_examples(records: list[dict], n: int = 2) -> list[dict]:
    """First file-order occurrence, then the first occurrence from a
    *different* finding if one exists in this bucket, else the next one in
    file order -- matches the archived figure's own selection rule."""
    if not records:
        return []
    selected = [records[0]]
    rest = records[1:]
    if rest:
        different = next((r for r in rest if r["canonical"] != records[0]["canonical"]), None)
        selected.append(different if different is not None else rest[0])
    return selected[:n]


def main() -> None:
    device = torch.device("cpu")  # GPU left free for other work
    test_boxes = load_test_boxes()
    n_images = test_boxes["image_id"].nunique()
    print(f"{len(test_boxes)} usable boxes across {n_images} test images "
          f"(manuscript states 118 across 114)")

    results = {}
    for name, run_dir in CONDITIONS.items():
        results[name] = evaluate_condition(name, run_dir, test_boxes, device)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # per-condition "records" are a side channel for the bucketing below --
    # excluded here so this file stays byte-identical to before that addition.
    results_summary = {name: {"per_finding": r["per_finding"], "macro": r["macro"]}
                        for name, r in results.items()}
    with open(OUT_DIR / "localization_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"n_boxes": len(test_boxes), "n_images": int(n_images), "results": results_summary},
                   f, indent=2)

    print(f"\n{'Condition':<14}{'n boxes':>9}{'IoU':>8}{'IoBB':>8}{'Pointing':>10}{'AP@0.1':>9}"
          f"{'AP@0.25':>9}{'AP@0.5':>9}")
    print("-" * 80)
    for name, r in results.items():
        m = r["macro"]
        print(f"{name:<14}{m['n_boxes']:>9}{m['iou']:>8.4f}{m['iobb']:>8.4f}{100*m['pointing']:>9.1f}%"
              f"{m['ap@0.1']:>9.4f}{m['ap@0.25']:>9.4f}{m['ap@0.5']:>9.4f}")

    print(f"\nWritten: {OUT_DIR / 'localization_metrics.json'}")

    buckets = bucket_agreement_patterns(results["flat"]["records"], results["hierarchical"]["records"])
    assert sum(len(v) for v in buckets.values()) == len(test_boxes), "bucket sizes must sum to all boxes"
    examples = {name: select_examples(recs, n=2) for name, recs in buckets.items()}

    with open(OUT_DIR / "gradcam_bucket_examples.json", "w", encoding="utf-8") as f:
        json.dump({"bucket_sizes": {k: len(v) for k, v in buckets.items()}, "examples": examples},
                   f, indent=2)

    print(f"\nBucket sizes: " + ", ".join(f"{k}={len(v)}" for k, v in buckets.items()))
    print(f"Written: {OUT_DIR / 'gradcam_bucket_examples.json'}")


if __name__ == "__main__":
    main()
