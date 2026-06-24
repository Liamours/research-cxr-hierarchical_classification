"""Evaluation orchestrator.

evaluate_predictions: pure compute over (probs, targets, mask) -> report dict.
evaluate_model: runs the trained model over a loader (eval mode, no_grad, no
weight updates) to gather predictions, then evaluates them and writes
eval_metrics_<split>.json into the run directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from src.data.label_space import CANONICAL_LABELS
from src.evaluate.calibration import compute_calibration
from src.evaluate.metrics import compute_classification_metrics, compute_map
from src.evaluate.selective import compute_aurc, coverage_accuracy
from src.model.mc_dropout import mc_dropout_predict, uncertainty_sanity_check


def evaluate_predictions(probs, targets, mask, conditions=CANONICAL_LABELS,
                         threshold=0.5, n_bins=15, metrics=None) -> dict:
    metrics = metrics or ["auroc", "f1", "ece"]
    report: dict = {}
    if "auroc" in metrics or "f1" in metrics:
        cm = compute_classification_metrics(probs, targets, mask, conditions, threshold)
        if "auroc" in metrics:
            report["auroc"] = cm["auroc"]
        if "f1" in metrics:
            report["f1"] = cm["f1"]
    if "ece" in metrics:
        report["calibration"] = compute_calibration(probs, targets, mask, n_bins, conditions)
    if "selective" in metrics:
        report["coverage_accuracy"] = coverage_accuracy(probs, targets, mask, threshold=threshold)
    if "map" in metrics:
        report["map"] = compute_map(probs, targets, mask, conditions)
    if "aurc" in metrics:
        report["aurc"] = compute_aurc(probs, targets, mask, conditions, threshold)
    return report


@torch.no_grad()
def gather_predictions(model, loader, device, use_amp: bool = False, mc_passes: int = 0):
    """Returns (image_ids, probs, targets, mask, var). var is None unless mc_passes > 0."""
    model.eval()
    image_ids = []
    probs, targets, masks, varis = [], [], [], []
    for batch in tqdm(loader, desc="eval gather", dynamic_ncols=True):
        x = batch["pixel_values"].to(device)
        image_ids.extend(batch["image_id"])
        if mc_passes > 0:
            mean, var = mc_dropout_predict(model, x, mc_passes)
            probs.append(mean.float().cpu())
            varis.append(var.float().cpu())
        else:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits = model(x)
            probs.append(torch.sigmoid(logits).float().cpu())
        targets.append(batch["labels"])
        masks.append(batch["label_mask"])
    return (
        image_ids,
        torch.cat(probs), torch.cat(targets), torch.cat(masks),
        torch.cat(varis) if varis else None,
    )


def save_predictions(image_ids, probs, targets, mask, conditions, var, out_path) -> None:
    """Writes one row per sample: image_id, then prob_/label_/mask_<condition>
    (+ var_<condition> when MC-Dropout variance is available)."""
    data: dict = {"image_id": image_ids}
    for j, c in enumerate(conditions):
        data[f"prob_{c}"] = probs[:, j].numpy()
        data[f"label_{c}"] = targets[:, j].numpy()
        data[f"mask_{c}"] = mask[:, j].numpy()
        if var is not None:
            data[f"var_{c}"] = var[:, j].numpy()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(out_path, index=False)


def _summary(report: dict) -> dict:
    s = {}
    if "auroc" in report:
        s["auroc_macro"] = report["auroc"]["macro"]
        s["auroc_micro"] = report["auroc"]["micro"]
    if "f1" in report:
        s["f1_macro"] = report["f1"]["macro"]
        s["f1_micro"] = report["f1"]["micro"]
    if "calibration" in report:
        s["ece"] = report["calibration"]["ece"]
    if "map" in report:
        s["map_macro"] = report["map"]["macro"]
    if "aurc" in report:
        s["aurc_macro"] = report["aurc"]["macro"]
        s["aurc_flat"] = report["aurc"]["flat"]
    return s


def evaluate_model(model, loader, device, cfg, split: str = "val", logger=None) -> dict:
    use_amp = cfg.training.bf16 and device.type == "cuda"
    mc_passes = cfg.uq.mc_passes if cfg.uq.method == "mc_dropout" else 0
    threshold = cfg.eval.threshold
    conditions = cfg.resolved_conditions()
    image_ids, probs, targets, mask, var = gather_predictions(
        model, loader, device, use_amp, mc_passes
    )

    # Hierarchical fallback: suppress uncertain child labels when both MC-Dropout
    # and hierarchical label structure are active.
    fallback_log = None
    if var is not None and cfg.label.label_structure == "hierarchical":
        from src.data.hierarchy import edge_index_pairs
        from src.inference.hierarchical_fallback import apply_hierarchical_fallback, fallback_summary
        pairs = edge_index_pairs(conditions)
        probs, fallback_log = apply_hierarchical_fallback(
            probs, var, pairs, gate_threshold=cfg.uq.gate_threshold
        )
        if logger is not None:
            logger.log(fallback_summary(fallback_log))

    report = evaluate_predictions(
        probs, targets, mask,
        conditions=conditions,
        threshold=threshold, n_bins=cfg.eval.reliability_bins, metrics=cfg.eval.metrics,
    )
    summary = _summary(report)

    if var is not None:
        msel = mask > 0
        sane = uncertainty_sanity_check(probs, var, targets, mask, threshold)
        report["uq"] = {
            "mc_passes": mc_passes,
            "mean_variance": float(var[msel].mean()) if msel.any() else float("nan"),
            "sanity": sane,
        }
        if fallback_log is not None:
            report["uq"]["hierarchical_fallback"] = fallback_log
        summary["uq_mean_variance"] = report["uq"]["mean_variance"]
        summary["uq_sanity_passes"] = sane["passes"]
        if fallback_log is not None:
            summary["uq_fallback_suppressed"] = fallback_log["total_suppressed"]

    run_dir = cfg.run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / f"eval_metrics_{split}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"split": split, "summary": summary, "report": report}, f, indent=2, default=str)

    save_predictions(image_ids, probs, targets, mask, conditions, var,
                      run_dir / "predictions" / f"{split}.csv")

    if logger is not None:
        logger.event("eval", split=split, **summary)
    return report
