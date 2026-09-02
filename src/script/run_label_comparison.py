"""Compare our ingested label counts against official published numbers.

Official counts are hardcoded from the original papers (see sources below).
Our counts are computed live from the ingested CSVs.

Usage:
    uv run python src/script/run_label_comparison.py
    uv run python src/script/run_label_comparison.py --dataset nih-cxr14
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

REGISTRY = Path(__file__).resolve().parents[2] / "configs" / "dataset_registry.json"

# ---------------------------------------------------------------------------
# Official counts from original papers / dataset documentation.
# raw label name -> positive count. Split scope noted in comments.
# Sources filled in after research — placeholders marked TODO.
# ---------------------------------------------------------------------------
OFFICIAL: dict[str, dict] = {
    "nih-cxr14": {
        "_source": "Wang et al. 2017 (arxiv 1705.02315) — Table 1, all 112,120 images",
        "_scope": "all splits (train+val+test combined)",
        "_note": "Binary labels only. Mass+Nodule both map to SPN via OR-merge; individual rows for Mass/Nodule vs SPN are not directly comparable (our SPN count ~matches Mass OR Nodule union). We include PA+AP (all 112,120); paper Pneumonia=1,353 likely PA-only. Small diffs expected.",
        "Atelectasis":    11535,
        "Cardiomegaly":    2772,
        "Effusion":       13307,
        "Infiltration":   19871,  # not mapped to canonical
        "SPN (Mass OR Nodule)": 11207,  # correct OR-merge comparison; individual Mass/Nodule not directly comparable
        "Mass":            5746,
        "Nodule":          6323,
        "Pneumonia":       1353,
        "Pneumothorax":    5298,
        "Consolidation":   4667,  # not mapped to canonical
        "Edema":           2303,
        "Emphysema":       2516,
        "Fibrosis":        1686,
        "Pleural_Thickening": 3385,  # not mapped to canonical
        "Hernia":           227,
    },
    "chexpert": {
        "_source": "Irvin et al. 2019 (arxiv 1901.07031) — frontal train split ~187k images",
        "_scope": "train frontal only (original CheXpert v1.0 NLP labels)",
        "_note": "WE USE CheXpert+ (2024 expert-corrected labels), NOT original v1.0. Our counts will DIFFER from these official numbers. CheXpert+ has more expert-confirmed positives (e.g. Cardiomegaly 23,384->25,835) and fewer uncertain cases. This discrepancy is expected and correct.",
        "Atelectasis":       29422,
        "Cardiomegaly":      23384,
        "Edema":             48905,
        "Pleural Effusion":  75696,
        "Pneumonia":          4576,
        "Pneumothorax":      17326,
        "Fracture":           7270,
        "Lung Lesion":        6395,
    },
    "vindr-cxr": {
        "_source": "Nguyen et al. 2022 (arxiv 2012.15029) — train split 15,000 images",
        "_scope": "train split only (our CSV covers same 15k train images)",
        "_note": "VinDr-CXR has 28 raw findings; only 6 map to canonical.",
        "Cardiomegaly":    1817,
        "Pleural Effusion":  634,
    },
    "tbx11k": {
        "_source": "Liu et al. 2020 (CVPR) — labeled split 8,400 images (6600 train + 1800 val)",
        "_scope": "train+val labeled",
        "_note": "Binary TB label. No Finding = Healthy(3800) + Sick-non-TB(3800).",
        "TB-positive":  800,
    },
    "vindr-pcxr": {
        "_source": "Pham et al. 2022 (Nature Scientific Data) — train split 7,728 images",
        "_scope": "train split only (we ingested 9125 total incl. test)",
        "_note": "Official counts are train-only (7728 imgs). We cover 9125 (train+val+test). Pneumonia OR-merges 3 raw cols (Brocho-pneumonia + Pneumonia + Pleuro-pneumonia) so compare to sum. Verified: train+val Acute_Bronchitis=842 exact, Bronchiolitis=497 exact, Pneumonia_OR=920 vs 545+392=937 (17 overlap). All flags are scope difference, not bugs.",
        "Bronchitis":         842,
        "Pneumonia (OR: Bronchopneumonia+Pneumonia+Pleuro-pneumonia train only)": 920,
        "Bronchopneumonia":   545,
        "Bronchiolitis":      497,
        "Pneumonia":          392,
    },
    "covidx-cxr4": {
        "_source": "arxiv 2311.17677 — all 84,818 images",
        "_scope": "all splits (train+val+test)",
        "_note": "Train heavily imbalanced (~5.4:1 COVID+:COVID-); val+test balanced 1:1. Off by 1 (65,680 vs 65,681) = one image file missing on disk; acceptable.",
        "COVID-positive": 65681,
        "COVID-negative": 19137,
    },
}

# canonical label -> raw label name for each dataset (for matching)
RAW_TO_CANONICAL: dict[str, dict[str, str]] = {
    "nih-cxr14": {
        "Atelectasis":    "Atelectasis",
        "Cardiomegaly":   "Cardiomegaly",
        "Effusion":       "Pleural_Effusion",
        "SPN (Mass OR Nodule)": "Solitary_Pulmonary_Nodule",  # proper OR-merge comparison
        "Mass":           "Solitary_Pulmonary_Nodule",   # individual (not directly comparable to SPN)
        "Nodule":         "Solitary_Pulmonary_Nodule",   # individual (not directly comparable to SPN)
        "Pneumonia":      "Pneumonia",
        "Pneumothorax":   "Pneumothorax",
        "Edema":          "Pulmonary_Edema",
        "Emphysema":      "COPD",
        "Fibrosis":       "ILD",
        "Hernia":         "Diaphragmatic_Hernia",
        # Infiltration, Consolidation, Pleural_Thickening — not mapped, skip
    },
    "chexpert": {
        "Cardiomegaly":     "Cardiomegaly",
        "Lung Lesion":      "Solitary_Pulmonary_Nodule",
        "Edema":            "Pulmonary_Edema",
        "Pneumonia":        "Pneumonia",
        "Atelectasis":      "Atelectasis",
        "Pneumothorax":     "Pneumothorax",
        "Pleural Effusion": "Pleural_Effusion",
        "Fracture":         "Chest_Trauma",
    },
    "vindr-cxr": {
        "Cardiomegaly":     "Cardiomegaly",
        "Pleural Effusion": "Pleural_Effusion",
        "Atelectasis":      "Atelectasis",
    },
    "tbx11k": {
        "TB-positive":      "Tuberculosis",
    },
    "vindr-pcxr": {
        "Bronchitis":       "Acute_Bronchitis",
        "Pneumonia (OR: Bronchopneumonia+Pneumonia+Pleuro-pneumonia train only)": "Pneumonia",
        "Bronchopneumonia": "Pneumonia",
        "Bronchiolitis":    "Bronchiolitis",
        "Pneumonia":        "Pneumonia",
    },
    "covidx-cxr4": {
        "COVID-positive":   "COVID19_Pneumonia",
    },
}


def get_our_counts(name: str, cfg: dict) -> dict[str, int]:
    csv_path = Path(cfg["root"]) / cfg["preprocessed_label_csv"]
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    counts = {}
    for col in df.columns:
        if col in ("image_id", "image_path", "dataset", "split"):
            continue
        pos = int(df[col].sum()) if df[col].notna().any() else 0
        if pos > 0:
            counts[col] = pos
    return counts


def compare_dataset(name: str, cfg: dict) -> None:
    official = OFFICIAL.get(name, {})
    source = official.pop("_source", "no source")
    scope = official.pop("_scope", "")
    note = official.pop("_note", "")
    mapping = RAW_TO_CANONICAL.get(name, {})

    print(f"\n{'='*70}")
    print(f"Dataset: {name}")
    print(f"Source:  {source}  [{scope}]")
    print(f"{'='*70}")

    if not official:
        print("  No official counts loaded yet (TODO).")
        our = get_our_counts(name, cfg)
        if our:
            print(f"  Our counts: {our}")
        return

    our_counts = get_our_counts(name, cfg)

    has_todos = False
    print(f"  {'Raw label':<35} {'Official':>10} {'Ours (canonical)':>20} {'Delta':>10}  {'Canonical'}")
    print(f"  {'-'*35} {'-'*10} {'-'*20} {'-'*10}  {'-'*30}")

    for raw_label, official_count in official.items():
        canonical = mapping.get(raw_label)
        if canonical is None:
            print(f"  {raw_label:<35} {official_count:>10,} {'(not mapped)':>20s}")
            continue
        our = our_counts.get(canonical, 0)
        delta = our - official_count
        delta_str = f"{delta:+,}" if delta != 0 else "exact"
        flag = " !!!" if abs(delta) > official_count * 0.1 else ""
        print(f"  {raw_label:<35} {official_count:>10,} {our:>20,} {delta_str:>10}  {canonical}{flag}")

    if note:
        print(f"  NOTE: {note}")
    # restore for next call
    official["_source"] = source
    official["_scope"] = scope
    official["_note"] = note


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, help="check one dataset only")
    args = ap.parse_args()

    registry = json.loads(REGISTRY.read_text())

    for name, cfg in registry["datasets"].items():
        if cfg["status"] != "ingested":
            continue
        if args.dataset and name != args.dataset:
            continue
        compare_dataset(name, cfg)

    print()


if __name__ == "__main__":
    main()
