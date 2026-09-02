"""A3: source-stratified test performance -- per-dataset AUROC/F1 and
positive counts, plus the 3-pretraining-overlap (nih-cxr14, padchest,
chexpert) vs 4-held-out (tbx11k, vindr-cxr, vindr-pcxr, covidx-cxr4)
aggregate split. Answers R1 pt.6, R2 pt.2/pt.7, R4 pt.2 -- R2 pt.7 asks
for per-source positive counts alongside the metrics, not metrics alone.

    uv run python src/script/run_source_stratified.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.data.label_space import CANONICAL_LABELS
from src.evaluate.metrics import _macro, per_class_auroc, per_class_f1

COMBINED_CSV = Path(r"C:\rifqi\research-cxr-hierarchical_classification\dataset\combined\combined.csv")
FLAT_CSV = Path(r"C:\rifqi\research-cxr-hierarchical_classification\inference_results\predictions-densenet121_xrv-flat-260719\test.csv")
HIER_CSV = Path(r"C:\rifqi\research-cxr-hierarchical_classification\inference_results\predictions-densenet121_xrv-hierarchical-260718\test.csv")

PRETRAIN_OVERLAP = {"nih-cxr14", "padchest", "chexpert"}
HELD_OUT = {"tbx11k", "vindr-cxr", "vindr-pcxr", "covidx-cxr4"}


def _load(csv_path: Path, id_to_dataset: dict) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["dataset"] = df["image_id"].map(id_to_dataset)
    missing = df["dataset"].isna().sum()
    if missing:
        print(f"  WARNING: {missing}/{len(df)} image_ids not found in combined.csv")
    return df


def _metrics_for(df: pd.DataFrame) -> dict:
    n = len(df)
    if n == 0:
        return {"n": 0, "auroc_macro": float("nan"), "f1_macro": float("nan"), "n_pos": 0}
    probs = np.stack([df[f"prob_{c}"].to_numpy() for c in CANONICAL_LABELS], axis=1)
    targets = np.stack([df[f"label_{c}"].to_numpy() for c in CANONICAL_LABELS], axis=1)
    mask = np.stack([df[f"mask_{c}"].to_numpy() for c in CANONICAL_LABELS], axis=1)
    auroc = _macro(per_class_auroc(probs, targets, mask, CANONICAL_LABELS))
    f1 = _macro(per_class_f1(probs, targets, mask, CANONICAL_LABELS, threshold=0.5))
    # R2 pt.7: positive counts alongside per-source metrics, not metrics alone.
    n_pos = int(np.nansum(np.where(mask == 1, targets, 0)))
    return {"n": n, "auroc_macro": auroc, "f1_macro": f1, "n_pos": n_pos}


def main() -> None:
    combined = pd.read_csv(COMBINED_CSV, usecols=["image_id", "dataset"])
    id_to_dataset = dict(zip(combined["image_id"], combined["dataset"]))

    print("Loading predictions...")
    flat = _load(FLAT_CSV, id_to_dataset)
    hier = _load(HIER_CSV, id_to_dataset)

    datasets = sorted(flat["dataset"].dropna().unique())
    print(f"\n{'dataset':<16}{'role':<10}{'n':>8}{'n_pos':>8}{'flat_auroc':>12}{'hier_auroc':>12}{'flat_f1':>10}{'hier_f1':>10}")
    print("-" * 88)
    for ds in datasets:
        role = "overlap" if ds in PRETRAIN_OVERLAP else ("held-out" if ds in HELD_OUT else "?")
        fm = _metrics_for(flat[flat["dataset"] == ds])
        hm = _metrics_for(hier[hier["dataset"] == ds])
        # n_pos is identical flat vs hierarchical (same ground truth) -- report once.
        print(f"{ds:<16}{role:<10}{fm['n']:>8}{fm['n_pos']:>8}{fm['auroc_macro']:>12.4f}{hm['auroc_macro']:>12.4f}"
              f"{fm['f1_macro']:>10.4f}{hm['f1_macro']:>10.4f}")

    print(f"\n{'aggregate':<16}{'role':<10}{'n':>8}{'n_pos':>8}{'flat_auroc':>12}{'hier_auroc':>12}{'flat_f1':>10}{'hier_f1':>10}")
    print("-" * 88)
    for role, dsset in (("overlap (3)", PRETRAIN_OVERLAP), ("held-out (4)", HELD_OUT)):
        fsub = flat[flat["dataset"].isin(dsset)]
        hsub = hier[hier["dataset"].isin(dsset)]
        fm, hm = _metrics_for(fsub), _metrics_for(hsub)
        print(f"{'':<16}{role:<10}{fm['n']:>8}{fm['n_pos']:>8}{fm['auroc_macro']:>12.4f}{hm['auroc_macro']:>12.4f}"
              f"{fm['f1_macro']:>10.4f}{hm['f1_macro']:>10.4f}")


if __name__ == "__main__":
    main()
