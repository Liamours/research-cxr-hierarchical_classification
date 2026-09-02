"""Item 4: per-source test performance (all data) and the pretraining-overlap
vs. held-out ("untrained") split, for every trained condition from the
2026-08-30 batch. Same method as run_source_stratified.py (the original A3
script, kept as-is for the two originally reported checkpoints); this one
loops over the full set instead.

"Untrained" test data = the 4 datasets never seen during backbone
pretraining (tbx11k, vindr-cxr, vindr-pcxr, covidx-cxr4), same definition
as A3 -- the closest thing to genuinely unseen data this project can
measure, since pooled fine-tuning itself always uses the full training
split.

    uv run python src/script/run_source_stratified_all.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.data.label_space import CANONICAL_LABELS
from src.evaluate.metrics import _macro, per_class_auroc, per_class_f1
from src.script._conditions import available, missing

COMBINED_CSV = Path(r"C:\rifqi\research-cxr-hierarchical_classification\dataset\combined\combined.csv")
PRETRAIN_OVERLAP = {"nih-cxr14", "padchest", "chexpert"}
HELD_OUT = {"tbx11k", "vindr-cxr", "vindr-pcxr", "covidx-cxr4"}


def _load(csv_path: Path, id_to_dataset: dict) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["dataset"] = df["image_id"].map(id_to_dataset)
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
    n_pos = int(np.nansum(np.where(mask == 1, targets, 0)))
    return {"n": n, "auroc_macro": auroc, "f1_macro": f1, "n_pos": n_pos}


def main() -> None:
    combined = pd.read_csv(COMBINED_CSV, usecols=["image_id", "dataset"])
    id_to_dataset = dict(zip(combined["image_id"], combined["dataset"]))

    conds = available()
    miss = missing()
    if miss:
        print(f"Skipping (predictions not ready yet): {miss}")

    print(f"\n{'condition':<32}{'role':<10}{'n':>8}{'n_pos':>8}{'auroc':>9}{'f1':>9}")
    print("-" * 76)
    for name, (_, test_csv) in conds.items():
        df = _load(test_csv, id_to_dataset)
        overlap = df[df["dataset"].isin(PRETRAIN_OVERLAP)]
        held_out = df[df["dataset"].isin(HELD_OUT)]
        all_m = _metrics_for(df)
        ov_m = _metrics_for(overlap)
        ho_m = _metrics_for(held_out)
        print(f"{name:<32}{'all':<10}{all_m['n']:>8}{all_m['n_pos']:>8}"
              f"{all_m['auroc_macro']:>9.4f}{all_m['f1_macro']:>9.4f}")
        print(f"{'':<32}{'overlap':<10}{ov_m['n']:>8}{ov_m['n_pos']:>8}"
              f"{ov_m['auroc_macro']:>9.4f}{ov_m['f1_macro']:>9.4f}")
        print(f"{'':<32}{'untrained':<10}{ho_m['n']:>8}{ho_m['n_pos']:>8}"
              f"{ho_m['auroc_macro']:>9.4f}{ho_m['f1_macro']:>9.4f}")


if __name__ == "__main__":
    main()
