"""Cross-check whether the HBCE penalty actually moved per-edge behavior.

Compares flat vs hierarchical test-set predictions: per-edge violation rate
(model behavior) against per-edge child-mask coverage in that same test split
(data availability), to see which edges the hierarchical run actually shows
a training effect on.

    uv run python src/script/run_edge_signal_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.data.hierarchy import PARENT_CHILD_EDGES
from src.data.label_space import CANONICAL_LABELS
from src.evaluate.metrics import hierarchy_violation_rate

FLAT_CSV = Path(r"C:\rifqi\research-cxr-hierarchical_classification\inference_results\predictions-densenet121_xrv-flat-260719\test.csv")
HIER_CSV = Path(r"C:\rifqi\research-cxr-hierarchical_classification\inference_results\predictions-densenet121_xrv-hierarchical-260718\test.csv")


def probs_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.stack([df[f"prob_{c}"].to_numpy() for c in CANONICAL_LABELS], axis=1)


def main() -> None:
    flat = pd.read_csv(FLAT_CSV)
    hier = pd.read_csv(HIER_CSV)
    assert len(flat) == len(hier), f"row count mismatch: flat={len(flat)} hier={len(hier)}"
    n = len(flat)

    flat_hcv = hierarchy_violation_rate(probs_matrix(flat), CANONICAL_LABELS, 0.5)
    hier_hcv = hierarchy_violation_rate(probs_matrix(hier), CANONICAL_LABELS, 0.5)

    print(f"test split: {n:,} samples\n")
    print(f"overall HCV: flat={flat_hcv['rate']*100:.3f}%  hierarchical={hier_hcv['rate']*100:.3f}%\n")

    print(f"{'edge':<44}{'child_ann(test)':>16}{'flat_viol':>12}{'hier_viol':>12}{'delta':>10}")
    print("-" * 96)
    for parent, child in PARENT_CHILD_EDGES:
        key = f"{parent}>{child}"
        child_ann = int(flat[f"mask_{child}"].sum()) if f"mask_{child}" in flat.columns else -1
        fv = flat_hcv["per_edge"].get(key, float("nan"))
        hv = hier_hcv["per_edge"].get(key, float("nan"))
        delta = hv - fv
        flag = "  <-- credited edge" if (parent, child) == ("Pneumonia", "COVID19_Pneumonia") else \
               ("  <-- newly found active" if child_ann > 0 else "")
        print(f"{key:<44}{child_ann:>16,}{fv*100:>11.3f}%{hv*100:>11.3f}%{delta*100:>9.3f}%{flag}")


if __name__ == "__main__":
    main()
