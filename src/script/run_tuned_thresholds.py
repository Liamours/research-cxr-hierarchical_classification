"""A6: validation-tuned per-label thresholds (R2 pt.3). For each label,
picks the threshold maximizing F1 on the val split, then reports test-set
F1 macro under those per-label thresholds vs. the manuscript's fixed 0.5.

Uses the manuscript's exact 25-label Table II scope (excludes Lung_Cancer
and Pleural_Empyema, too few positives for a valid split) as the primary
number, and separately flags labels where tuning is not meaningful: if
every candidate threshold ties at F1=0 on validation (too few val
positives to ever produce a true positive), the "best" threshold is an
arbitrary tie-break, not a real optimum. Those labels default to 0.5
instead, so they contribute identically under both the fixed and tuned
scheme by construction rather than by coincidence.

    uv run python src/script/run_tuned_thresholds.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.data.label_space import CANONICAL_LABELS

CANDIDATES = np.arange(0.05, 0.96, 0.01)
EXCLUDED_MANUSCRIPT = {"Lung_Cancer", "Pleural_Empyema"}  # Table II's own exclusion


def _load(csv_path: Path):
    df = pd.read_csv(csv_path)
    probs = np.stack([df[f"prob_{c}"].to_numpy() for c in CANONICAL_LABELS], axis=1)
    targets = np.stack([df[f"label_{c}"].to_numpy() for c in CANONICAL_LABELS], axis=1)
    mask = np.stack([df[f"mask_{c}"].to_numpy() for c in CANONICAL_LABELS], axis=1)
    return probs, targets, mask


def tune_thresholds(val_probs, val_targets, val_mask) -> tuple[dict, set]:
    """Returns (thresholds, degenerate_labels). A label is degenerate when
    no candidate threshold beats F1=0 on validation -- tuning has no signal
    to act on, so it's left at 0.5 rather than an arbitrary tie-break."""
    thresholds, degenerate = {}, set()
    for j, c in enumerate(CANONICAL_LABELS):
        sel = val_mask[:, j] == 1
        if sel.sum() == 0 or len(np.unique(val_targets[sel, j])) < 2:
            thresholds[c] = 0.5
            continue
        y, p = val_targets[sel, j], val_probs[sel, j]
        f1s = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in CANDIDATES]
        if max(f1s) == 0.0:
            thresholds[c] = 0.5
            degenerate.add(c)
        else:
            thresholds[c] = float(CANDIDATES[int(np.argmax(f1s))])
    return thresholds, degenerate


def f1_per_label(probs, targets, mask, thresholds: dict) -> dict:
    out = {}
    for j, c in enumerate(CANONICAL_LABELS):
        sel = mask[:, j] == 1
        if sel.sum() == 0:
            out[c] = float("nan")
        else:
            pred = (probs[sel, j] >= thresholds[c]).astype(int)
            out[c] = float(f1_score(targets[sel, j], pred, zero_division=0))
    return out


def macro(d: dict, exclude: set) -> tuple[float, int]:
    vals = [v for k, v in d.items() if k not in exclude and not np.isnan(v)]
    return (float(np.mean(vals)) if vals else float("nan")), len(vals)


def run_condition(name: str, val_csv: Path, test_csv: Path) -> None:
    vp, vt, vm = _load(val_csv)
    tp, tt, tm = _load(test_csv)

    thresholds, degenerate = tune_thresholds(vp, vt, vm)
    f1_fixed_dict = f1_per_label(tp, tt, tm, {c: 0.5 for c in CANONICAL_LABELS})
    f1_tuned_dict = f1_per_label(tp, tt, tm, thresholds)

    m_fixed, n = macro(f1_fixed_dict, EXCLUDED_MANUSCRIPT)
    m_tuned, _ = macro(f1_tuned_dict, EXCLUDED_MANUSCRIPT)

    print(f"\n=== {name} (manuscript's exact {n}-label Table II scope) ===")
    print(f"F1 macro, fixed 0.5:        {m_fixed:.4f}")
    print(f"F1 macro, val-tuned/label:  {m_tuned:.4f}  (delta {m_tuned - m_fixed:+.4f})")
    print(f"Un-tunable labels (every threshold ties at F1=0 on val, left at 0.5): "
          f"{len(degenerate - EXCLUDED_MANUSCRIPT)} -> {sorted(degenerate - EXCLUDED_MANUSCRIPT)}")

    tuned_real = {c: t for c, t in thresholds.items()
                  if c not in EXCLUDED_MANUSCRIPT and c not in degenerate and abs(t - 0.5) > 1e-9}
    print(f"Meaningfully tuned labels (threshold != 0.5, not a tie-break): {len(tuned_real)}")
    for c, t in sorted(tuned_real.items(), key=lambda kv: -abs(kv[1] - 0.5))[:10]:
        print(f"  {c:<38} threshold={t:.2f}  test_f1={f1_tuned_dict[c]:.4f}")


def main() -> None:
    base = Path(r"C:\rifqi\research-cxr-hierarchical_classification\inference_results")
    run_condition("flat", base / "predictions-densenet121_xrv-flat-260719" / "val.csv",
                  base / "predictions-densenet121_xrv-flat-260719" / "test.csv")
    run_condition("hierarchical", base / "predictions-densenet121_xrv-hierarchical-260718" / "val.csv",
                  base / "predictions-densenet121_xrv-hierarchical-260718" / "test.csv")


if __name__ == "__main__":
    main()
