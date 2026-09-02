"""Item 3: validation-tuned per-label thresholds vs. the fixed 0.5 baseline,
for every trained condition from the 2026-08-30 batch (seeds 42/43/44,
backbone ablation, soft-penalty). Same method as run_tuned_thresholds.py
(the manuscript's original A6 script, kept as-is for the two originally
reported checkpoints); this one loops over the full set instead.

    uv run python src/script/run_tuned_thresholds_all.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.data.label_space import CANONICAL_LABELS
from src.script._conditions import available, missing

CANDIDATES = np.arange(0.05, 0.96, 0.01)
EXCLUDED_MANUSCRIPT = {"Lung_Cancer", "Pleural_Empyema"}  # Table II's own exclusion


def _load(csv_path: Path):
    df = pd.read_csv(csv_path)
    probs = np.stack([df[f"prob_{c}"].to_numpy() for c in CANONICAL_LABELS], axis=1)
    targets = np.stack([df[f"label_{c}"].to_numpy() for c in CANONICAL_LABELS], axis=1)
    mask = np.stack([df[f"mask_{c}"].to_numpy() for c in CANONICAL_LABELS], axis=1)
    return probs, targets, mask


def tune_thresholds(val_probs, val_targets, val_mask) -> tuple[dict, set]:
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


def run_condition(name: str, val_csv: Path, test_csv: Path) -> dict:
    vp, vt, vm = _load(val_csv)
    tp, tt, tm = _load(test_csv)

    thresholds, degenerate = tune_thresholds(vp, vt, vm)
    f1_fixed_dict = f1_per_label(tp, tt, tm, {c: 0.5 for c in CANONICAL_LABELS})
    f1_tuned_dict = f1_per_label(tp, tt, tm, thresholds)

    m_fixed, n = macro(f1_fixed_dict, EXCLUDED_MANUSCRIPT)
    m_tuned, _ = macro(f1_tuned_dict, EXCLUDED_MANUSCRIPT)

    print(f"\n=== {name} ({n}-label scope) ===")
    print(f"F1 macro, fixed 0.5:        {m_fixed:.4f}")
    print(f"F1 macro, val-tuned/label:  {m_tuned:.4f}  (delta {m_tuned - m_fixed:+.4f})")
    print(f"Un-tunable labels: {len(degenerate - EXCLUDED_MANUSCRIPT)}")
    return {"name": name, "n_labels": n, "f1_fixed": m_fixed, "f1_tuned": m_tuned,
            "delta": m_tuned - m_fixed}


def main() -> None:
    conds = available()
    miss = missing()
    if miss:
        print(f"Skipping (predictions not ready yet): {miss}")
    rows = [run_condition(name, val, test) for name, (val, test) in conds.items()]

    print(f"\n{'condition':<32}{'n':>4}{'fixed':>10}{'tuned':>10}{'delta':>10}")
    print("-" * 66)
    for r in rows:
        print(f"{r['name']:<32}{r['n_labels']:>4}{r['f1_fixed']:>10.4f}{r['f1_tuned']:>10.4f}{r['delta']:>+10.4f}")


if __name__ == "__main__":
    main()
