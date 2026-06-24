"""Calibration: Expected Calibration Error and reliability-diagram bin data.

For multi-label CXR, each masked (sample, condition) prediction is treated as a
binary calibration problem on the positive-class probability. Predictions are
binned by predicted probability; per bin we record the count, mean predicted
confidence, and empirical positive rate. ECE is the count-weighted mean gap
between confidence and empirical rate. Masked entries are excluded.
"""

from __future__ import annotations

import numpy as np

from src.data.label_space import CANONICAL_LABELS
from src.evaluate.metrics import _flat_masked, _np


def reliability_bins(probs, targets, mask, n_bins: int = 15) -> dict:
    p, y = _flat_masked(probs, targets, mask)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    bins = []
    for b in range(n_bins):
        m = idx == b
        count = int(m.sum())
        bins.append({
            "bin_lo": float(edges[b]),
            "bin_hi": float(edges[b + 1]),
            "count": count,
            "avg_conf": float(p[m].mean()) if count else 0.0,
            "avg_acc": float(y[m].mean()) if count else 0.0,
        })
    return {"bins": bins, "n": int(len(p))}


def expected_calibration_error(probs, targets, mask, n_bins: int = 15) -> float:
    rb = reliability_bins(probs, targets, mask, n_bins)
    n = rb["n"]
    if n == 0:
        return float("nan")
    return float(sum(b["count"] / n * abs(b["avg_conf"] - b["avg_acc"]) for b in rb["bins"]))


def per_class_ece(probs, targets, mask, conditions=CANONICAL_LABELS, n_bins: int = 15) -> dict:
    probs, targets, mask = _np(probs), _np(targets), _np(mask)
    out = {}
    for j, c in enumerate(conditions):
        col_mask = np.zeros_like(mask)
        col_mask[:, j] = mask[:, j]
        out[c] = expected_calibration_error(probs, targets, col_mask, n_bins)
    return out


def compute_calibration(probs, targets, mask, n_bins: int = 15,
                        conditions=CANONICAL_LABELS) -> dict:
    return {
        "ece": expected_calibration_error(probs, targets, mask, n_bins),
        "reliability": reliability_bins(probs, targets, mask, n_bins),
        "per_class_ece": per_class_ece(probs, targets, mask, conditions, n_bins=n_bins),
    }
