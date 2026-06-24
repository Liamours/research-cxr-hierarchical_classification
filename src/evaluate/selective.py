"""Selective prediction metrics: risk-coverage curves and AURC.

coverage_accuracy — basic risk-coverage curve (n_points buckets).

aurc_flat / per_class_aurc — Area Under the Risk-Coverage Curve using the
    Zhou et al. 2025 (ICML, PMLR 267) plug-in estimator â'_i:

        â'_i = -ln(1 - r_i / (n+1))

    where r_i is the rank of sample i sorted *ascending* by confidence
    (r_i=1 = least confident). The weight â'_i is large for high-confidence
    samples, so errors on confident predictions are penalised more heavily.

    Convergence rate O(sqrt(ln(n)/n)) — better than naive empirical AURC
    O(n^-1/2). Confidence score: CSF = max(p, 1-p) ∈ [0.5, 1].

    Lower AURC is better: a model that is wrong on low-confidence predictions
    (and right on high-confidence ones) scores lower.
"""

from __future__ import annotations

import numpy as np

from src.data.label_space import CANONICAL_LABELS
from src.evaluate.metrics import _flat_masked, _np, _macro


# ---------------------------------------------------------------------------
# Coverage-accuracy (original implementation, unchanged)
# ---------------------------------------------------------------------------

def coverage_accuracy(probs, targets, mask, n_points: int = 20,
                      threshold: float = 0.5) -> dict:
    p, y = _flat_masked(probs, targets, mask)
    n = len(p)
    if n == 0:
        return {"curve": [], "n": 0}
    conf = np.maximum(p, 1.0 - p)
    correct = ((p >= threshold).astype(int) == y).astype(float)
    order = np.argsort(-conf)
    sorted_correct = correct[order]
    curve = []
    for frac in np.linspace(1.0 / n_points, 1.0, n_points):
        k = max(int(round(frac * n)), 1)
        acc = float(sorted_correct[:k].mean())
        curve.append({"coverage": round(k / n, 4), "accuracy": round(acc, 4),
                      "risk": round(1.0 - acc, 4)})
    return {"curve": curve, "n": int(n)}


# ---------------------------------------------------------------------------
# AURC — Zhou et al. 2025 plug-in estimator
# ---------------------------------------------------------------------------

def _aurc_from_arrays(p: np.ndarray, y: np.ndarray,
                      threshold: float = 0.5) -> float:
    """Compute AURC plug-in estimate â' for a 1-D array of (prob, target) pairs."""
    n = len(p)
    if n < 2:
        return float("nan")
    conf = np.maximum(p, 1.0 - p)                         # CSF ∈ [0.5, 1]
    pred_err = ((p >= threshold).astype(int) != y.astype(int)).astype(float)
    order = np.argsort(conf)                               # ascending → rank 1 = least confident
    sorted_err = pred_err[order]
    ranks = np.arange(1, n + 1, dtype=np.float64)
    weights = -np.log(1.0 - ranks / (n + 1))              # â'_i = -ln(1 - r_i/(n+1))
    return float(np.dot(weights, sorted_err) / n)


def aurc_flat(probs, targets, mask, threshold: float = 0.5) -> float:
    """Single AURC over all masked (sample, label) pairs flattened together."""
    p, y = _flat_masked(probs, targets, mask)
    return _aurc_from_arrays(p, y, threshold)


def per_class_aurc(probs, targets, mask, conditions=CANONICAL_LABELS,
                   threshold: float = 0.5) -> dict:
    """Per-label AURC using the plug-in estimator. NaN when label has <2 samples."""
    probs, targets, mask = _np(probs), _np(targets), _np(mask)
    out = {}
    for j, c in enumerate(conditions):
        sel = mask[:, j] == 1
        if sel.sum() < 2:
            out[c] = float("nan")
            continue
        out[c] = _aurc_from_arrays(probs[sel, j], targets[sel, j], threshold)
    return out


def compute_aurc(probs, targets, mask, conditions=CANONICAL_LABELS,
                 threshold: float = 0.5) -> dict:
    """Full AURC report: flat + per-class + macro-average."""
    pc = per_class_aurc(probs, targets, mask, conditions, threshold)
    return {
        "flat": aurc_flat(probs, targets, mask, threshold),
        "per_class": pc,
        "macro": _macro(pc),
    }


