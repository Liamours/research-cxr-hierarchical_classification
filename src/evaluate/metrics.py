"""Multi-label classification metrics, mask-aware.

All metrics ignore entries where label_mask == 0 (condition not applicable for
that dataset), so NIH/VinDr non-applicable columns never enter AUROC or F1.
A per-class metric is NaN when its masked subset has fewer than two classes
(AUROC) or no applicable samples; NaNs are dropped from macro averages.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from src.data.label_space import CANONICAL_LABELS


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def per_class_auroc(probs, targets, mask, conditions=CANONICAL_LABELS) -> dict:
    probs, targets, mask = _np(probs), _np(targets), _np(mask)
    out = {}
    for j, c in enumerate(conditions):
        sel = mask[:, j] == 1
        y = targets[sel, j]
        if sel.sum() == 0 or len(np.unique(y)) < 2:
            out[c] = float("nan")
        else:
            out[c] = float(roc_auc_score(y, probs[sel, j]))
    return out


def per_class_f1(probs, targets, mask, conditions=CANONICAL_LABELS, threshold=0.5) -> dict:
    probs, targets, mask = _np(probs), _np(targets), _np(mask)
    out = {}
    for j, c in enumerate(conditions):
        sel = mask[:, j] == 1
        if sel.sum() == 0:
            out[c] = float("nan")
        else:
            pred = (probs[sel, j] >= threshold).astype(int)
            out[c] = float(f1_score(targets[sel, j], pred, zero_division=0))
    return out


def _flat_masked(probs, targets, mask):
    probs, targets, mask = _np(probs), _np(targets), _np(mask)
    sel = mask.reshape(-1) == 1
    return probs.reshape(-1)[sel], targets.reshape(-1)[sel]


def micro_auroc(probs, targets, mask) -> float:
    p, y = _flat_masked(probs, targets, mask)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def micro_f1(probs, targets, mask, threshold=0.5) -> float:
    p, y = _flat_masked(probs, targets, mask)
    if len(y) == 0:
        return float("nan")
    return float(f1_score(y, (p >= threshold).astype(int), zero_division=0))


def _macro(d: dict) -> float:
    vals = [v for v in d.values() if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def per_class_ap(probs, targets, mask, conditions=CANONICAL_LABELS) -> dict:
    """Per-label Average Precision (area under PR curve).

    NaN when a label has no masked samples or no positives. Used for mAP.
    Preferred over per-class AUROC for rare/long-tail labels (Lin 2025 CXR-LT).
    """
    probs, targets, mask = _np(probs), _np(targets), _np(mask)
    out = {}
    for j, c in enumerate(conditions):
        sel = mask[:, j] == 1
        y = targets[sel, j]
        if sel.sum() == 0 or y.sum() == 0:
            out[c] = float("nan")
        else:
            out[c] = float(average_precision_score(y, probs[sel, j]))
    return out


def compute_map(probs, targets, mask, conditions=CANONICAL_LABELS) -> dict:
    """Mean Average Precision: per-class AP + macro average over classes with signal."""
    pc = per_class_ap(probs, targets, mask, conditions)
    return {"per_class": pc, "macro": _macro(pc)}


def compute_classification_metrics(probs, targets, mask, conditions=CANONICAL_LABELS,
                                   threshold=0.5) -> dict:
    pc_auroc = per_class_auroc(probs, targets, mask, conditions)
    pc_f1 = per_class_f1(probs, targets, mask, conditions, threshold)
    return {
        "auroc": {"per_class": pc_auroc, "macro": _macro(pc_auroc),
                  "micro": micro_auroc(probs, targets, mask)},
        "f1": {"per_class": pc_f1, "macro": _macro(pc_f1),
               "micro": micro_f1(probs, targets, mask, threshold), "threshold": threshold},
    }
