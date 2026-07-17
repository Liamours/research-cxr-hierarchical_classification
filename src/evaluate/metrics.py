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


def _metrics_from_counts(tp, fp, fn, tn) -> dict:
    n = tp + fp + fn + tn
    prec = tp / (tp + fp) if tp + fp > 0 else float("nan")
    rec = tp / (tp + fn) if tp + fn > 0 else float("nan")          # sensitivity / recall
    spec = tn / (tn + fp) if tn + fp > 0 else float("nan")
    acc = (tp + tn) / n if n > 0 else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else float("nan")
    bal = (rec + spec) / 2 if not (np.isnan(rec) or np.isnan(spec)) else float("nan")
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den > 0 else float("nan")
    return {"precision": prec, "recall": rec, "specificity": spec, "accuracy": acc,
            "f1": f1, "balanced_accuracy": bal, "mcc": mcc}


_CLF_KEYS = ("precision", "recall", "specificity", "accuracy", "f1", "balanced_accuracy", "mcc")


def compute_confusion_metrics(probs, targets, mask, conditions=CANONICAL_LABELS,
                              threshold=0.5) -> dict:
    """Threshold-based classification family (precision, recall/sensitivity,
    specificity, accuracy, F1, balanced accuracy, MCC) as per-class + macro +
    micro (pooled counts) + weighted (by positive support). Plus subset accuracy
    (exact match over each sample's applicable labels). Mask-aware.
    """
    probs, targets, mask = _np(probs), _np(targets), _np(mask)
    per_class, counts, support = {}, {}, {}
    for j, c in enumerate(conditions):
        sel = mask[:, j] == 1
        if sel.sum() == 0:
            per_class[c] = {k: float("nan") for k in _CLF_KEYS}
            continue
        y = targets[sel, j]
        pb = (probs[sel, j] >= threshold).astype(int)
        tp = float(np.sum((pb == 1) & (y == 1)))
        fp = float(np.sum((pb == 1) & (y == 0)))
        fn = float(np.sum((pb == 0) & (y == 1)))
        tn = float(np.sum((pb == 0) & (y == 0)))
        counts[c] = (tp, fp, fn, tn)
        support[c] = tp + fn
        per_class[c] = _metrics_from_counts(tp, fp, fn, tn)

    macro = {k: _macro({c: per_class[c][k] for c in per_class}) for k in _CLF_KEYS}
    if counts:
        TP = sum(v[0] for v in counts.values()); FP = sum(v[1] for v in counts.values())
        FN = sum(v[2] for v in counts.values()); TN = sum(v[3] for v in counts.values())
        micro = _metrics_from_counts(TP, FP, FN, TN)
    else:
        micro = {k: float("nan") for k in _CLF_KEYS}
    tot = sum(s for s in support.values() if s > 0)
    weighted = {}
    for k in _CLF_KEYS:
        vals = [per_class[c][k] * support[c] for c in per_class
                if support.get(c, 0) > 0 and not np.isnan(per_class[c][k])]
        weighted[k] = float(np.sum(vals) / tot) if tot > 0 else float("nan")

    return {"per_class": per_class, "macro": macro, "micro": micro,
            "weighted": weighted, "subset_accuracy": _subset_accuracy(probs, targets, mask, threshold),
            "threshold": threshold}


def _subset_accuracy(probs, targets, mask, threshold=0.5) -> float:
    """Fraction of samples where every applicable (masked) label is predicted correctly."""
    probs, targets, mask = _np(probs), _np(targets), _np(mask)
    pb = (probs >= threshold).astype(int)
    hits = []
    for i in range(probs.shape[0]):
        m = mask[i] == 1
        if m.sum() == 0:
            continue
        hits.append(bool(np.all(pb[i][m] == targets[i][m])))
    return float(np.mean(hits)) if hits else float("nan")


def hierarchy_violation_rate(probs, conditions=CANONICAL_LABELS, threshold=0.5) -> dict:
    """Fraction of (sample, edge) pairs predicting child>=thr while parent<thr.

    Measures prediction consistency (the target of the HBCE loss), so it is
    computed from probs alone and is mask-independent. Lower is better.
    """
    from src.data.hierarchy import edge_index_pairs

    probs = _np(probs)
    pairs = edge_index_pairs(conditions)
    n = probs.shape[0]
    if not pairs or n == 0:
        return {"rate": float("nan"), "n_violations": 0, "n_pairs": 0, "per_edge": {}}
    per_edge, total_viol = {}, 0
    for p_idx, c_idx in pairs:
        viol = int(((probs[:, c_idx] >= threshold) & (probs[:, p_idx] < threshold)).sum())
        per_edge[f"{conditions[p_idx]}>{conditions[c_idx]}"] = viol / n
        total_viol += viol
    return {"rate": total_viol / (n * len(pairs)), "n_violations": total_viol,
            "n_pairs": n * len(pairs), "per_edge": per_edge}


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
