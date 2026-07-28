"""Statistical inference for metric comparison.

bootstrap_metric_ci: 95% CI via percentile bootstrap (n_boot=1000) over
    masked (sample, label) pairs read from a predictions CSV.

compare_aurc: pairwise comparison of AURC between two conditions using
    bootstrap hypothesis test (one-tailed: H1: AURC_A < AURC_B).

compare_auroc_macro, compare_f1_macro: same paired bootstrap-delta pattern
    as compare_aurc, applied to AUROC macro and F1 macro (one-tailed:
    H1: metric_A > metric_B). All three share the _compare_metric helper.

mcnemar_comparison: McNemar's test comparing two classifiers on matched
    binary decisions (threshold 0.5) over all masked (sample, label) pairs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from src.evaluate.metrics import (
    _flat_masked, _macro, per_class_auroc, per_class_f1,
    per_class_ap, _np,
)
from src.evaluate.selective import aurc_flat, per_class_aurc
from src.data.label_space import CANONICAL_LABELS


# ---------------------------------------------------------------------------
# Predictions CSV helpers
# ---------------------------------------------------------------------------

def load_predictions(pred_csv: str | Path, conditions=CANONICAL_LABELS):
    """Load a predictions CSV saved by evaluator.save_predictions.

    Returns (probs, targets, mask, var_or_None) as numpy arrays (N, C).
    """
    df = pd.read_csv(pred_csv)
    N, C = len(df), len(conditions)
    probs = np.stack([df[f"prob_{c}"].values for c in conditions], axis=1).astype(np.float32)
    targets = np.stack([df[f"label_{c}"].values for c in conditions], axis=1).astype(np.float32)
    mask = np.stack([df[f"mask_{c}"].values for c in conditions], axis=1).astype(np.float32)
    var_cols = [f"var_{c}" for c in conditions]
    if all(c in df.columns for c in var_cols):
        var = np.stack([df[c].values for c in var_cols], axis=1).astype(np.float32)
    else:
        var = None
    return probs, targets, mask, var


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def _bootstrap_metric(metric_fn, *arrays, n_boot: int = 1000, alpha: float = 0.05,
                      rng=None):
    """Percentile bootstrap CI for metric_fn(*arrays) -> float.

    Resamples rows (first axis of each array) together. Returns
    (estimate, lo, hi) where lo/hi are (alpha/2) and (1-alpha/2) percentiles.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    N = arrays[0].shape[0]
    base = metric_fn(*arrays)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, N, size=N)
        v = metric_fn(*[a[idx] for a in arrays])
        if not np.isnan(v):
            boot.append(v)
    boot = np.array(boot)
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return float(base), lo, hi


def bootstrap_metric_ci(probs, targets, mask, conditions=CANONICAL_LABELS,
                        n_boot: int = 1000) -> dict:
    """95% CIs for AUROC macro, F1 macro, mAP macro, AURC macro and flat."""
    probs, targets, mask = _np(probs), _np(targets), _np(mask)
    rng = np.random.default_rng(0)

    def _auroc_macro(p, t, m):
        return _macro(per_class_auroc(p, t, m, conditions))

    def _f1_macro(p, t, m):
        return _macro(per_class_f1(p, t, m, conditions))

    def _map_macro(p, t, m):
        return _macro(per_class_ap(p, t, m, conditions))

    def _aurc_flat_fn(p, t, m):
        return aurc_flat(p, t, m)

    def _aurc_macro_fn(p, t, m):
        return _macro(per_class_aurc(p, t, m, conditions))

    out = {}
    for name, fn in [
        ("auroc_macro", _auroc_macro),
        ("f1_macro",    _f1_macro),
        ("map_macro",   _map_macro),
        ("aurc_flat",   _aurc_flat_fn),
        ("aurc_macro",  _aurc_macro_fn),
    ]:
        est, lo, hi = _bootstrap_metric(fn, probs, targets, mask, n_boot=n_boot, rng=rng)
        out[name] = {"estimate": round(est, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}

    return out


# ---------------------------------------------------------------------------
# Paired bootstrap-delta comparison (AURC, AUROC macro, F1 macro)
# ---------------------------------------------------------------------------

def _compare_metric(
    metric_fn, metric_name,
    probs_a, targets_a, mask_a,
    probs_b, targets_b, mask_b,
    higher_is_better: bool,
    n_boot: int = 1000,
    label: tuple[str, str] = ("A", "B"),
) -> dict:
    """Shared bootstrap one-tailed delta test for any metric_fn(probs, targets, mask) -> float.

    Samples (N, C) row-paired when arrays have the same first dimension
    (same test set, different model predictions). Falls back to independent
    resampling when sizes differ. H1 is "A is better than B" in whichever
    direction higher_is_better implies.
    """
    probs_a, targets_a, mask_a = _np(probs_a), _np(targets_a), _np(mask_a)
    probs_b, targets_b, mask_b = _np(probs_b), _np(targets_b), _np(mask_b)
    paired = probs_a.shape[0] == probs_b.shape[0]
    rng = np.random.default_rng(0)

    est_a = metric_fn(probs_a, targets_a, mask_a)
    est_b = metric_fn(probs_b, targets_b, mask_b)
    obs_delta = est_a - est_b

    Na, Nb = probs_a.shape[0], probs_b.shape[0]
    boot_deltas = []
    for _ in range(n_boot):
        if paired:
            idx = rng.integers(0, Na, size=Na)
            a = metric_fn(probs_a[idx], targets_a[idx], mask_a[idx])
            b = metric_fn(probs_b[idx], targets_b[idx], mask_b[idx])
        else:
            ia = rng.integers(0, Na, size=Na)
            ib = rng.integers(0, Nb, size=Nb)
            a = metric_fn(probs_a[ia], targets_a[ia], mask_a[ia])
            b = metric_fn(probs_b[ib], targets_b[ib], mask_b[ib])
        boot_deltas.append(a - b)

    boot_deltas = np.array(boot_deltas)
    boot_deltas = boot_deltas[~np.isnan(boot_deltas)]
    # p-value for H1 "A better than B": higher-is-better -> A>B, lower-is-better -> A<B
    p_value = float(np.mean(boot_deltas <= 0)) if higher_is_better else float(np.mean(boot_deltas >= 0))
    ci_delta_lo = float(np.percentile(boot_deltas, 2.5))
    ci_delta_hi = float(np.percentile(boot_deltas, 97.5))
    direction = ">" if higher_is_better else "<"

    return {
        f"{metric_name}_a": {"name": label[0], "estimate": round(est_a, 4)},
        f"{metric_name}_b": {"name": label[1], "estimate": round(est_b, 4)},
        "delta_a_minus_b": round(obs_delta, 4),
        "delta_ci_95": [round(ci_delta_lo, 4), round(ci_delta_hi, 4)],
        "p_value_h1_a_better": round(p_value, 4),
        "significant_at_0.05": bool(p_value < 0.05),
        "note": f"one-tailed bootstrap; H1: {metric_name}_A {direction} {metric_name}_B",
    }


def compare_aurc(
    probs_a, targets_a, mask_a,
    probs_b, targets_b, mask_b,
    n_boot: int = 1000,
    label: tuple[str, str] = ("A", "B"),
) -> dict:
    """Bootstrap one-tailed test: H0: AURC_A >= AURC_B, H1: AURC_A < AURC_B.

    Returns p-value and CIs for both conditions, plus the observed delta.
    """
    out = _compare_metric(
        aurc_flat, "aurc",
        probs_a, targets_a, mask_a, probs_b, targets_b, mask_b,
        higher_is_better=False, n_boot=n_boot, label=label,
    )
    # preserve original key names / p-value semantics for backward compatibility
    return {
        "aurc_a": out["aurc_a"], "aurc_b": out["aurc_b"],
        "delta_a_minus_b": out["delta_a_minus_b"],
        "delta_ci_95": out["delta_ci_95"],
        "p_value_h1_a_lt_b": out["p_value_h1_a_better"],
        "significant_at_0.05": out["significant_at_0.05"],
        "note": "one-tailed bootstrap; H1: AURC_A < AURC_B (A has lower risk)",
    }


def compare_auroc_macro(
    probs_a, targets_a, mask_a,
    probs_b, targets_b, mask_b,
    conditions=CANONICAL_LABELS,
    n_boot: int = 1000,
    label: tuple[str, str] = ("A", "B"),
) -> dict:
    """Bootstrap one-tailed test: H1: AUROC_macro_A > AUROC_macro_B."""
    def _auroc_macro(p, t, m):
        return _macro(per_class_auroc(p, t, m, conditions))
    return _compare_metric(
        _auroc_macro, "auroc_macro",
        probs_a, targets_a, mask_a, probs_b, targets_b, mask_b,
        higher_is_better=True, n_boot=n_boot, label=label,
    )


def compare_f1_macro(
    probs_a, targets_a, mask_a,
    probs_b, targets_b, mask_b,
    conditions=CANONICAL_LABELS,
    n_boot: int = 1000,
    label: tuple[str, str] = ("A", "B"),
) -> dict:
    """Bootstrap one-tailed test: H1: F1_macro_A > F1_macro_B."""
    def _f1_macro(p, t, m):
        return _macro(per_class_f1(p, t, m, conditions))
    return _compare_metric(
        _f1_macro, "f1_macro",
        probs_a, targets_a, mask_a, probs_b, targets_b, mask_b,
        higher_is_better=True, n_boot=n_boot, label=label,
    )


# ---------------------------------------------------------------------------
# McNemar's test
# ---------------------------------------------------------------------------

def mcnemar_comparison(
    probs_a, targets_a, mask_a,
    probs_b, targets_b, mask_b,
    threshold: float = 0.5,
    label: tuple[str, str] = ("A", "B"),
) -> dict:
    """McNemar's test on matched binary decisions over all masked (sample, label) pairs.

    Requires same test set (paired predictions). Uses mid-p McNemar for small
    discordant counts, exact binomial for n_01 + n_10 < 25.
    """
    from scipy.stats import binom

    probs_a, targets_a, mask_a = _np(probs_a), _np(targets_a), _np(mask_a)
    probs_b, targets_b, mask_b = _np(probs_b), _np(targets_b), _np(mask_b)

    assert probs_a.shape == probs_b.shape, "McNemar requires paired predictions (same test set)"

    sel = (mask_a.reshape(-1) == 1) & (mask_b.reshape(-1) == 1)
    ya = (probs_a.reshape(-1)[sel] >= threshold).astype(int)
    yb = (probs_b.reshape(-1)[sel] >= threshold).astype(int)
    gt = targets_a.reshape(-1)[sel].astype(int)

    correct_a = (ya == gt)
    correct_b = (yb == gt)

    n_10 = int(( correct_a & ~correct_b).sum())   # A right, B wrong
    n_01 = int((~correct_a &  correct_b).sum())   # A wrong, B right
    n_11 = int(( correct_a &  correct_b).sum())
    n_00 = int((~correct_a & ~correct_b).sum())

    n_disc = n_10 + n_01
    if n_disc == 0:
        p_value = 1.0
    elif n_disc < 25:
        # exact two-tailed binomial
        k = min(n_10, n_01)
        p_value = float(2 * binom.cdf(k, n_disc, 0.5))
    else:
        # mid-p McNemar
        p_value = float(2 * binom.cdf(min(n_10, n_01) - 1, n_disc, 0.5)
                        + binom.pmf(min(n_10, n_01), n_disc, 0.5))

    return {
        "contingency": {"n_11": n_11, "n_10": n_10, "n_01": n_01, "n_00": n_00},
        "n_discordant": n_disc,
        "label_a_wins": n_10,
        "label_b_wins": n_01,
        "p_value_two_tailed": round(p_value, 4),
        "significant_at_0.05": bool(p_value < 0.05),
        f"{label[0]}_acc": round((n_10 + n_11) / max(n_10 + n_01 + n_11 + n_00, 1), 4),
        f"{label[1]}_acc": round((n_01 + n_11) / max(n_10 + n_01 + n_11 + n_00, 1), 4),
    }
