"""Statistical inference tests (fast; synthetic data, no real predictions)."""

from __future__ import annotations

import numpy as np

from src.evaluate.statistics import (
    compare_aurc, compare_auroc_macro, compare_f1_macro, mcnemar_comparison,
)

CONDITIONS = ["a", "b"]


def _synthetic(n=200, seed=0, flip_frac=0.0):
    rng = np.random.default_rng(seed)
    targets = (rng.random((n, 2)) > 0.5).astype(np.float32)
    probs = np.where(targets == 1, rng.uniform(0.6, 1.0, (n, 2)), rng.uniform(0.0, 0.4, (n, 2))).astype(np.float32)
    if flip_frac:
        idx = rng.choice(n, size=int(n * flip_frac), replace=False)
        probs[idx] = 1 - probs[idx]
    mask = np.ones((n, 2), dtype=np.float32)
    return probs, targets, mask


def test_compare_functions_identical_inputs_not_significant():
    probs, targets, mask = _synthetic()
    for fn in (compare_aurc, compare_auroc_macro, compare_f1_macro):
        kwargs = {} if fn is compare_aurc else {"conditions": CONDITIONS}
        out = fn(probs, targets, mask, probs, targets, mask, **kwargs)
        assert out["delta_a_minus_b"] == 0.0
        assert not out["significant_at_0.05"]


def test_compare_metric_detects_worse_condition():
    probs_a, targets, mask = _synthetic(seed=1)
    probs_b = probs_a.copy()
    probs_b, _, _ = _synthetic(seed=1, flip_frac=0.5)  # b: half its predictions flipped -> worse
    out = compare_auroc_macro(probs_a, targets, mask, probs_b, targets, mask, conditions=CONDITIONS)
    assert out["delta_a_minus_b"] > 0
    assert out["significant_at_0.05"]

    out_aurc = compare_aurc(probs_a, targets, mask, probs_b, targets, mask)
    assert out_aurc["delta_a_minus_b"] < 0
    assert out_aurc["significant_at_0.05"]


def test_mcnemar_identical_predictions_not_significant():
    probs, targets, mask = _synthetic()
    out = mcnemar_comparison(probs, targets, mask, probs, targets, mask)
    assert out["n_discordant"] == 0
    assert out["p_value_two_tailed"] == 1.0
    assert not out["significant_at_0.05"]
