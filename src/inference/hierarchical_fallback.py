"""Hierarchical fallback at inference time (our novel contribution).

When MC-Dropout variance for a child label exceeds the gate_threshold, we
"retreat to the coarser parent label": the child prediction is suppressed
(set to 0) while the parent prediction stands on its own.

Conceptual basis: Mortier et al. 2026 (CRSVP, AISTATS 2026, PMLR 300) show
that predicting ancestor nodes when uncertain about leaf nodes preserves a
formal coverage guarantee. We adapt this idea to multi-label inference without
the formal conformal framework: uncertain child → abstain on child, keep parent.

Implementation note: suppression is applied to *mean_probs* in-place on a
clone. Variance tensor is read-only. Cascading is single-level (child→parent
only; parents without a parent in the DAG are never suppressed by this step).
"""

from __future__ import annotations

import torch


def apply_hierarchical_fallback(
    mean_probs: torch.Tensor,
    variance: torch.Tensor,
    edge_pairs: list[tuple[int, int]],
    gate_threshold: float = 0.5,
) -> tuple[torch.Tensor, dict]:
    """Suppress uncertain child label predictions; retreat to parent.

    Args:
        mean_probs:     (B, C) or (C,) mean probabilities from MC-Dropout.
        variance:       same shape as mean_probs — per-label predictive variance.
        edge_pairs:     list of (parent_col_idx, child_col_idx) from
                        src.data.hierarchy.edge_index_pairs(conditions).
        gate_threshold: variance above this → child suppressed (default 0.5,
                        from cfg.uq.gate_threshold).

    Returns:
        adjusted_probs: clone of mean_probs with suppressed children set to 0.
        log:            dict with total suppressed count and per-edge stats.
    """
    single = mean_probs.dim() == 1
    if single:
        mean_probs = mean_probs.unsqueeze(0)
        variance   = variance.unsqueeze(0)

    adjusted = mean_probs.clone()
    log: dict = {"gate_threshold": gate_threshold, "edges": [], "total_suppressed": 0}

    for parent_idx, child_idx in edge_pairs:
        uncertain = variance[:, child_idx] > gate_threshold   # (B,) bool
        n_suppressed = int(uncertain.sum().item())
        if n_suppressed > 0:
            adjusted[uncertain, child_idx] = 0.0
        log["edges"].append({
            "parent_idx": parent_idx,
            "child_idx": child_idx,
            "suppressed": n_suppressed,
        })
        log["total_suppressed"] += n_suppressed

    if single:
        adjusted = adjusted.squeeze(0)

    return adjusted, log


def fallback_summary(log: dict) -> str:
    """Human-readable one-liner for logging."""
    n = log["total_suppressed"]
    thr = log["gate_threshold"]
    return f"hierarchical fallback: {n} child predictions suppressed (var > {thr})"
