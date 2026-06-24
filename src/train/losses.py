"""Losses for flat and hierarchical multi-label CXR classification.

MaskedBCELoss — flat baseline. Excludes not-applicable conditions (label_mask
== 0) from numerator and denominator so dataset columns that do not apply
to a sample never contribute a spurious gradient.

HBCELoss — hierarchical extension (Asadi 2025 CIHMLC). Adds a differentiable
penalty for parent-child violations: when child_prob > 0.5 but parent_prob <
0.5. Penalty = relu(child - 0.5) × relu(0.5 - parent) — zero outside the
violation zone, smooth gradient inside. Only edges where both parent and child
are observed (mask == 1) contribute to the penalty.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor,
                           mask: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    loss = loss * mask
    denom = mask.sum().clamp_min(1.0)
    return loss.sum() / denom


class MaskedBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._last_bce: float = 0.0
        self._last_penalty: float = 0.0

    def forward(self, logits, targets, mask):
        loss = masked_bce_with_logits(logits, targets, mask)
        self._last_bce = float(loss.detach())
        self._last_penalty = 0.0
        return loss


class HBCELoss(nn.Module):
    """Hierarchical BCE loss (Asadi 2025 CIHMLC).

    L_HBCE = L_BCE + lam * mean_over_edges[ relu(child-0.5) * relu(0.5-parent) ]

    The relu-product is a differentiable approximation of Asadi's hard indicator
    1{parent<0.5 AND child>0.5}: it is zero outside the violation region and
    provides gradient signal only when both the child is confidently positive and
    the parent is confidently negative.

    edge_pairs: list of (parent_col_idx, child_col_idx) — from
        src.data.hierarchy.edge_index_pairs(conditions).
    lam: penalty weight (Asadi best result: 0.5 with data-driven penalty scale).
    """

    def __init__(self, edge_pairs: list[tuple[int, int]], lam: float = 0.5):
        super().__init__()
        self.lam = lam
        self._last_bce: float = 0.0
        self._last_penalty: float = 0.0
        if edge_pairs:
            pairs = torch.tensor(edge_pairs, dtype=torch.long)
            self.register_buffer("parent_idx", pairs[:, 0])
            self.register_buffer("child_idx", pairs[:, 1])
        else:
            self.register_buffer("parent_idx", torch.zeros(0, dtype=torch.long))
            self.register_buffer("child_idx", torch.zeros(0, dtype=torch.long))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        bce = masked_bce_with_logits(logits, targets, mask)

        if self.lam == 0.0 or self.parent_idx.numel() == 0:
            self._last_bce = float(bce.detach())
            self._last_penalty = 0.0
            return bce

        probs = torch.sigmoid(logits)                      # (B, C)

        parent_prob = probs[:, self.parent_idx]            # (B, E)
        child_prob  = probs[:, self.child_idx]             # (B, E)
        edge_mask   = mask[:, self.parent_idx] * mask[:, self.child_idx]  # (B, E)

        violation = F.relu(child_prob - 0.5) * F.relu(0.5 - parent_prob)

        denom   = edge_mask.sum().clamp_min(1.0)
        penalty = (violation * edge_mask).sum() / denom
        scaled_penalty = self.lam * penalty

        self._last_bce = float(bce.detach())
        self._last_penalty = float(scaled_penalty.detach())
        return bce + scaled_penalty


def build_loss(label_structure: str, conditions: list[str],
               lam: float = 0.5) -> nn.Module:
    """Factory used by the trainer. label_structure: 'flat' | 'hierarchical'."""
    if label_structure == "hierarchical":
        from src.data.hierarchy import edge_index_pairs, validate_edges
        warnings = validate_edges(conditions)
        for w in warnings:
            import warnings as _w
            _w.warn(f"[HBCE] {w}", stacklevel=2)
        pairs = edge_index_pairs(conditions)
        return HBCELoss(pairs, lam=lam)
    return MaskedBCELoss()
