"""Losses for flat, hierarchical, and BAFL multi-label CXR classification.

MaskedBCELoss — flat baseline. Excludes not-applicable conditions (label_mask
== 0) from numerator and denominator so dataset columns that do not apply
to a sample never contribute a spurious gradient.

HBCELoss — hierarchical extension (Asadi 2025 CIHMLC). Adds a differentiable
penalty for parent-child violations: when child_prob > 0.5 but parent_prob <
0.5. Penalty = relu(child - 0.5) × relu(0.5 - parent) — zero outside the
violation zone, smooth gradient inside. Only edges where both parent and child
are observed (mask == 1) contribute to the penalty.

BAFLLoss — Balanced Adaptive Focal Loss, isolated from the HP-ViT architecture
(Khan et al. 2026, Sect. 3.3; see context/hpvit-bafl-ablation-proposal.md).
Per-class effective-number-of-samples weight (Cui et al. 2019) times a focal
term (1-p_t)^gamma, where gamma ramps gamma_init -> gamma_final over
t_warmup epochs. Same NaN-mask handling as MaskedBCELoss. Isolated on purpose:
backbone, dataset, splits, and label hierarchy are unchanged -- only the loss.
"""

from __future__ import annotations

import numpy as np
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
        edge_mask   = mask[:, self.child_idx]                              # (B, E)

        violation = F.relu(child_prob - 0.5) * F.relu(0.5 - parent_prob)

        denom   = edge_mask.sum().clamp_min(1.0)
        penalty = (violation * edge_mask).sum() / denom
        scaled_penalty = self.lam * penalty

        self._last_bce = float(bce.detach())
        self._last_penalty = float(scaled_penalty.detach())
        return bce + scaled_penalty


class BAFLLoss(nn.Module):
    """Balanced Adaptive Focal Loss (HP-ViT paper, Sect. 3.3), applied on top of
    the existing masked multi-label setup instead of HP-ViT's full architecture.

    class_weights: per-condition effective-number weight (see
        train_class_weights), same order as `conditions`.
    gamma ramps linearly gamma_init -> gamma_final over t_warmup epochs;
    call set_epoch(epoch) once per training epoch (trainer does this).
    """

    def __init__(self, class_weights: torch.Tensor, gamma_init: float = 0.5,
                 gamma_final: float = 2.5, t_warmup: int = 30):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.gamma_init = gamma_init
        self.gamma_final = gamma_final
        self.t_warmup = max(t_warmup, 1)
        self.gamma = gamma_init
        self._last_bce: float = 0.0
        self._last_penalty: float = 0.0

    def set_epoch(self, epoch: int) -> None:
        progress = min(epoch / self.t_warmup, 1.0)
        self.gamma = self.gamma_init + progress * (self.gamma_final - self.gamma_init)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal = (1 - p_t).clamp_min(0.0).pow(self.gamma)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        weighted = self.class_weights * focal * bce * mask
        denom = mask.sum().clamp_min(1.0)
        loss = weighted.sum() / denom
        self._last_bce = float(loss.detach())
        self._last_penalty = 0.0
        return loss


def train_class_weights(label_csv, conditions: list[str], beta: float = 0.999) -> torch.Tensor:
    """Cui et al. 2019 effective-number-of-samples weight per condition, from
    the train split of label_csv. w_c = (1-beta)/(1-beta^n_c), n_c = positive
    count (floor 1 to avoid div-by-0 for always-masked conditions). Normalized
    to mean 1 over conditions with train-split support only -- conditions with
    n_c==0 are always mask==0 in the loss, so they must not dilute the
    normalization of the conditions that actually train."""
    import pandas as pd
    from src.eda.dataset_stats import class_distribution

    df = pd.read_csv(label_csv, low_memory=False)
    train_df = df[df["split"] == "train"] if "split" in df.columns else df
    stats = class_distribution(train_df, conditions)
    pos = stats["positive"].to_numpy(dtype=np.float64)
    n = np.maximum(pos, 1.0)
    w = (1.0 - beta) / (1.0 - np.power(beta, n))
    active = pos > 0
    if active.any():
        w = w / w[active].mean()
    return torch.tensor(w, dtype=torch.float32)


def build_loss(label_structure: str, conditions: list[str], lam: float = 0.5,
               *, bafl_weights: torch.Tensor | None = None,
               bafl_gamma_init: float = 0.5, bafl_gamma_final: float = 2.5,
               bafl_t_warmup: int = 30) -> nn.Module:
    """Factory used by the trainer. label_structure: 'flat' | 'hierarchical' | 'bafl'."""
    if label_structure == "hierarchical":
        import warnings as _w
        from src.data.hierarchy import edge_index_pairs, validate_edges
        for w in validate_edges(conditions):
            _w.warn(f"[HBCE] {w}", stacklevel=2)
        pairs = edge_index_pairs(conditions)
        return HBCELoss(pairs, lam=lam)
    if label_structure == "bafl":
        if bafl_weights is None:
            bafl_weights = torch.ones(len(conditions))
        return BAFLLoss(bafl_weights, gamma_init=bafl_gamma_init,
                        gamma_final=bafl_gamma_final, t_warmup=bafl_t_warmup)
    return MaskedBCELoss()
