"""Classification head — flat multi-label.

FlatMultiLabelHead is the only head. The dropout layer is the toggle point for
MC Dropout at inference (uncertainty quantification).
"""

from __future__ import annotations

import torch.nn as nn

from src.data.label_space import CANONICAL_LABELS


class FlatMultiLabelHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = len(CANONICAL_LABELS), dropout: float = 0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, feats):
        return self.fc(self.dropout(feats))
