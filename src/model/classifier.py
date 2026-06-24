"""CxrClassifier — backbone + flat multi-label head, with differential-LR param groups.

Exposes param_groups for differential learning rates (backbone 1e-5 / head 1e-4)
and param_counts for logging. enable_mc_dropout activates dropout at inference for
MC Dropout. Segmentation input adaptation (seg_extra_channels) is wired here via
build_model_from_cfg.
"""

from __future__ import annotations

import torch.nn as nn

from src.data.label_space import CANONICAL_LABELS
from src.model.backbones import adapt_backbone_input, build_backbone
from src.model.heads import FlatMultiLabelHead


class CxrClassifier(nn.Module):
    def __init__(
        self,
        backbone: str = "densenet121_xrv",
        pretrained: bool = True,
        dropout: float = 0.2,
        num_classes: int = len(CANONICAL_LABELS),
        seg_extra_channels: int = 0,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.backbone, feat_dim, self.norm_kind = build_backbone(backbone, pretrained=pretrained)
        self.feature_dim = feat_dim

        # expand the first conv to take the extra segmentation mask channel(s)
        adapt_backbone_input(self.backbone, backbone, seg_extra_channels)

        self.head = FlatMultiLabelHead(feat_dim, num_classes, dropout)

    def forward(self, pixel_values):
        feats = self.backbone(pixel_values)
        return self.head(feats)

    def param_groups(self, backbone_lr: float, head_lr: float) -> list[dict]:
        return [
            {"params": list(self.backbone.parameters()), "lr": backbone_lr},
            {"params": list(self.head.parameters()), "lr": head_lr},
        ]

    def param_counts(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "backbone": sum(p.numel() for p in self.backbone.parameters()),
            "head": sum(p.numel() for p in self.head.parameters()),
        }

    def enable_mc_dropout(self) -> None:
        """Keep dropout active during eval for MC Dropout sampling."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()


def build_model_from_cfg(cfg, pretrained: bool | None = None) -> "CxrClassifier":
    """Single source of truth for model construction from a config, so every
    entry point (train/grid/eval/inference/xai) builds an identical architecture
    -- including the seg first-conv expansion -- and checkpoints load cleanly.
    pretrained=None uses cfg.model.pretrained; eval/inference pass False (weights
    come from the checkpoint)."""
    from src.data.segmentation import seg_extra_channels

    extra = seg_extra_channels(cfg.seg.method) if cfg.seg.enabled else 0
    return CxrClassifier(
        backbone=cfg.model.backbone,
        pretrained=cfg.model.pretrained if pretrained is None else pretrained,
        dropout=cfg.model.dropout,
        seg_extra_channels=extra,
    )
