"""CxrClassifier — backbone + flat multi-label head, with differential-LR param groups.

Exposes param_groups for differential learning rates (backbone 1e-5 / head 1e-4)
and param_counts for logging. enable_mc_dropout activates dropout at inference for
MC Dropout. Segmentation input adaptation (seg_extra_channels) is wired here via
build_model_from_cfg.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.data.label_space import CANONICAL_LABELS
from src.model.backbones import adapt_backbone_input, build_backbone


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

        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_dim, num_classes))

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


def model_profile(
    model: "CxrClassifier",
    seg_extra_channels: int = 0,
    device="cpu",
    run_dir=None,
) -> dict:
    """FLOPs (GMACs) + parameter counts + full torchinfo layer table.

    Writes the full layer summary to <run_dir>/model_summary.txt when run_dir
    is given. Returns dict with: total, trainable, backbone, head, gmacs,
    params_mb, summary_str."""
    from pathlib import Path

    counts = model.param_counts()
    was_training = model.training
    base_channels = {"xrv": 1, "imagenet": 3}.get(model.norm_kind, 1)
    try:
        from torchinfo import summary as ti_summary
        stats = ti_summary(
            model,
            input_size=(1, base_channels + seg_extra_channels, 224, 224),
            device=device,
            verbose=0,
            depth=10,
            col_names=("input_size", "output_size", "num_params", "mult_adds"),
        )
        model.train(was_training)
        counts["gmacs"]     = round(stats.total_mult_adds / 1e9, 3)
        counts["params_mb"] = round(stats.total_param_bytes / 1e6, 2)
        counts["summary_str"] = str(stats)
    except ImportError:
        model.train(was_training)
        counts["gmacs"]       = None
        counts["params_mb"]   = None
        counts["summary_str"] = str(model)

    if run_dir is not None:
        p = Path(run_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / "model_summary.txt").write_text(counts["summary_str"], encoding="utf-8")

    return counts


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
