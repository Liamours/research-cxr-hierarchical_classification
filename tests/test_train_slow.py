"""Slow tests: training sanity (overfit one batch, loss decreases, checkpoints)
and loader modes. Run with: uv run pytest -m slow"""

from __future__ import annotations

import pytest
import torch
from torch.optim import AdamW

from src.data.loader import build_loaders
from src.model.classifier import build_model_from_cfg
from src.train.losses import MaskedBCELoss
from src.train.trainer import MultiLabelTrainer
from src.util.logging import RunLogger
from src.util.seed import set_seed

pytestmark = pytest.mark.slow


def test_overfit_single_batch(make_cfg):
    cfg = make_cfg(name="overfit")
    set_seed(42)
    loaders = build_loaders(cfg)
    batch = next(iter(loaders["train"]))
    model = build_model_from_cfg(cfg, pretrained=False)
    model.train()
    opt = AdamW(model.parameters(), lr=1e-2)
    crit = MaskedBCELoss()
    x, y, m = batch["pixel_values"], batch["labels"], batch["label_mask"]
    losses = []
    for _ in range(60):
        opt.zero_grad()
        loss = crit(model(x), y, m)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.5, f"did not overfit: {losses[0]:.3f} -> {losses[-1]:.3f}"


def test_trainer_saves_four_checkpoints(make_cfg):
    cfg = make_cfg(name="ckpt")
    set_seed(42)
    loaders = build_loaders(cfg)
    model = build_model_from_cfg(cfg, pretrained=False)
    logger = RunLogger(cfg.run_dir(), level=cfg.logging.level)
    logger.snapshot_config(cfg)
    MultiLabelTrainer(model, loaders, cfg, torch.device("cpu"), logger).train()
    names = {p.stem for p in (cfg.run_dir() / "checkpoints").glob("*.pt")}
    assert {"last", "best_val_loss", "best_val_auroc_macro", "best_val_f1_macro"} <= names

