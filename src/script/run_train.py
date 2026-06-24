"""Training entry point. Every run goes through a YAML config.

    uv run python src/script/run_train.py --config configs/grid/densenet121_xrv__seg-off__uq-off.yaml
    uv run python src/script/run_train.py --config configs/grid/densenet121_xrv__seg-off__uq-off.yaml --max-samples 64
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.utils.data import DataLoader, Subset

from src.config.experiment_config import ExperimentConfig
from src.data.loader import build_loaders
from src.model.classifier import build_model_from_cfg
from src.train.trainer import MultiLabelTrainer
from src.util.logging import RunLogger
from src.util.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--max-samples", type=int, default=None)
    return p.parse_args()


def build_overrides(args) -> dict:
    ov = {}
    if args.epochs is not None:
        ov["training.epochs"] = args.epochs
    if args.batch_size is not None:
        ov["data.batch_size"] = args.batch_size
    if args.no_pretrained:
        ov["model.pretrained"] = False
    return ov


def _subset(loaders, cfg, max_samples):
    for split in ("train", "val"):
        loader = loaders.get(split)
        if loader is None:
            continue
        ds = loader.dataset
        sub = Subset(ds, list(range(min(max_samples, len(ds)))))
        loaders[split] = DataLoader(
            sub, batch_size=cfg.data.batch_size, shuffle=(split == "train"),
            drop_last=(split == "train"),
        )
    return loaders


def main():
    args = parse_args()
    cfg = ExperimentConfig.from_yaml(args.config, overrides=build_overrides(args))
    set_seed(cfg.experiment.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Config:\n" + cfg.summary())
    print(f"Device: {device}")

    loaders = build_loaders(cfg)
    if args.max_samples:
        loaders = _subset(loaders, cfg, args.max_samples)
    if loaders.get("train") is None:
        print(f"No training data ({cfg.data.label_csv} / {cfg.data.image_root}). "
              "Preprocess datasets first. Nothing to train.")
        return

    model = build_model_from_cfg(cfg)
    logger = RunLogger(cfg.run_dir(), level=cfg.logging.level)
    logger.snapshot_config(cfg)

    trainer = MultiLabelTrainer(model, loaders, cfg, device, logger)
    trainer.train()
    print(f"Run dir: {cfg.run_dir()}")


if __name__ == "__main__":
    main()
