"""Run (or generate) the 8-condition ablation grid and aggregate results.

    # write the 8 committed condition configs
    uv run python src/script/run_grid.py --base-config configs/grid_base.yaml --generate

    # run the whole grid on real data
    uv run python src/script/run_grid.py --base-config configs/grid_base.yaml

    # smoke a few conditions on a subset
    uv run python src/script/run_grid.py --base-config <cfg> --limit 3 --epochs 1 --max-samples 64
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
from src.evaluate.evaluator import evaluate_model
from src.experiment.aggregate import aggregate
from src.experiment.grid import build_grid, write_grid_configs
from src.model.classifier import build_model_from_cfg
from src.train.trainer import MultiLabelTrainer
from src.util.logging import RunLogger
from src.util.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-config", required=True)
    p.add_argument("--generate", action="store_true", help="write the 8 configs to configs/grid and exit")
    p.add_argument("--out-dir", default="configs/grid")
    p.add_argument("--limit", type=int, default=None, help="run only the first N conditions")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--eval-split", default="val", choices=["train", "val", "test"])
    return p.parse_args()


def _subset(loaders, cfg, max_samples):
    for split in ("train", "val", "test"):
        loader = loaders.get(split)
        if loader is None:
            continue
        ds = loader.dataset
        sub = Subset(ds, list(range(min(max_samples, len(ds)))))
        loaders[split] = DataLoader(sub, batch_size=cfg.data.batch_size,
                                    shuffle=(split == "train"), drop_last=(split == "train"))
    return loaders


def run_condition(cfg, device, max_samples=None, eval_split="val") -> str:
    set_seed(cfg.experiment.seed)
    loaders = build_loaders(cfg)
    if max_samples:
        loaders = _subset(loaders, cfg, max_samples)
    if loaders.get("train") is None:
        return "no_data"

    model = build_model_from_cfg(cfg)
    logger = RunLogger(cfg.run_dir(), level=cfg.logging.level)
    logger.snapshot_config(cfg)
    MultiLabelTrainer(model, loaders, cfg, device, logger).train()

    eval_loader = loaders.get(eval_split) or loaders.get("val")
    if eval_loader is not None:
        evaluate_model(model, eval_loader, device, cfg, split=eval_split, logger=logger)
    return "done"


def main():
    args = parse_args()
    base = ExperimentConfig.from_yaml(args.base_config)

    if args.generate:
        paths = write_grid_configs(base, args.out_dir)
        print(f"Wrote {len(paths)} condition configs to {args.out_dir}:")
        for p in paths:
            print(f"  {p.name}")
        return

    configs = build_grid(base)
    if args.epochs is not None:
        for c in configs:
            c.training.epochs = args.epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    to_run = configs[: args.limit] if args.limit else configs
    print(f"Grid: {len(configs)} conditions; running {len(to_run)} on {device}")
    for i, cfg in enumerate(to_run, 1):
        print(f"\n[{i}/{len(to_run)}] {cfg.experiment.name}")
        status = run_condition(cfg, device, args.max_samples, args.eval_split)
        print(f"  -> {status}")

    df = aggregate(configs, args.eval_split)
    print("\n=== Grid results ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
