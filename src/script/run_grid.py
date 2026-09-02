"""Run (or generate) the 4-condition ablation grid and aggregate results.

    # write the 4 committed condition configs
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
from src.script.bench_num_workers import bench, gpu_warmup
from src.train.trainer import MultiLabelTrainer
from src.util.logging import RunLogger
from src.util.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-config", required=True)
    p.add_argument("--generate", action="store_true", help="write the 4 configs to configs/grid and exit")
    p.add_argument("--out-dir", default="configs/grid")
    p.add_argument("--limit", type=int, default=None, help="run only the first N conditions")
    p.add_argument("--offset", type=int, default=0, help="skip the first N conditions")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--eval-split", default="val", choices=["train", "val", "test"])
    return p.parse_args()


def _auto_workers(cfg, device: torch.device) -> int:
    candidates = [4, 8, 12, 16]
    gpu_warmup(device, cfg.data.batch_size, cfg.data.image_size)
    print("num_workers sweep:", end="", flush=True)
    results = {}
    for nw in candidates:
        try:
            rate = bench(cfg, nw, "train", n_batches=20, n_warmup=5, device=device)
            results[nw] = rate
            print(f"  {nw}→{rate:.0f}", end="", flush=True)
        except Exception:
            pass
    print()
    best = max(results, key=results.get) if results else cfg.data.num_workers
    print(f"best num_workers={best}  (was {cfg.data.num_workers})")
    return best


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
        best_ckpt = cfg.run_dir() / "checkpoints" / "best_val_auroc_macro.pt"
        if best_ckpt.exists():
            missing, unexpected = model.load_state_dict(
                torch.load(best_ckpt, map_location=device, weights_only=True), strict=False
            )
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    base.data.num_workers = _auto_workers(base, device)

    configs = build_grid(base)
    if args.epochs is not None:
        for c in configs:
            c.training.epochs = args.epochs

    to_run = configs[args.offset: args.offset + args.limit if args.limit else None]
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
