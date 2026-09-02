"""Benchmark DataLoader throughput across num_workers values.

    uv run python src/script/bench_num_workers.py --config configs/grid_base.yaml
    uv run python src/script/bench_num_workers.py --config configs/grid_base.yaml --batches 100 --warmup 20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config.experiment_config import ExperimentConfig
from src.data.dataset import CxrClsDataset
from src.data.transforms import build_transform, norm_kind_for_backbone
from src.util.seed import seed_worker


def gpu_warmup(device: torch.device, batch_size: int, image_size: int, n: int = 10):
    if device.type != "cuda":
        return
    dummy = torch.randn(batch_size, 1, image_size, image_size, device=device)
    for _ in range(n):
        _ = dummy * 2 + 1
    torch.cuda.synchronize()


def make_loader(cfg, num_workers: int, split: str) -> DataLoader:
    norm_kind = norm_kind_for_backbone(cfg.model.backbone)
    transform = build_transform(cfg.data.image_size, augment=False,
                                norm_kind=norm_kind, aug_params={})
    ds = CxrClsDataset(cfg.data.label_csv, cfg.data.image_root, split, transform,
                       skip_missing_check=cfg.data.skip_missing_check)
    pin = torch.cuda.is_available()
    return DataLoader(
        ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker if num_workers > 0 else None,
    )


def bench(cfg, num_workers: int, split: str, n_batches: int, n_warmup: int,
          device: torch.device) -> float:
    loader = make_loader(cfg, num_workers, split)
    it = iter(loader)

    for _ in tqdm(range(n_warmup), desc=f"nw={num_workers:>2} warmup", leave=False):
        batch = next(it)
        if device.type == "cuda":
            batch["pixel_values"].to(device, non_blocking=True)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in tqdm(range(n_batches), desc=f"nw={num_workers:>2} bench ", leave=False):
        batch = next(it)
        if device.type == "cuda":
            batch["pixel_values"].to(device, non_blocking=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return n_batches * cfg.data.batch_size / elapsed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--split", default="train", choices=["train", "val"])
    p.add_argument("--batches", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--workers", nargs="+", type=int, default=[0, 2, 4, 8, 12, 16])
    return p.parse_args()


def main():
    args = parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"device={device}  batch_size={cfg.data.batch_size}  "
          f"split={args.split}  warmup={args.warmup}  batches={args.batches}")

    gpu_warmup(device, cfg.data.batch_size, cfg.data.image_size)

    print(f"\n{'num_workers':>12}  {'img/s':>10}")
    results = {}
    for nw in args.workers:
        try:
            rate = bench(cfg, nw, args.split, args.batches, args.warmup, device)
            results[nw] = rate
            print(f"{nw:>12}  {rate:>10.1f}")
        except StopIteration:
            print(f"{nw:>12}  SKIP (dataset too small for warmup+batches)")
        except Exception as e:
            print(f"{nw:>12}  ERROR: {e}")

    if results:
        best = max(results, key=results.get)
        print(f"\nbest: num_workers={best}  ({results[best]:.1f} img/s)")


if __name__ == "__main__":
    main()
