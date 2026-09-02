"""Training entry point. Every run goes through a YAML config. After training,
the best checkpoint is evaluated on val and test (skipped for --max-samples smokes).

    uv run python src/script/run_train.py --config configs/densenet121_xrv__flat.yaml
    uv run python src/script/run_train.py --config configs/densenet121_xrv__flat.yaml --max-samples 64
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.utils.data import DataLoader, Subset

from src.config.experiment_config import ExperimentConfig
from src.data.loader import build_loaders
from src.evaluate.evaluator import evaluate_model
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
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
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

    # Drop the train loader's persistent workers before final eval -- with
    # num_workers>0 every DataLoader in `loaders` keeps its worker pool alive
    # for the loader's lifetime (persistent_workers=True), so train+val+test
    # can otherwise all be resident at once. Windows is slow to reclaim that
    # committed memory, which crashed the val->test transition (WinError 1455,
    # paging file too small) on two separate runs before this fix.
    del trainer.train_loader, loaders["train"], trainer
    gc.collect()

    if not args.max_samples:                       # skip eval for smoke runs
        _final_eval(cfg, model, loaders, device, logger)
    print(f"Run dir: {cfg.run_dir()}")


def _final_eval(cfg, model, loaders, device, logger):
    """Evaluate the best-AUROC checkpoint on val and test, writing
    eval_metrics_<split>.json + predictions into the run dir."""
    ckpt = cfg.run_dir() / "checkpoints" / "best_val_auroc_macro.pt"
    if ckpt.exists():
        model.load_state_dict(
            torch.load(ckpt, map_location=device, weights_only=True), strict=False)
        logger.log(f"final eval: loaded {ckpt.name}")
    else:
        logger.log(f"final eval: {ckpt.name} missing; evaluating current weights")
    for split in ("val", "test"):
        loader = loaders.get(split)
        if loader is not None:
            evaluate_model(model, loader, device, cfg, split=split, logger=logger)
            # Release this split's persistent workers before the next split
            # builds its own -- same reasoning as the train-loader drop above.
            del loaders[split], loader
            gc.collect()


if __name__ == "__main__":
    main()
