"""Evaluation entry point. Uses the trained model only (no refit).

    uv run python src/script/run_evaluate.py --config configs/grid/densenet121_xrv__seg-off__uq-off.yaml --split val
    uv run python src/script/run_evaluate.py --config <cfg> --ckpt-name best_val_f1_macro
    uv run python src/script/run_evaluate.py --config <cfg> --checkpoint result/<run>/checkpoints/last.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from src.config.experiment_config import ExperimentConfig
from src.data.loader import build_loaders
from src.evaluate.evaluator import evaluate_model
from src.model.classifier import build_model_from_cfg
from src.util.logging import RunLogger
from src.util.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--ckpt-name", default="best_val_auroc_macro",
                    choices=["last", "best_val_loss", "best_val_auroc_macro",
                             "best_val_f1_macro", "best_val_aurc_macro"])
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--run-dir", default=None, help="override run directory for checkpoint lookup and output")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    set_seed(cfg.experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model_from_cfg(cfg, pretrained=False)  # weights from checkpoint
    base_dir = Path(args.run_dir) if args.run_dir else cfg.run_dir()
    ckpt = Path(args.checkpoint) if args.checkpoint else base_dir / "checkpoints" / f"{args.ckpt_name}.pt"
    if ckpt.exists():
        missing, unexpected = model.load_state_dict(
            torch.load(ckpt, map_location="cpu", weights_only=True), strict=False
        )
        if unexpected:
            print(f"  ignored keys (checkpoint-only buffers): {unexpected}")
        print(f"Loaded checkpoint: {ckpt}")
    else:
        print(f"WARNING: checkpoint not found ({ckpt}); evaluating randomly-initialized weights.")
    model.to(device)

    loaders = build_loaders(cfg)
    loader = loaders.get(args.split)
    if loader is None:
        print(f"No data for split '{args.split}' at {cfg.data.label_csv}. Nothing to evaluate.")
        return

    logger = RunLogger(base_dir, level=cfg.logging.level)
    report = evaluate_model(model, loader, device, cfg, split=args.split, logger=logger,
                            run_dir=args.run_dir)

    conditions = cfg.resolved_conditions()
    print(f"\n=== Evaluation ({args.split}) — {cfg.experiment.name} ===")
    if "auroc" in report:
        print(f"AUROC macro={report['auroc']['macro']:.4f}  micro={report['auroc']['micro']:.4f}")
    if "f1" in report:
        print(f"F1    macro={report['f1']['macro']:.4f}  micro={report['f1']['micro']:.4f}")
    if "map" in report:
        print(f"mAP   macro={report['map']['macro']:.4f}")
    if "aurc" in report:
        print(f"AURC  flat={report['aurc']['flat']:.4f}  macro={report['aurc']['macro']:.4f}")
    if "calibration" in report:
        print(f"ECE={report['calibration']['ece']:.4f}")
    if "uq" in report:
        uq = report["uq"]
        print(f"MC-Dropout mean_var={uq['mean_variance']:.4f}  sanity={uq['sanity']['passes']}")
        if "hierarchical_fallback" in uq:
            print(f"  fallback suppressed={uq['hierarchical_fallback']['total_suppressed']}")

    if "auroc" in report:
        print("\nper-class AUROC:")
        for c in conditions:
            v = report["auroc"]["per_class"].get(c, float("nan"))
            print(f"  {c:38s} {v:.4f}" if v == v else f"  {c:38s}   n/a")

    if "map" in report:
        print("\nper-class AP:")
        for c in conditions:
            v = report["map"]["per_class"].get(c, float("nan"))
            print(f"  {c:38s} {v:.4f}" if v == v else f"  {c:38s}   n/a")

    if "aurc" in report:
        print("\nper-class AURC:")
        for c in conditions:
            v = report["aurc"]["per_class"].get(c, float("nan"))
            print(f"  {c:38s} {v:.4f}" if v == v else f"  {c:38s}   n/a")

    print(f"\nWritten: {base_dir / ('eval_metrics_' + args.split + '.json')}")


if __name__ == "__main__":
    main()
