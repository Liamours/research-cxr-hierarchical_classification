"""Inference entry point — predict conditions for a single chest X-ray.

    uv run python src/script/run_inference.py \
      --config configs/grid/densenet121_xrv__seg-off__uq-off.yaml \
      --ckpt-name best_val_auroc_macro \
      --image path/to/cxr.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.experiment_config import ExperimentConfig
from src.inference.predict import CxrPredictor
from src.util.logging import RunLogger


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--ckpt-name", default="best_val_auroc_macro",
                    choices=["last", "best_val_loss", "best_val_auroc_macro", "best_val_f1_macro"])
    p.add_argument("--image", required=True)
    p.add_argument("--top-k", type=int, default=14)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    ckpt = args.checkpoint or (cfg.run_dir() / "checkpoints" / f"{args.ckpt_name}.pt")

    predictor = CxrPredictor(cfg, checkpoint=ckpt)
    if predictor.loaded:
        print(f"Loaded checkpoint: {predictor.loaded}")
    else:
        print(f"WARNING: no checkpoint at {ckpt}; using randomly-initialized weights.")
    print(f"\n=== Prediction: {args.image} ===")
    if cfg.uq.method == "mc_dropout":
        ranked = predictor.predict_with_uncertainty(args.image)
        print(f"(MC Dropout, {cfg.uq.mc_passes} passes — prob +/- std [epistemic/aleatoric])")
        for cond, mean, var, epistemic, aleatoric in ranked[: args.top_k]:
            print(f"  {cond:30s} {mean:.4f} +/- {var ** 0.5:.4f}"
                  f"  [epi={epistemic:.4f} alea={aleatoric:.4f}]")
        top = {"condition": ranked[0][0], "prob": round(ranked[0][1], 4)}
    else:
        ranked = predictor.predict_image(args.image)
        for cond, prob in ranked[: args.top_k]:
            print(f"  {cond:30s} {prob:.4f}")
        top = {"condition": ranked[0][0], "prob": round(ranked[0][1], 4)}

    logger = RunLogger(cfg.run_dir(), level=cfg.logging.level)
    logger.event("inference", image=str(args.image), backbone=cfg.model.backbone,
                 uq=cfg.uq.method, top=top, checkpoint=str(predictor.loaded))


if __name__ == "__main__":
    main()
