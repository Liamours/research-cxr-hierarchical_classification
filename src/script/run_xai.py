"""Grad-CAM localization for a single chest X-ray, config + checkpoint driven.

    uv run python src/script/run_xai.py \
      --config configs/grid/densenet121_xrv__seg-off__uq-off.yaml \
      --ckpt-name best_val_auroc_macro \
      --image path/to/cxr.png

Writes a heatmap overlay PNG into <run_dir>/xai/ and logs an "xai" event.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image

from src.config.experiment_config import ExperimentConfig
from src.data.label_space import CANONICAL_LABELS
from src.data.segmentation import apply_seg_to_tensor
from src.data.transforms import build_transform, norm_kind_for_backbone
from src.model.classifier import build_model_from_cfg
from src.util.logging import RunLogger
from src.xai.gradcam import GradCAM, resolve_target_layer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--ckpt-name", default="best_val_auroc_macro",
                    choices=["last", "best_val_loss", "best_val_auroc_macro",
                             "best_val_f1_macro", "best_val_aurc_macro"])
    p.add_argument("--image", required=True)
    p.add_argument("--class-name", default=None, help="condition to explain; default = top prediction")
    return p.parse_args()


def save_overlay(image_path, cam, out_path, alpha=0.5, image_size=224):
    base = np.array(Image.open(image_path).convert("L").resize((image_size, image_size)))
    base = np.stack([base] * 3, axis=-1).astype(float) / 255.0
    heat = cm.jet(np.asarray(cam))[..., :3]
    blend = ((1 - alpha) * base + alpha * heat)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((blend * 255).astype(np.uint8)).save(out_path)


def main():
    args = parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model_from_cfg(cfg, pretrained=False)
    ckpt = Path(args.checkpoint) if args.checkpoint else cfg.run_dir() / "checkpoints" / f"{args.ckpt_name}.pt"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    else:
        print(f"WARNING: no checkpoint at {ckpt}; using random weights.")
    model.to(device)

    transform = build_transform(cfg.data.image_size, augment=False,
                                norm_kind=norm_kind_for_backbone(cfg.model.backbone))
    t = transform(Image.open(args.image).convert("RGB"))
    t = apply_seg_to_tensor(t, cfg.seg, cfg.data.image_size, Path(args.image).stem)
    x = t.unsqueeze(0).to(device)

    class_idx = CANONICAL_LABELS.index(args.class_name) if args.class_name else None
    target = resolve_target_layer(model, cfg.xai.target_layer)
    cam_fn = GradCAM(model, target)
    cam = cam_fn(x, class_idx)[0].cpu().numpy()
    cam_fn.remove()

    if class_idx is None:
        with torch.no_grad():
            class_idx = int(model(x).argmax(1).item())
    cond = CANONICAL_LABELS[class_idx]

    out_path = cfg.run_dir() / "xai" / f"{Path(args.image).stem}__{cond}__gradcam.png"
    save_overlay(args.image, cam, out_path, alpha=cfg.xai.overlay_alpha, image_size=cfg.data.image_size)

    logger = RunLogger(cfg.run_dir(), level=cfg.logging.level)
    logger.event("xai", method="gradcam", backbone=cfg.model.backbone,
                 image=str(args.image), condition=cond, cam_max=float(cam.max()),
                 overlay=str(out_path))
    print(f"Grad-CAM [{cfg.model.backbone}] condition={cond}  overlay -> {out_path}")


if __name__ == "__main__":
    main()
