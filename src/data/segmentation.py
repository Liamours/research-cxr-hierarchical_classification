"""Factor 2 — segmentation conditioning with CheXmask-U anatomical masks.

Two conditioning methods, switchable by config (cfg.seg.method):
  concat_channel  append the binary mask as an extra input channel (LOCKED as
                  the canonical ablation method; preserves the full image and
                  adds anatomy as a prior).
  crop            multiply the image by the mask (zero outside the anatomy).

Mask conditioning is applied in the dataset __getitem__ (after the image
transform), not inside build_transform, because masks are per-sample and keyed
by image_id. The real provider reads precomputed CheXmask-U masks from disk;
the synthetic provider is for smoke tests only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


class MaskProvider:
    def get(self, image_id: str) -> torch.Tensor | None:
        raise NotImplementedError


class SyntheticMaskProvider(MaskProvider):
    """Deterministic centred-ellipse mask. Smoke only — not a real segmentation."""

    def __init__(self, size: int = 224):
        self.size = size

    def get(self, image_id: str) -> torch.Tensor:
        h = w = self.size
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy, rx, ry = w / 2, h / 2, w * 0.3, h * 0.4
        m = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1.0
        return torch.from_numpy(m.astype(np.float32))[None]  # (1, H, W)


class CheXmaskUProvider(MaskProvider):
    """Loads precomputed CheXmask-U masks (one PNG per image_id) from disk."""

    def __init__(self, mask_root: str | Path, size: int = 224):
        self.mask_root = Path(mask_root)
        self.size = size
        if not self.mask_root.exists():
            raise FileNotFoundError(self.mask_root)

    def get(self, image_id: str) -> torch.Tensor | None:
        p = self.mask_root / f"{image_id}.png"
        if not p.exists():
            return None
        m = Image.open(p).convert("L").resize((self.size, self.size), Image.NEAREST)
        return torch.from_numpy((np.array(m) > 127).astype(np.float32))[None]


def build_mask_provider(seg_cfg, size: int) -> MaskProvider:
    if seg_cfg.mask_source == "synthetic":
        return SyntheticMaskProvider(size)
    if seg_cfg.mask_source == "chexmask_u":
        return CheXmaskUProvider(seg_cfg.mask_root, size)
    raise ValueError(f"unknown mask_source: {seg_cfg.mask_source!r}")


def apply_mask_conditioning(image: torch.Tensor, mask: torch.Tensor, method: str) -> torch.Tensor:
    if method == "concat_channel":
        return torch.cat([image, mask], dim=0)
    if method == "crop":
        return image * mask
    raise ValueError(f"unknown seg method: {method!r}")


def seg_extra_channels(method: str) -> int:
    return 1 if method == "concat_channel" else 0


def apply_seg_to_tensor(pixel_values: torch.Tensor, seg_cfg, image_size: int,
                        image_id: str) -> torch.Tensor:
    """Apply seg conditioning to a single (C,H,W) tensor for inference/XAI, so
    the input channel count matches a seg-trained checkpoint. Missing masks fall
    back to zeros (keeps channel count correct)."""
    if not seg_cfg.enabled:
        return pixel_values
    try:
        provider = build_mask_provider(seg_cfg, image_size)
        mask = provider.get(image_id)
    except FileNotFoundError:
        mask = None
    if mask is None:
        mask = torch.zeros((1, pixel_values.shape[-2], pixel_values.shape[-1]))
    return apply_mask_conditioning(pixel_values, mask, seg_cfg.method)
