"""Image transforms with backbone-aware normalization.

Preprocessed images are stored as 224x224x3 uint8 PNG (see preprocessing
protocol), so the transform does not re-crop from a larger canvas; it only
ensures size, applies train-time augmentation, and normalizes per backbone.

Normalization regimes:
  xrv       -> DenseNet121 (torchxrayvision): 1-channel grayscale, [-1024, 1024].
  imagenet  -> fallback for any non-xrv backbone: 3-channel ImageNet mean/std.
"""

from __future__ import annotations

import torch
from torchvision import transforms

# Clinically-safe CXR defaults: small affine jitter + mild photometric jitter,
# hflip off (laterality). Mirrors typical CheXpert/CheXNet-style pipelines.
DEFAULT_AUG = {
    "hflip": False,
    "rotation_deg": 10.0,
    "translate": 0.05,
    "scale_jitter": 0.05,
    "brightness": 0.1,
    "contrast": 0.1,
}


def norm_kind_for_backbone(backbone: str) -> str:
    if backbone.startswith("densenet121_xrv"):
        return "xrv"
    raise ValueError(f"unknown backbone for normalization: {backbone!r}")


def _xrv_normalize(t: torch.Tensor) -> torch.Tensor:
    # t in [0, 1] from ToTensor -> torchxrayvision [-1024, 1024] convention.
    return (2.0 * t - 1.0) * 1024.0


def _aug_ops(p: dict) -> list:
    ops = []
    if p.get("hflip"):
        ops.append(transforms.RandomHorizontalFlip())
    deg = p.get("rotation_deg", 0.0)
    tr = p.get("translate", 0.0)
    sc = p.get("scale_jitter", 0.0)
    if deg or tr or sc:
        ops.append(transforms.RandomAffine(
            degrees=deg,
            translate=(tr, tr) if tr else None,
            scale=(1 - sc, 1 + sc) if sc else None,
        ))
    b, c = p.get("brightness", 0.0), p.get("contrast", 0.0)
    if b or c:
        ops.append(transforms.ColorJitter(brightness=b, contrast=c))
    return ops


def build_transform(
    image_size: int = 224,
    augment: bool = False,
    norm_kind: str = "imagenet",
    aug_params: dict | None = None,
) -> transforms.Compose:
    # Segmentation mask conditioning (Factor 2) is applied per-sample in the
    # dataset __getitem__ (src/data/segmentation.py), not here, because masks
    # are keyed by image_id. build_transform only produces the image tensor.
    aug = _aug_ops(aug_params if aug_params is not None else DEFAULT_AUG) if augment else []

    if norm_kind != "xrv":
        raise ValueError(f"unknown norm_kind: {norm_kind!r}")
    ops = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size)),
        *aug,
        transforms.ToTensor(),
        transforms.Lambda(_xrv_normalize),
    ]
    return transforms.Compose(ops)
