"""Device helpers: pick compute device, check bf16 support."""

from __future__ import annotations

import torch


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def bf16_supported(device: torch.device | None = None) -> bool:
    """True only when bf16 autocast is actually usable: a CUDA device whose
    compute capability supports bfloat16 (Ampere/Ada/Blackwell yes; older cards
    and CPU no). Lets the same training config run on any GPU without bf16 errors."""
    if device is not None and device.type != "cuda":
        return False
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


