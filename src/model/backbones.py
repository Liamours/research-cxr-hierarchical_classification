"""Vision backbones for CXR classification.

Each builder returns (module, feature_dim, norm_kind). The module maps an
input image batch to a pooled feature vector; the head (nn.Sequential in
classifier.py) maps that to logits. norm_kind must agree with transforms.norm_kind_for_backbone so the
dataset normalizes correctly.

  densenet121_xrv  torchxrayvision DenseNet121 (weights=densenet121-res224-all),
                   1-channel grayscale [-1024, 1024], 1024-dim feature vector.
                   Pretrained on NIH+PadChest+CheXpert+MIMIC+OpenI+RSNA (CXR-domain).

  vit_base_imagenet  timm vit_base_patch16_224, ImageNet-1k pretrained (generic
                   natural-image domain, not CXR-specific -- no CXR-pretrained ViT
                   checkpoint exists publicly). 3-channel (grayscale replicated),
                   ImageNet mean/std, 768-dim feature vector. Architecture-comparison
                   arm against densenet121_xrv; the pretraining-domain mismatch is a
                   known confound, not a bug.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _XrvFeatures(nn.Module):
    """Wraps a torchxrayvision DenseNet to emit pooled 1024-dim features."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        f = self.net.features(x)
        f = F.relu(f)  # not inplace: keeps backward hooks on features valid (Grad-CAM)
        f = F.adaptive_avg_pool2d(f, (1, 1)).flatten(1)
        return f


def build_backbone(name: str, pretrained: bool = True):
    if name == "densenet121_xrv":
        import torchxrayvision as xrv
        weights = "densenet121-res224-all" if pretrained else None
        net = xrv.models.DenseNet(weights=weights)
        return _XrvFeatures(net), 1024, "xrv"
    if name == "vit_base_imagenet":
        import timm
        net = timm.create_model("vit_base_patch16_224", pretrained=pretrained, num_classes=0)
        return net, net.num_features, "imagenet"
    raise ValueError(f"unknown backbone: {name!r} (supported: 'densenet121_xrv', 'vit_base_imagenet')")


def expand_conv_in_channels(conv: nn.Conv2d, new_in: int) -> nn.Conv2d:
    """Return a Conv2d with new_in input channels, copying the original weights
    and initializing extra channels with the mean of the originals."""
    if new_in == conv.in_channels:
        return conv
    new = nn.Conv2d(
        new_in, conv.out_channels, kernel_size=conv.kernel_size,
        stride=conv.stride, padding=conv.padding, bias=conv.bias is not None,
    )
    with torch.no_grad():
        new.weight[:, : conv.in_channels] = conv.weight
        if new_in > conv.in_channels:
            mean_w = conv.weight.mean(dim=1, keepdim=True)
            new.weight[:, conv.in_channels:] = mean_w.repeat(1, new_in - conv.in_channels, 1, 1)
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new


def adapt_backbone_input(backbone, backbone_name: str, extra_channels: int) -> None:
    """In-place expand the first conv of a backbone by extra_channels."""
    if extra_channels <= 0:
        return
    if backbone_name == "densenet121_xrv":
        conv0 = backbone.net.features.conv0
        backbone.net.features.conv0 = expand_conv_in_channels(conv0, conv0.in_channels + extra_channels)
    else:
        raise ValueError(f"no input adaptation defined for backbone {backbone_name!r}")
