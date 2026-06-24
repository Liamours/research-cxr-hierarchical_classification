"""Grad-CAM for DenseNet121 (TorchXRayVision backbone).

Hooks the last convolutional feature map of DenseNet121, weights activations
by GAP of gradients (standard Grad-CAM), and upsamples to input resolution.
Returns a per-sample heatmap normalized to [0, 1].
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def resolve_target_layer(classifier, override: str | None = None):
    """Return the target module for Grad-CAM.

    If override is a dotted module path (cfg.xai.target_layer != "auto"),
    that module is used. Otherwise resolves per backbone.
    """
    if override and override != "auto":
        mod = classifier
        for part in override.split("."):
            mod = getattr(mod, part)
        return mod
    name = classifier.backbone_name
    if name == "densenet121_xrv":
        return classifier.backbone.net.features
    raise ValueError(f"no Grad-CAM target defined for backbone {name!r}")


class GradCAM:
    def __init__(self, model, target_module):
        self.model = model
        self._acts = None
        self._grads = None
        self._h1 = target_module.register_forward_hook(self._fwd)
        self._h2 = target_module.register_full_backward_hook(self._bwd)

    def _fwd(self, _m, _inp, out):
        self._acts = out

    def _bwd(self, _m, _gin, gout):
        self._grads = gout[0]

    def remove(self):
        self._h1.remove()
        self._h2.remove()

    def __call__(self, x, class_idx=None) -> torch.Tensor:
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1)
        elif isinstance(class_idx, int):
            class_idx = torch.full((x.size(0),), class_idx, device=x.device, dtype=torch.long)
        score = logits.gather(1, class_idx.view(-1, 1)).sum()
        score.backward()

        weights = self._grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self._acts).sum(dim=1))  # (B, h, w)
        cam = F.interpolate(cam.unsqueeze(1), size=x.shape[-2:],
                            mode="bilinear", align_corners=False).squeeze(1)
        cam = cam - cam.amin(dim=(1, 2), keepdim=True)
        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
        return cam.detach()
