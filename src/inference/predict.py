"""Inference — load a trained checkpoint and predict per-condition probabilities.

Preprocessing is identical to validation/test (build_transform with
augment=False and the backbone's norm_kind), so train/inference parity holds.
No weights are updated. predict_with_uncertainty runs MC Dropout (Factor 4,
uq.method == "mc_dropout") and returns per-condition mean + variance.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from src.data.hierarchy import edge_index_pairs
from src.data.segmentation import apply_mask_conditioning, build_mask_provider
from src.data.transforms import build_transform, norm_kind_for_backbone
from src.inference.hierarchical_fallback import apply_hierarchical_fallback
from src.model.classifier import build_model_from_cfg
from src.model.mc_dropout import mc_dropout_predict


class CxrPredictor:
    def __init__(self, cfg, checkpoint: str | Path | None = None, device=None):
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conditions = cfg.resolved_conditions()
        self.norm_kind = norm_kind_for_backbone(cfg.model.backbone)
        self.transform = build_transform(cfg.data.image_size, augment=False, norm_kind=self.norm_kind)
        self.use_amp = cfg.training.bf16 and self.device.type == "cuda"

        # Build mask provider once (not on every predict call).
        self._seg_provider = None
        if cfg.seg.enabled:
            try:
                self._seg_provider = build_mask_provider(cfg.seg, cfg.data.image_size)
            except FileNotFoundError:
                pass

        self.model = build_model_from_cfg(cfg, pretrained=False)
        if checkpoint is not None and Path(checkpoint).exists():
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state)
            self.loaded = str(checkpoint)
        else:
            self.loaded = None
        self.model.to(self.device).eval()

    def _load_tensor(self, image_path) -> torch.Tensor:
        img = Image.open(image_path).convert("RGB")
        t = self.transform(img)
        if self._seg_provider is not None:
            image_id = Path(image_path).stem
            mask = self._seg_provider.get(image_id)
            if mask is None:
                mask = torch.zeros((1, t.shape[-2], t.shape[-1]))
            t = apply_mask_conditioning(t, mask, self.cfg.seg.method)
        return t.unsqueeze(0)

    @torch.no_grad()
    def predict_image(self, image_path) -> list[tuple[str, float]]:
        x = self._load_tensor(image_path).to(self.device)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
            logits = self.model(x)
        probs = torch.sigmoid(logits)[0].float().cpu().tolist()
        return sorted(zip(self.conditions, probs), key=lambda kv: kv[1], reverse=True)

    @torch.no_grad()
    def predict_with_uncertainty(self, image_path, n_passes: int | None = None):
        """MC Dropout inference: returns (condition, mean_prob, variance, epistemic,
        aleatoric) ranked by mean probability. variance = epistemic + aleatoric."""
        passes = n_passes or self.cfg.uq.mc_passes
        x = self._load_tensor(image_path).to(self.device)
        mean, epistemic, aleatoric = mc_dropout_predict(self.model, x, n_passes=passes)
        mean, epistemic, aleatoric = mean[0].cpu(), epistemic[0].cpu(), aleatoric[0].cpu()
        var = epistemic + aleatoric
        quints = list(zip(self.conditions, mean.tolist(), var.tolist(),
                          epistemic.tolist(), aleatoric.tolist()))
        return sorted(quints, key=lambda t: t[1], reverse=True)

    @torch.no_grad()
    def predict_hierarchical(self, image_path, n_passes: int | None = None):
        """MC Dropout + hierarchical fallback.

        When variance for a child label exceeds cfg.uq.gate_threshold, the child
        prediction is suppressed (set to 0) and the parent label stands.

        Returns (condition, adjusted_prob, raw_mean, variance, suppressed) tuples
        ranked by adjusted probability. suppressed=True marks labels zeroed by fallback.
        """
        passes = n_passes or self.cfg.uq.mc_passes
        x = self._load_tensor(image_path).to(self.device)
        mean, epistemic, aleatoric = mc_dropout_predict(self.model, x, n_passes=passes)
        mean, var = mean[0].cpu(), (epistemic[0] + aleatoric[0]).cpu()

        pairs = edge_index_pairs(self.conditions)
        adjusted, log = apply_hierarchical_fallback(
            mean, var, pairs, gate_threshold=self.cfg.uq.gate_threshold
        )

        suppressed_child_indices = {
            e["child_idx"] for e in log["edges"] if e["suppressed"] > 0
        }
        results = [
            (
                self.conditions[j],
                float(adjusted[j]),
                float(mean[j]),
                float(var[j]),
                j in suppressed_child_indices,
            )
            for j in range(len(self.conditions))
        ]
        return sorted(results, key=lambda t: t[1], reverse=True), log
