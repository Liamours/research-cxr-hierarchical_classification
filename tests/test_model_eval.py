"""Model, loss, metrics, uncertainty, and XAI tests (fast; no training)."""

from __future__ import annotations

import numpy as np
import torch

from src.data.label_space import CANONICAL_LABELS
from src.evaluate.evaluator import evaluate_model
from src.evaluate.metrics import compute_classification_metrics
from src.data.loader import build_loaders
from src.model.classifier import CxrClassifier, build_model_from_cfg
from src.model.mc_dropout import mc_dropout_predict
from src.train.losses import masked_bce_with_logits
from src.util.seed import set_seed
from src.xai.gradcam import GradCAM, resolve_target_layer


def test_backbones_forward_shapes():
    for backbone, ch in [("densenet121_xrv", 1)]:
        m = CxrClassifier(backbone, pretrained=False)
        assert tuple(m(torch.randn(2, ch, 224, 224)).shape) == (2, len(CANONICAL_LABELS))


def test_determinism_same_seed():
    x = torch.randn(2, 1, 224, 224)
    set_seed(42); m1 = CxrClassifier("densenet121_xrv", pretrained=False).eval()
    set_seed(42); m2 = CxrClassifier("densenet121_xrv", pretrained=False).eval()
    with torch.no_grad():
        assert torch.allclose(m1(x), m2(x), atol=1e-6)


def test_grad_flow_and_finite_loss():
    m = CxrClassifier("densenet121_xrv", pretrained=False)
    out = m(torch.randn(2, 1, 224, 224))
    y = (torch.rand(2, len(CANONICAL_LABELS)) > 0.5).float()
    loss = masked_bce_with_logits(out, y, torch.ones(2, len(CANONICAL_LABELS)))
    loss.backward()
    assert torch.isfinite(loss)
    grads = [p.grad for p in m.head.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_masked_loss_ignores_nan_columns():
    probs = np.random.default_rng(0).random((20, len(CANONICAL_LABELS)))
    t = (np.random.default_rng(1).random((20, len(CANONICAL_LABELS))) > 0.5).astype(float)
    m = np.ones((20, len(CANONICAL_LABELS))); m[:, 5] = 0
    cm = compute_classification_metrics(probs, t, m)
    assert np.isnan(cm["auroc"]["per_class"][CANONICAL_LABELS[5]])


def test_mc_dropout_mean_var():
    m = CxrClassifier("densenet121_xrv", pretrained=False)
    mean, var = mc_dropout_predict(m, torch.randn(3, 1, 224, 224), n_passes=8)
    assert tuple(mean.shape) == (3, len(CANONICAL_LABELS)) and float(var.sum()) > 0


def test_eval_no_refit(make_cfg):
    cfg = make_cfg(name="norefit")
    model = build_model_from_cfg(cfg, pretrained=False)
    w = model.head.fc.weight.detach().clone()
    loaders = build_loaders(cfg)
    evaluate_model(model, loaders["val"], torch.device("cpu"), cfg, split="val")
    assert torch.equal(model.head.fc.weight.detach(), w)
    assert (cfg.run_dir() / "predictions" / "val.csv").exists()


def test_gradcam():
    m = CxrClassifier("densenet121_xrv", pretrained=False)
    target = resolve_target_layer(m)
    cam_fn = GradCAM(m, target)
    cam = cam_fn(torch.randn(1, 1, 224, 224))[0].numpy()
    cam_fn.remove()
    assert cam.shape == (224, 224) and cam.min() >= 0 and cam.max() <= 1.0
