"""Tests for modules added/changed this session.

Covers: HBCELoss penalty logic, hierarchical_fallback, vectorized CxrClsDataset,
AURC plug-in estimator, calibration ECE, statistics (bootstrap CI / compare_aurc /
mcnemar), leakage check, and edge_index_pairs.

All tests are fast (no model forward, no disk I/O beyond tmp_path).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.label_space import CANONICAL_LABELS
from src.data.hierarchy import edge_index_pairs, validate_edges
from src.train.losses import BAFLLoss, HBCELoss, MaskedBCELoss, train_class_weights
from src.inference.hierarchical_fallback import apply_hierarchical_fallback
from src.evaluate.calibration import compute_calibration, expected_calibration_error
from src.evaluate.selective import compute_aurc, aurc_flat, per_class_aurc
from src.evaluate.statistics import (
    bootstrap_metric_ci, compare_aurc, mcnemar_comparison,
)
from src.script.run_leakage_check import check_image_leakage, check_patient_leakage


# ---------------------------------------------------------------------------
# HBCELoss
# ---------------------------------------------------------------------------

def _logit(p: float) -> float:
    return math.log(p / (1 - p))


def test_hbce_penalty_fires_when_child_positive_parent_negative():
    # edge (0 → 1): parent col 0, child col 1
    pairs = [(0, 1)]
    loss_fn = HBCELoss(pairs, lam=1.0)
    # child confidently positive (>0.5), parent confidently negative (<0.5)
    logits = torch.tensor([[_logit(0.1), _logit(0.9)]])   # (1, 2)
    targets = torch.zeros(1, 2)
    mask    = torch.ones(1, 2)
    loss = loss_fn(logits, targets, mask)
    assert loss_fn._last_penalty > 0.0


def test_hbce_no_penalty_when_consistent():
    pairs = [(0, 1)]
    loss_fn = HBCELoss(pairs, lam=1.0)
    # child negative, parent positive — no violation
    logits = torch.tensor([[_logit(0.9), _logit(0.1)]])
    targets = torch.zeros(1, 2)
    mask    = torch.ones(1, 2)
    loss_fn(logits, targets, mask)
    assert loss_fn._last_penalty == pytest.approx(0.0, abs=1e-6)


def test_hbce_penalty_zero_when_edge_masked():
    pairs = [(0, 1)]
    loss_fn = HBCELoss(pairs, lam=1.0)
    logits = torch.tensor([[_logit(0.1), _logit(0.9)]])
    targets = torch.zeros(1, 2)
    mask = torch.zeros(1, 2)   # neither parent nor child applicable
    loss_fn(logits, targets, mask)
    assert loss_fn._last_penalty == pytest.approx(0.0, abs=1e-6)


def test_hbce_equals_bce_when_lam_zero():
    pairs = [(0, 1)]
    hbce = HBCELoss(pairs, lam=0.0)
    bce  = MaskedBCELoss()
    logits = torch.randn(4, 5)
    targets = (torch.rand(4, 5) > 0.5).float()
    mask    = torch.ones(4, 5)
    assert hbce(logits, targets, mask).item() == pytest.approx(
        bce(logits, targets, mask).item(), rel=1e-5
    )


def test_hbce_no_edges_equals_bce():
    hbce = HBCELoss([], lam=1.0)   # empty edge list
    bce  = MaskedBCELoss()
    logits = torch.randn(3, 4)
    targets = (torch.rand(3, 4) > 0.5).float()
    mask    = torch.ones(3, 4)
    assert hbce(logits, targets, mask).item() == pytest.approx(
        bce(logits, targets, mask).item(), rel=1e-5
    )


# ---------------------------------------------------------------------------
# BAFLLoss
# ---------------------------------------------------------------------------

def test_bafl_gamma_ramps_and_clamps():
    loss_fn = BAFLLoss(torch.ones(2), gamma_init=0.5, gamma_final=2.5, t_warmup=10)
    loss_fn.set_epoch(0)
    assert loss_fn.gamma == pytest.approx(0.5)
    loss_fn.set_epoch(5)
    assert loss_fn.gamma == pytest.approx(1.5)
    loss_fn.set_epoch(10)
    assert loss_fn.gamma == pytest.approx(2.5)
    loss_fn.set_epoch(999)  # past warmup: clamp, don't overshoot
    assert loss_fn.gamma == pytest.approx(2.5)


def test_bafl_masks_like_masked_bce():
    loss_fn = BAFLLoss(torch.ones(3), gamma_init=0.0, gamma_final=0.0)  # gamma=0 -> focal term is 1
    bce = MaskedBCELoss()
    logits = torch.randn(4, 3)
    targets = (torch.rand(4, 3) > 0.5).float()
    mask = (torch.rand(4, 3) > 0.3).float()
    assert loss_fn(logits, targets, mask).item() == pytest.approx(
        bce(logits, targets, mask).item(), rel=1e-5
    )


def test_bafl_downweights_confident_correct_predictions():
    loss_fn = BAFLLoss(torch.ones(1), gamma_init=2.0, gamma_final=2.0)
    confident_correct = torch.tensor([[_logit(0.95)]])
    unsure_correct = torch.tensor([[_logit(0.6)]])
    targets = torch.ones(1, 1)
    mask = torch.ones(1, 1)
    loss_confident = loss_fn(confident_correct, targets, mask).item()
    loss_unsure = loss_fn(unsure_correct, targets, mask).item()
    assert loss_confident < loss_unsure


def test_train_class_weights_rarer_class_gets_higher_weight(tmp_path):
    conditions = ["Common", "Rare"]
    df = pd.DataFrame({
        "split": ["train"] * 100,
        "Common": [1.0] * 50 + [0.0] * 50,
        "Rare": [1.0] * 2 + [0.0] * 98,
    })
    csv_path = tmp_path / "labels.csv"
    df.to_csv(csv_path, index=False)
    w = train_class_weights(csv_path, conditions, beta=0.999)
    assert w[1] > w[0]  # Rare (n=2) weighted higher than Common (n=50)
    assert w.mean().item() == pytest.approx(1.0, abs=1e-4)  # both active -> same as global mean


def test_train_class_weights_dead_labels_dont_dilute_normalization(tmp_path):
    conditions = ["Common", "Rare", "Dead"]
    df = pd.DataFrame({
        "split": ["train"] * 100,
        "Common": [1.0] * 50 + [0.0] * 50,
        "Rare": [1.0] * 2 + [0.0] * 98,
        "Dead": [np.nan] * 100,   # never annotated -> always mask=0 in training
    })
    csv_path = tmp_path / "labels.csv"
    df.to_csv(csv_path, index=False)
    w = train_class_weights(csv_path, conditions, beta=0.999)
    assert w[:2].mean().item() == pytest.approx(1.0, abs=1e-4)  # active labels normalize to 1
    assert w.mean().item() != pytest.approx(1.0, abs=1e-4)      # Dead's presence skews the raw mean


# ---------------------------------------------------------------------------
# Hierarchical fallback
# ---------------------------------------------------------------------------

def test_fallback_suppresses_uncertain_child():
    mean = torch.tensor([[0.8, 0.8]])   # both parent (0) and child (1) predicted positive
    var  = torch.tensor([[0.1, 0.9]])   # child highly uncertain
    pairs = [(0, 1)]
    adj, log = apply_hierarchical_fallback(mean, var, pairs, gate_threshold=0.5)
    # child suppressed (var > 0.5), parent kept
    assert float(adj[0, 1]) == 0.0
    assert float(adj[0, 0]) == pytest.approx(0.8)
    assert log["total_suppressed"] == 1


def test_fallback_keeps_certain_child():
    mean = torch.tensor([[0.8, 0.8]])
    var  = torch.tensor([[0.1, 0.2]])   # child certain (var <= 0.5)
    pairs = [(0, 1)]
    adj, log = apply_hierarchical_fallback(mean, var, pairs, gate_threshold=0.5)
    assert float(adj[0, 1]) == pytest.approx(0.8)
    assert log["total_suppressed"] == 0


def test_fallback_single_sample_squeezed():
    mean = torch.tensor([0.7, 0.9])
    var  = torch.tensor([0.1, 0.8])
    adj, _ = apply_hierarchical_fallback(mean, var, [(0, 1)], gate_threshold=0.5)
    assert adj.dim() == 1
    assert float(adj[1]) == 0.0


def test_fallback_batch_partial():
    # batch of 3: only the second sample has uncertain child
    mean = torch.tensor([[0.8, 0.8], [0.8, 0.8], [0.8, 0.8]])
    var  = torch.tensor([[0.2, 0.2], [0.2, 0.9], [0.2, 0.2]])
    adj, log = apply_hierarchical_fallback(mean, var, [(0, 1)], gate_threshold=0.5)
    assert float(adj[0, 1]) == pytest.approx(0.8)
    assert float(adj[1, 1]) == 0.0
    assert float(adj[2, 1]) == pytest.approx(0.8)
    assert log["total_suppressed"] == 1


# ---------------------------------------------------------------------------
# Vectorized CxrClsDataset
# ---------------------------------------------------------------------------

def _make_synth_csv(tmp_path, n: int = 8):
    """Write a minimal 55-column label CSV with controlled NaN entries."""
    from PIL import Image
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    rows = []
    for i in range(n):
        iid = f"img_{i:03d}"
        Image.fromarray(
            np.random.default_rng(i).integers(0, 255, (8, 8), dtype=np.uint8), "L"
        ).save(img_dir / f"{iid}.png")
        row = {"image_id": iid, "image_path": str(img_dir / f"{iid}.png"),
               "dataset": "synth", "split": "train" if i < 6 else "val"}
        for c in CANONICAL_LABELS:
            row[c] = float(i % 2) if c in ("Atelectasis", "Cardiomegaly") else float("nan")
        rows.append(row)
    csv = tmp_path / "synth.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv, img_dir


def test_dataset_nan_becomes_zero_mask_zero(tmp_path):
    from src.data.dataset import CxrClsDataset
    from torchvision import transforms
    csv, root = _make_synth_csv(tmp_path)
    tf = transforms.Compose([transforms.Grayscale(1), transforms.Resize((8, 8)), transforms.ToTensor()])
    ds = CxrClsDataset(csv, root, split="train", transform=tf,
                       conditions=CANONICAL_LABELS, skip_missing_check=True)
    sample = ds[0]
    labels     = sample["labels"]
    label_mask = sample["label_mask"]
    ai = CANONICAL_LABELS.index("Atelectasis")
    ti = CANONICAL_LABELS.index("Tuberculosis")
    # Atelectasis is annotated: mask=1, label=0 or 1
    assert float(label_mask[ai]) == 1.0
    # Tuberculosis is NaN: mask=0, label=0
    assert float(label_mask[ti]) == 0.0
    assert float(labels[ti]) == 0.0


def test_dataset_skip_missing_check(tmp_path):
    from src.data.dataset import CxrClsDataset
    from torchvision import transforms
    csv, root = _make_synth_csv(tmp_path)
    tf = transforms.Compose([transforms.Grayscale(1), transforms.Resize((8, 8)), transforms.ToTensor()])
    # skip_missing_check=True → no file-existence syscalls, all rows kept
    ds = CxrClsDataset(csv, root, split=None, transform=tf, skip_missing_check=True)
    assert len(ds) == 8
    assert ds.skipped_missing == 0


def test_dataset_split_filter(tmp_path):
    from src.data.dataset import CxrClsDataset
    from torchvision import transforms
    csv, root = _make_synth_csv(tmp_path, n=8)
    tf = transforms.Compose([transforms.Grayscale(1), transforms.Resize((8, 8)), transforms.ToTensor()])
    train_ds = CxrClsDataset(csv, root, split="train", transform=tf, skip_missing_check=True)
    val_ds   = CxrClsDataset(csv, root, split="val",   transform=tf, skip_missing_check=True)
    assert len(train_ds) == 6
    assert len(val_ds)   == 2


def test_dataset_label_tensor_shape(tmp_path):
    from src.data.dataset import CxrClsDataset
    from torchvision import transforms
    csv, root = _make_synth_csv(tmp_path)
    tf = transforms.Compose([transforms.Grayscale(1), transforms.Resize((8, 8)), transforms.ToTensor()])
    ds = CxrClsDataset(csv, root, split=None, transform=tf, skip_missing_check=True)
    sample = ds[0]
    assert sample["labels"].shape == (len(CANONICAL_LABELS),)
    assert sample["label_mask"].shape == (len(CANONICAL_LABELS),)


# ---------------------------------------------------------------------------
# AURC plug-in estimator
# ---------------------------------------------------------------------------

def _rng_arrays(n=100, c=5, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.random((n, c)).astype(np.float32)
    t = (rng.random((n, c)) > 0.5).astype(np.float32)
    m = np.ones((n, c), dtype=np.float32)
    return p, t, m


def test_aurc_flat_returns_scalar():
    p, t, m = _rng_arrays()
    v = aurc_flat(p, t, m)
    assert np.isfinite(v) and v >= 0


def test_aurc_flat_perfect_lower_than_random():
    rng = np.random.default_rng(1)
    n = 200
    true = (rng.random(n) > 0.5).astype(float)
    perfect_p = true * 0.95 + (1 - true) * 0.05
    random_p  = rng.random(n)
    mask = np.ones(n)
    perfect_p_2d = perfect_p[:, None]
    random_p_2d  = random_p[:, None]
    true_2d = true[:, None]
    mask_2d = mask[:, None]
    assert aurc_flat(perfect_p_2d, true_2d, mask_2d) < aurc_flat(random_p_2d, true_2d, mask_2d)


def test_compute_aurc_structure():
    p, t, m = _rng_arrays(n=50, c=3)
    r = compute_aurc(p, t, m, conditions=CANONICAL_LABELS[:3])
    assert "flat" in r and "per_class" in r and "macro" in r
    assert np.isfinite(r["flat"]) and np.isfinite(r["macro"])


def test_per_class_aurc_nan_for_tiny_class():
    p, t, m = _rng_arrays(n=50, c=3)
    m[:, 2] = 0   # class 2 has no applicable samples
    pc = per_class_aurc(p, t, m, conditions=CANONICAL_LABELS[:3])
    assert np.isnan(pc[CANONICAL_LABELS[2]])


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def test_ece_range():
    rng = np.random.default_rng(0)
    p = rng.random((100, 5)).astype(np.float32)
    t = (rng.random((100, 5)) > 0.5).astype(np.float32)
    m = np.ones((100, 5), np.float32)
    ece = expected_calibration_error(p, t, m, n_bins=10)
    assert 0.0 <= ece <= 1.0


def test_ece_near_zero_for_calibrated_predictions():
    # perfectly calibrated: predicted prob == empirical rate
    rng = np.random.default_rng(2)
    probs = np.tile(np.linspace(0.1, 0.9, 9), 20)[:, None]
    targets = (rng.random(len(probs)) < probs[:, 0]).astype(float)[:, None]
    mask = np.ones_like(targets)
    ece = expected_calibration_error(probs, targets, mask, n_bins=9)
    assert ece < 0.15   # not perfect but should be low


def test_compute_calibration_structure():
    p, t, m = _rng_arrays(n=80, c=4)
    r = compute_calibration(p, t, m, n_bins=10, conditions=CANONICAL_LABELS[:4])
    assert "ece" in r and "reliability" in r and "per_class_ece" in r
    assert 0.0 <= r["ece"] <= 1.0


# ---------------------------------------------------------------------------
# Statistics (bootstrap CI, compare_aurc, McNemar)
# ---------------------------------------------------------------------------

def test_bootstrap_ci_keys():
    p, t, m = _rng_arrays(n=100, c=5)
    ci = bootstrap_metric_ci(p, t, m, conditions=CANONICAL_LABELS[:5], n_boot=20)
    expected = {"auroc_macro", "f1_macro", "map_macro", "aurc_flat", "aurc_macro"}
    assert expected <= set(ci.keys())


def test_bootstrap_ci_bounds():
    p, t, m = _rng_arrays(n=100, c=5)
    ci = bootstrap_metric_ci(p, t, m, conditions=CANONICAL_LABELS[:5], n_boot=20)
    for name, d in ci.items():
        assert d["ci_lo"] <= d["ci_hi"], f"CI inverted for {name}"
        assert np.isfinite(d["estimate"]), f"estimate NaN for {name}"


def test_compare_aurc_structure():
    p, t, m = _rng_arrays(n=80, c=5)
    r = compare_aurc(p, t, m, p + 0.1, t, m, n_boot=20, label=("A", "B"))
    assert "aurc_a" in r and "aurc_b" in r
    assert "p_value_h1_a_lt_b" in r and "significant_at_0.05" in r
    assert 0.0 <= r["p_value_h1_a_lt_b"] <= 1.0


def test_compare_aurc_identical_p_near_half():
    p, t, m = _rng_arrays(n=80, c=5, seed=7)
    r = compare_aurc(p, t, m, p.copy(), t, m, n_boot=50, label=("A", "A"))
    # same predictions → delta ≈ 0, p-value should be near 0.5
    assert abs(r["delta_a_minus_b"]) < 1e-6
    assert r["p_value_h1_a_lt_b"] > 0.2


def test_mcnemar_identical_predictions():
    p, t, m = _rng_arrays(n=100, c=5)
    r = mcnemar_comparison(p, t, m, p.copy(), t, m, label=("A", "B"))
    # identical predictions → all discordant = 0 → p_value = 1.0
    assert r["n_discordant"] == 0
    assert r["p_value_two_tailed"] == 1.0
    assert not r["significant_at_0.05"]


def test_mcnemar_clearly_different():
    rng = np.random.default_rng(0)
    n = 200
    t = (rng.random((n, 1)) > 0.5).astype(float)
    m = np.ones((n, 1))
    # A: near-perfect; B: near-random
    pa = t * 0.95 + (1 - t) * 0.05
    pb = rng.random((n, 1)).astype(float)
    r = mcnemar_comparison(pa, t, m, pb, t, m, label=("A", "B"))
    # A should win on accuracy
    assert r["A_acc"] > r["B_acc"]


# ---------------------------------------------------------------------------
# Image-level leakage check
# ---------------------------------------------------------------------------

def test_check_image_leakage_detects_duplicate():
    df = pd.DataFrame({
        "image_id": ["a", "b", "c", "b"],
        "split":    ["train", "train", "val", "test"],
    })
    r = check_image_leakage(df)
    assert not r["passes"]
    assert r["leaked_ids"] == 1
    assert "b" in r["examples"]


def test_check_image_leakage_passes_clean():
    df = pd.DataFrame({
        "image_id": ["a", "b", "c", "d"],
        "split":    ["train", "train", "val", "test"],
    })
    r = check_image_leakage(df)
    assert r["passes"]
    assert r["leaked_ids"] == 0


def test_check_patient_leakage_detects_train_test_overlap():
    # NIH-style image_ids: "00000001_000" → patient "00000001"
    def _nih(iid): return iid.split("_")[0]

    df = pd.DataFrame({
        "image_id": ["00000001_000", "00000001_001", "00000002_000"],
        "split":    ["train",         "test",          "val"],
        "dataset":  ["nih-cxr14"] * 3,
    })
    r = check_patient_leakage(df, "nih-cxr14", _nih)
    assert not r["passes"]
    assert r["overlap_train_test"] == 1


def test_check_patient_leakage_passes_clean():
    def _nih(iid): return iid.split("_")[0]

    df = pd.DataFrame({
        "image_id": ["00000001_000", "00000002_000", "00000003_000"],
        "split":    ["train",         "val",           "test"],
        "dataset":  ["nih-cxr14"] * 3,
    })
    r = check_patient_leakage(df, "nih-cxr14", _nih)
    assert r["passes"]
    assert r["overlap_train_test"] == 0


# ---------------------------------------------------------------------------
# Hierarchy edge_index_pairs and validate_edges
# ---------------------------------------------------------------------------

def test_edge_index_pairs_correct():
    # Known edge: Pneumonia → COVID19_Pneumonia
    conds = CANONICAL_LABELS
    pairs = edge_index_pairs(conds)
    pi = conds.index("Pneumonia")
    ci = conds.index("COVID19_Pneumonia")
    assert (pi, ci) in pairs


def test_edge_index_pairs_skips_missing_labels():
    subset = ["Pneumonia", "Atelectasis"]   # COVID19_Pneumonia not present
    pairs = edge_index_pairs(subset)
    assert len(pairs) == 0  # Pneumonia edge dropped (child missing)


def test_validate_edges_warns_missing():
    warnings = validate_edges(["Pneumonia"])  # children of Pneumonia missing
    assert any("COVID19_Pneumonia" in w for w in warnings)


def test_validate_edges_clean():
    warnings = validate_edges(CANONICAL_LABELS)
    assert len(warnings) == 0  # all 51 labels present — no warnings
