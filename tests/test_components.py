"""Component tests for under-covered modules: segmentation, backbones,
mc_dropout, and the preprocess CLI."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.config.experiment_config import SegCfg
from src.data import segmentation as S
from src.data.label_space import CANONICAL_LABELS
from src.model import backbones as B
from src.model.classifier import CxrClassifier
from src.model.mc_dropout import mc_dropout_predict, uncertainty_sanity_check


def test_segmentation_providers_and_conditioning():
    mask = S.SyntheticMaskProvider(224).get("x")
    assert tuple(mask.shape) == (1, 224, 224)
    img = torch.rand(3, 224, 224)
    assert S.apply_mask_conditioning(img, mask, "concat_channel").shape[0] == 4
    assert S.apply_mask_conditioning(img, mask, "crop").shape[0] == 3
    assert S.seg_extra_channels("concat_channel") == 1 and S.seg_extra_channels("crop") == 0
    with pytest.raises(FileNotFoundError):
        S.CheXmaskUProvider("does/not/exist", 224)


def test_apply_seg_to_tensor():
    on = SegCfg(enabled=True, method="concat_channel", mask_source="synthetic")
    assert S.apply_seg_to_tensor(torch.rand(3, 224, 224), on, 224, "id").shape[0] == 4
    off = SegCfg(enabled=False)
    assert S.apply_seg_to_tensor(torch.rand(3, 224, 224), off, 224, "id").shape[0] == 3


def test_backbones_build_and_conv_adapt():
    for name, dim, nk in [("densenet121_xrv", 1024, "xrv")]:
        _, d, k = B.build_backbone(name, pretrained=False)
        assert d == dim and k == nk
    conv = nn.Conv2d(3, 8, 3)
    new = B.expand_conv_in_channels(conv, 4)
    assert new.in_channels == 4 and torch.allclose(new.weight[:, :3], conv.weight)
    assert B.expand_conv_in_channels(conv, 3) is conv  # no-op when equal


def test_mc_dropout_predict_mean_var():
    flat = CxrClassifier("densenet121_xrv", pretrained=False)
    mean, epistemic, aleatoric = mc_dropout_predict(flat, torch.randn(2, 1, 224, 224), 4)
    shape = (2, len(CANONICAL_LABELS))
    assert tuple(mean.shape) == shape and tuple(epistemic.shape) == shape and tuple(aleatoric.shape) == shape
    assert (epistemic >= 0).all() and (aleatoric >= 0).all()


def test_uncertainty_helpers():
    mean, var = torch.rand(4, len(CANONICAL_LABELS)), torch.rand(4, len(CANONICAL_LABELS))
    t, m = (torch.rand(4, len(CANONICAL_LABELS)) > 0.5).float(), torch.ones(4, len(CANONICAL_LABELS))
    sc = uncertainty_sanity_check(mean, var, t, m)
    assert set(sc) == {"unc_correct", "unc_wrong", "passes"}


def test_run_preprocess_cli(tmp_path):
    import pandas as pd
    from PIL import Image
    raw = tmp_path / "nih"; (raw / "images").mkdir(parents=True)
    rng = np.random.default_rng(0)
    ent = []
    for i in range(3):
        nm = f"{i:08d}_000.png"
        Image.fromarray(rng.integers(0, 255, (100, 100), np.uint8), "L").save(raw / "images" / nm)
        ent.append({"Image Index": nm, "Finding Labels": "Effusion" if i == 0 else "No Finding",
                    "View Position": "PA"})
    pd.DataFrame(ent).to_csv(raw / "Data_Entry_2017.csv", index=False)
    (raw / "test_list.txt").write_text("")
    r = subprocess.run([sys.executable, "src/script/run_preprocess.py", "--dataset", "nih-cxr14",
                        "--raw-root", str(raw), "--out-root", str(tmp_path / "pre")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-300:]
    assert (tmp_path / "pre" / "labels" / "nih-cxr14.csv").exists()
