"""Coverage for library modules previously exercised only via subprocess:
preprocess adapters, EDA, selective, inference predictor, grid + aggregate."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.label_space import CANONICAL_LABELS
from src.data.preprocess import common, chexpert_plus, nih_cxr14, mimic_cxr, vindr_cxr


def _img(p, mode, sz=(96, 96)):
    from PIL import Image
    p.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 255, sz + ((3,) if mode == "RGB" else ()), np.uint8), mode).save(p)


def test_preprocess_chexpert(tmp_path):
    chex = list(common.CHEXPERT_LABEL_MAP.keys())
    rows = []
    for i in range(10):
        for s in range(2):
            rel = f"train/p{i}/s{s}/v1_frontal.jpg"; _img(tmp_path / rel, "RGB")
            rows.append({"Path": rel, "Frontal/Lateral": "Frontal", **{c: 0.0 for c in chex}})
    pd.DataFrame(rows).to_csv(tmp_path / "train.csv", index=False)
    out = chexpert_plus.preprocess_chexpert(tmp_path, tmp_path / "pre")
    df = pd.read_csv(out)
    mapped = sorted(set(common.CHEXPERT_LABEL_MAP.values()))  # only mapped diseases filled
    assert len(df) == 20 and df[mapped].notna().all().all()

    # patient-level split: every patient's rows share one split, both splits
    # present, no patient appears in both (no leakage)
    df["patient"] = df["image_id"].str.split("_").str[0]
    assert (df.groupby("patient")["split"].nunique() == 1).all()
    assert set(df["split"]) == {"train", "val"}


def test_preprocess_nih(tmp_path):
    ent = []
    for i in range(3):
        nm = f"{i:08d}_000.png"; _img(tmp_path / "images" / nm, "L", (100, 100))
        ent.append({"Image Index": nm, "Finding Labels": "Effusion" if i == 0 else "No Finding",
                    "View Position": "PA"})
    pd.DataFrame(ent).to_csv(tmp_path / "Data_Entry_2017.csv", index=False)
    (tmp_path / "test_list.txt").write_text("")
    out = nih_cxr14.preprocess_nih(tmp_path, tmp_path / "pre")
    df = pd.read_csv(out)
    assert df[df.image_id == "00000000_000"].iloc[0]["Pleural_Effusion"] == 1.0
    assert math.isnan(df.iloc[0]["Tuberculosis"])  # not annotated by NIH


def test_preprocess_adapters_defer_without_data(tmp_path):
    for fn in (mimic_cxr.preprocess_mimic, vindr_cxr.preprocess_vindr):
        with pytest.raises(FileNotFoundError):
            fn(tmp_path / "nope", tmp_path / "o")


def test_eda_runs(make_cfg):
    from src.eda.dataset_stats import run_eda
    rep = run_eda(make_cfg(name="eda"), max_image_check=50)
    assert len(rep["class_distribution"]) == len(CANONICAL_LABELS)
    assert rep["label_sanity"]["ok"] and rep["integrity"]["bad_size"] == 0


def test_selective_curves():
    from src.evaluate.selective import coverage_accuracy
    rng = np.random.default_rng(0)
    p = rng.random((30, len(CANONICAL_LABELS))); t = (rng.random((30, len(CANONICAL_LABELS))) > 0.5).astype(float); m = np.ones((30, len(CANONICAL_LABELS)))
    cov = coverage_accuracy(p, t, m)
    assert cov["curve"][-1]["coverage"] == 1.0


def test_predictor_point_and_uncertainty(make_cfg, synth):
    from src.model.classifier import build_model_from_cfg
    from src.inference.predict import CxrPredictor
    cfg = make_cfg(name="pred")
    ck = cfg.run_dir() / "checkpoints"; ck.mkdir(parents=True, exist_ok=True)
    torch.save(build_model_from_cfg(cfg, pretrained=False).state_dict(), ck / "last.pt")
    pred = CxrPredictor(cfg, checkpoint=ck / "last.pt", device=torch.device("cpu"))
    ranked = pred.predict_image(synth["image"])
    assert len(ranked) == len(CANONICAL_LABELS) and all(0 <= v <= 1 for _, v in ranked)
    cfg.uq.mc_passes = 4
    triples = pred.predict_with_uncertainty(synth["image"], n_passes=4)
    assert len(triples) == len(CANONICAL_LABELS) and len(triples[0]) == 3


def test_grid_build_and_aggregate(make_cfg):
    from src.experiment.grid import build_grid, enumerate_conditions
    from src.experiment.aggregate import aggregate
    cfg = make_cfg(name="grid")
    conds = enumerate_conditions()
    assert len(conds) == 8
    configs = build_grid(cfg)
    df = aggregate(configs, "val")
    assert len(df) == 8  # none run yet -> all rows present (missing)
