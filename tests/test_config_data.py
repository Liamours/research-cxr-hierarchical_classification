"""Config + data + label-equivalence tests."""

from __future__ import annotations

import glob
import json

import numpy as np
import pandas as pd
import pytest

from src.config.experiment_config import ExperimentConfig
from src.data.label_map import LabelEquivalence, load_equivalence
from src.data.label_space import CANONICAL_LABELS
from src.data.preprocess import common


def test_config_round_trip(tmp_path):
    c = ExperimentConfig()
    c.experiment.name = "rt"
    c.training.epochs = 9
    c.to_yaml(tmp_path / "c.yaml")
    back = ExperimentConfig.from_yaml(tmp_path / "c.yaml")
    assert back.experiment.name == "rt" and back.training.epochs == 9


def test_config_override_and_reject(tmp_path):
    c = ExperimentConfig.from_yaml("configs/densenet121_xrv__flat.yaml",
                                   overrides={"training.epochs": 2})
    assert c.training.epochs == 2
    (tmp_path / "bad.yaml").write_text("training:\n  not_a_field: 1\n")
    with pytest.raises(ValueError):
        ExperimentConfig.from_yaml(tmp_path / "bad.yaml")


def test_all_committed_configs_parse():
    paths = glob.glob("configs/*.yaml") + glob.glob("configs/grid/*.yaml")
    assert len(paths) >= 3
    for p in paths:
        ExperimentConfig.from_yaml(p)


def test_label_csv_schema(synth):
    df = pd.read_csv(synth["csv"])
    assert list(df.columns) == ["image_id", "image_path", "dataset", "split"] + CANONICAL_LABELS
    assert df["image_id"].is_unique
    for c in CANONICAL_LABELS:
        assert set(df[c].dropna().unique()) <= {0.0, 1.0}


def test_split_leakage_detected():
    from src.eda.dataset_stats import split_leakage
    df = pd.DataFrame({"image_id": ["p1_s1", "p1_s2", "p2_s1"],
                       "split": ["train", "val", "train"]})
    rep = split_leakage(df, "chexpert")
    assert rep["n_leaked"] == 1 and rep["leaked"] == ["p1"]


def test_label_equivalence_resolves():
    eq = load_equivalence()
    assert eq.to_canonical("Effusion", "nih-cxr14") == "Pleural_Effusion"
    assert eq.to_canonical("effusion") == "Pleural_Effusion"
    assert eq.to_canonical("ptx") == "Pneumothorax"
    assert eq.to_canonical("nonsense xyz") is None
    cov = eq.coverage(["Atelectasis", "Effusion", "ZZZ"], "nih-cxr14")
    assert cov["unmapped"] == ["ZZZ"] and "Pleural_Effusion" in cov["canonical_covered"]


def test_label_equivalence_default_without_json():
    eq = LabelEquivalence.default()
    assert eq.to_canonical("Effusion", "nih-cxr14") == "Pleural_Effusion"


def test_equivalence_json_matches_code_maps():
    """Drift guard: dataset pairings in configs/label_equivalence.json must equal
    the code maps preprocessing actually uses (replaces the old generator)."""
    from src.data.label_space import NIH_LABEL_MAP, VINDR_LABEL_MAP
    from src.data.preprocess.common import CHEXPERT_LABEL_MAP
    eq = load_equivalence()

    assert eq.dataset_to_canonical("nih-cxr14") == NIH_LABEL_MAP
    assert eq.dataset_to_canonical("vindr-cxr") == VINDR_LABEL_MAP
    assert eq.dataset_to_canonical("chexpert") == CHEXPERT_LABEL_MAP


def test_preprocess_input_router_consistent(tmp_path):
    from PIL import Image
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 255, (60, 80, 3), np.uint8), "RGB").save(tmp_path / "a.jpg")
    Image.fromarray(rng.integers(0, 255, (120, 100), np.uint8), "L").save(tmp_path / "b.png")
    for f in ["a.jpg", "b.png"]:
        arr = np.array(common.process_path(tmp_path / f))
        assert arr.shape == (224, 224, 3) and arr.dtype == np.uint8
