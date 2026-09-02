"""Shared pytest fixtures: a tiny synthetic preprocessed dataset and a config
factory. Everything is built in a temp dir so tests never touch real data or the
repo's result/ folder."""

from __future__ import annotations

import uuid

import numpy as np
import pytest

from src.config.experiment_config import ExperimentConfig
from src.data.label_space import CANONICAL_LABELS
from src.data.preprocess import common


@pytest.fixture(scope="session")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth")
    out = root / "pre"
    rng = np.random.default_rng(42)
    recs = []
    for i in range(12):
        recs.append({
            "image_id": f"p{i}_s1_d0",                     # distinct patient per image -> no leakage
            "src_image": rng.integers(20, 230, (80, 80), np.uint8),
            "split": "train" if i < 8 else "val",
            "labels": {**{c: 0.0 for c in CANONICAL_LABELS}, "Pulmonary_Edema": float(i % 2),
                       "Cardiomegaly": float((i + 1) % 2)},
            "applicable": None,
        })
    common.write_label_csv(common.preprocess_records(recs, out, "synth"), out / "labels" / "synth.csv")
    first = f"{out / 'images' / 'synth' / 'p0_s1_d0.png'}"
    return {"csv": out / "labels" / "synth.csv", "images": out / "images",
            "root": root, "image": first}


@pytest.fixture
def make_cfg(synth):
    def _make(**kw):
        c = ExperimentConfig.from_yaml("configs/densenet121_xrv__flat__seed42.yaml")
        c.experiment.name = kw.pop("name", f"t_{uuid.uuid4().hex[:8]}")
        c.model.pretrained = False
        c.data.label_csv = synth["csv"]
        c.data.image_root = synth["images"]
        c.data.batch_size = 2
        c.data.num_workers = 0
        c.training.epochs = 1
        c.training.bf16 = False
        c.training.warmup_steps = 1
        c.paths.result_root = synth["root"] / "runs"
        c.logging.val_display_rows = 1
        for k, v in kw.items():
            sec, key = k.split(".")
            setattr(getattr(c, sec), key, v)
        return c
    return _make
