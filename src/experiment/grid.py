"""The 4-condition ablation grid: 1 backbone × 2 label_structure × 2 uq.

Backbone is fixed: densenet121_xrv (TorchXRayVision DenseNet121, CXR-pretrained).
Segmentation input is fixed: off (no mask channel).

Factor 1 — label_structure: flat | hierarchical (HBCE loss)
Factor 2 — UQ:              none | mc_dropout

Condition configs are derived from a base config (shared data/training/paths)
by overriding the two factors. Names are fully descriptive so no legend is needed.
"""

from __future__ import annotations

import copy
import itertools
from pathlib import Path

SEG = [False]
LABEL_STRUCTURES = ["flat", "hierarchical"]
UQ = [False, True]

BASE_METRICS = ["auroc", "f1", "ece", "map", "aurc", "hcv", "clf"]


def enumerate_conditions() -> list[dict]:
    return [
        {"model": "densenet121_xrv", "seg": s, "label": l, "uq": u}
        for s, l, u in itertools.product(SEG, LABEL_STRUCTURES, UQ)
    ]


def grid_name(c: dict) -> str:
    seg = "seg-concat" if c["seg"] else "seg-off"
    uq = "uq-mcdropout" if c["uq"] else "uq-off"
    base = f"{c['model']}__{seg}__{uq}"
    return base if c["label"] == "flat" else f"{base}__hierarchical"


def build_condition_config(base, c: dict):
    cfg = copy.deepcopy(base)
    cfg.model.backbone = c["model"]
    cfg.seg.enabled = c["seg"]
    cfg.seg.method = "concat_channel"
    cfg.label.label_structure = c["label"]
    cfg.uq.method = "mc_dropout" if c["uq"] else "none"
    cfg.experiment.name = grid_name(c)
    cfg.experiment.notes = " | ".join([
        c["model"],
        "seg-concat" if c["seg"] else "seg-off",
        "mc-dropout" if c["uq"] else "uq-off",
        f"{c['label']} loss",
    ])
    cfg.eval.metrics = list(BASE_METRICS)
    return cfg


def build_grid(base) -> list:
    return [build_condition_config(base, c) for c in enumerate_conditions()]


def write_grid_configs(base, out_dir) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for cfg in build_grid(base):
        p = out_dir / f"{cfg.experiment.name}.yaml"
        cfg.to_yaml(p)
        paths.append(p)
    return paths
