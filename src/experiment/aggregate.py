"""Aggregate per-condition eval metrics into one comparison table."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def collect_summary(run_dir, split: str = "val") -> dict | None:
    p = Path(run_dir) / f"eval_metrics_{split}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8")).get("summary")


def aggregate(configs, split: str = "val") -> pd.DataFrame:
    rows = []
    for cfg in configs:
        row = {
            "name":            cfg.experiment.name,
            "backbone":        cfg.model.backbone,
            "label_structure": cfg.label.label_structure,
            "seg":             cfg.seg.enabled,
            "uq":              cfg.uq.method,
        }
        summary = collect_summary(cfg.run_dir(), split)
        if summary:
            row.update({
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in summary.items()
            })
        else:
            row["status"] = "missing"
        rows.append(row)
    df = pd.DataFrame(rows)
    priority = [
        "name", "backbone", "label_structure", "seg", "uq",
        "auroc_macro", "auroc_micro",
        "f1_macro", "f1_micro",
        "map_macro",
        "aurc_macro", "aurc_flat",
        "ece",
        "uq_mean_variance", "uq_sanity_passes", "uq_fallback_suppressed",
    ]
    ordered = [c for c in priority if c in df.columns]
    rest = [c for c in df.columns if c not in ordered]
    return df[ordered + rest]
