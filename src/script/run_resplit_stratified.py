"""Re-split every ingested dataset with multi-label iterative stratification.

Stratifies at GROUP level (patient/study) where a group key is recoverable, so
multiple images of one patient never split across train/val/test (no leakage),
while each label's positives are spread ~proportionally across splits. Only the
`split` column of each preprocessed labels CSV is rewritten (images untouched);
a .bak is written first. Then regenerate combined.csv separately.

    uv run python src/script/run_resplit_stratified.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.data.label_space import CANONICAL_LABELS
from src.data.stratify import split_labels

SEED = 42
VAL, TEST = 0.1, 0.1
REGISTRY = Path(__file__).resolve().parents[2] / "configs" / "dataset_registry.json"


def _padchest_patient_map() -> dict:
    raw = pd.read_csv(Path("dataset/padchest/labels/PADCHEST.csv"),
                      usecols=["ImageID", "PatientID"], low_memory=False)
    raw["stem"] = raw["ImageID"].str.replace(".png", "", regex=False)
    return dict(zip(raw["stem"], raw["PatientID"].astype(str)))


def group_key(name: str, df: pd.DataFrame) -> pd.Series:
    """Recover the leakage-safe grouping unit per dataset."""
    ids = df["image_id"].astype(str)
    if name in ("nih-cxr14", "chexpert"):
        return ids.str.split("_").str[0]                    # patient prefix
    if name == "covidx-cxr4":
        return ids.str.split("_").str[:2].str.join("_")     # 'train_<N>' study/patient
    if name == "padchest":
        m = _padchest_patient_map()
        return ids.map(m).fillna(ids)                       # patient via raw join
    return ids                                              # per-row (~1 image/unit)


def resplit_dataset(name: str, csv: Path) -> dict:
    df = pd.read_csv(csv, low_memory=False)
    applicable = [c for c in CANONICAL_LABELS if c in df.columns and df[c].notna().any()]
    keys = group_key(name, df)

    # aggregate each group's label vector (positive if ANY image is positive)
    gdf = df[applicable].fillna(0).astype(int)
    gdf["_g"] = keys.values
    grouped = gdf.groupby("_g")[applicable].max()
    g_split = split_labels(grouped.values, seed=SEED, val=VAL, test=TEST)
    g_map = dict(zip(grouped.index, g_split))

    bak = Path(str(csv) + ".bak")
    if not bak.exists():                     # preserve the ORIGINAL (official) split
        shutil.copy(csv, bak)
    df["split"] = keys.map(g_map).values
    df.to_csv(csv, index=False)

    vc = df["split"].value_counts()
    return {
        "counts": {s: int(vc.get(s, 0)) for s in ("train", "val", "test")},
        "groups": int(grouped.shape[0]),
        "labels": len(applicable),
        "unit": "group" if keys.nunique() < len(df) else "row",
    }


def main():
    import json
    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))["datasets"]
    for name, cfg in reg.items():
        if cfg.get("status") != "ingested":
            continue
        csv = Path(cfg["root"]) / cfg["preprocessed_label_csv"]
        if not csv.exists():
            print(f"{name:12s} SKIP (no csv)")
            continue
        r = resplit_dataset(name, csv)
        print(f"{name:12s} unit={r['unit']:5s} groups={r['groups']:7d} labels={r['labels']:2d}  {r['counts']}")


if __name__ == "__main__":
    main()
