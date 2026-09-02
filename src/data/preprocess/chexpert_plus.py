"""CheXpert+ -> canonical preprocessing (secondary dataset).

Requires the Stanford CheXpert+ files on disk:
  train.csv  (Path, Frontal/Lateral, 14 CheXpert-labeler columns)
  <Path>     JPEG images referenced relative to raw_root

Without them this raises FileNotFoundError and the run is deferred (mock mode).
CheXpert+ ships no official val/test split, so a patient-level train/val split
is derived here (VAL_FRACTION, seed SPLIT_SEED) -- no patient appears in both
splits. Shares the image/label/CSV core via common.preprocess_records and the
CheXpert label map from common.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.label_map import load_equivalence
from src.data.preprocess import common

VAL_FRACTION = 0.1
SPLIT_SEED = 42


def _image_id(path: str) -> str:
    parts = Path(path).parts[-3:]  # patient / study / view_frontal.jpg
    return "_".join(parts).rsplit(".", 1)[0]


def _patient_id(path: str) -> str:
    return Path(path).parts[-3]


def _patient_splits(patient_ids, val_fraction: float, seed: int) -> dict[str, str]:
    """Assign each unique patient to train or val (no patient in both)."""
    unique = np.array(sorted(set(patient_ids)))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_val = max(1, round(len(unique) * val_fraction))
    val_ids = set(unique[:n_val])
    return {p: ("val" if p in val_ids else "train") for p in unique}


def preprocess_chexpert(raw_root, out_root, limit: int | None = None) -> Path:
    raw_root = Path(raw_root)
    label_csv = raw_root / "train.csv"
    if not label_csv.exists():
        raise FileNotFoundError(label_csv)

    df = pd.read_csv(label_csv)
    if "Frontal/Lateral" in df.columns:
        df = df[df["Frontal/Lateral"] == "Frontal"]

    split_of = _patient_splits(df["Path"].map(_patient_id), VAL_FRACTION, SPLIT_SEED)
    chex_map = load_equivalence().dataset_to_canonical("chexpert")  # raw column -> canonical
    applicable = set(chex_map.values())

    records = []
    for _, row in df.iterrows():
        rel = row["Path"]
        src = raw_root / rel
        if not src.exists():
            continue
        raw_labels = {canon: row.get(col) for col, canon in chex_map.items()}
        records.append({
            "image_id": _image_id(rel),
            "src_image": src,
            "split": split_of[_patient_id(rel)],
            "labels": raw_labels,
            "applicable": applicable,  # only mapped conditions; rest NaN (masked)
        })
        if limit is not None and len(records) >= limit:
            break

    rows = common.preprocess_records(records, out_root, "chexpert")
    out_csv = Path(out_root) / "labels" / "chexpert.csv"
    common.verify_label_df(common.write_label_csv(rows, out_csv), "chexpert")
    return out_csv
