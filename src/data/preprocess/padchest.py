"""PadChest -> canonical preprocessing.

Requires on disk:
  labels/PADCHEST.csv                                 (ImageID, PatientID, StudyID,
                                                       Projection, Labels [python-list string])
  preprocessing/patient_<PatientID>/study_<StudyID>/<ImageID>_<Projection>.png

Only frontal projections (PA / AP / AP_horizontal) are kept; laterals and
non-frontal views are dropped. The free-text `Labels` list is mapped to canonical
labels via the 'padchest' block in configs/label_equivalence.json; only those
canonical conditions apply (others written NaN). PadChest has no official split,
so a patient-level train/val/test split is carved (seed 42). Rows whose image
file is missing on disk are skipped.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.label_map import load_equivalence
from src.data.preprocess import common

FRONTAL = {"PA", "AP", "AP_horizontal"}
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1
SPLIT_SEED = 42


def _parse_labels(cell) -> set[str]:
    if not isinstance(cell, str):
        return set()
    try:
        return {str(x).strip().lower() for x in ast.literal_eval(cell)}
    except (ValueError, SyntaxError):
        return set()


def _assign_splits(patient_ids, val_frac, test_frac, seed):
    unique = sorted(set(patient_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n = len(unique)
    n_test = max(1, round(n * test_frac))
    n_val = max(1, round(n * val_frac))
    test = set(unique[:n_test])
    val = set(unique[n_test:n_test + n_val])
    return val, test


def preprocess_padchest(raw_root, out_root, images_dir=None, limit: int | None = None) -> Path:
    raw_root = Path(raw_root)
    label_csv = raw_root / "labels" / "PADCHEST.csv"
    if not label_csv.exists():
        raise FileNotFoundError(label_csv)
    images_dir = Path(images_dir) if images_dir else raw_root / "preprocessing"

    df = pd.read_csv(label_csv, low_memory=False)
    df = df[df["Projection"].isin(FRONTAL)]

    pc_map = load_equivalence().dataset_to_canonical("padchest")  # raw finding -> canonical
    applicable = set(pc_map.values())

    val_p, test_p = _assign_splits(df["PatientID"].astype(str), VAL_FRACTION, TEST_FRACTION, SPLIT_SEED)

    records = []
    for _, row in df.iterrows():
        image_id = row["ImageID"]
        proj = row["Projection"]
        fname = f"{image_id}_{proj}.png"
        src = images_dir / f"patient_{row['PatientID']}" / f"study_{row['StudyID']}" / fname
        if not src.exists():
            continue

        findings = _parse_labels(row["Labels"])
        positives = {pc_map[f] for f in findings if f in pc_map}
        labels = {c: (1.0 if c in positives else 0.0) for c in applicable}

        pid = str(row["PatientID"])
        split = "test" if pid in test_p else "val" if pid in val_p else "train"

        records.append({
            "image_id": Path(str(image_id)).stem,
            "src_image": src,
            "split": split,
            "labels": labels,
            "applicable": applicable,
        })
        if limit is not None and len(records) >= limit:
            break

    rows = common.preprocess_records(records, out_root, "padchest")
    out_csv = Path(out_root) / "labels" / "padchest.csv"
    common.verify_label_df(common.write_label_csv(rows, out_csv), "padchest")
    return out_csv
