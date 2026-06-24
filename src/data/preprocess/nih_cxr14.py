"""NIH ChestX-ray14 -> canonical preprocessing (external validation, flat only).

Requires on disk:
  Data_Entry_2017.csv   (Image Index, Finding Labels [pipe-separated], View Position)
  test_list.txt         (official test image list; rest -> train)
  <images_dir>/<name>   PNG images (flatten the 12 image archives into one dir)

Only the conditions mapped for nih-cxr14 in configs/label_equivalence.json apply
(the others are written NaN, not applicable). NIH has no official validation
split, so a patient-level validation split is carved from the non-test images
(VAL_FRACTION, SPLIT_SEED). Splits produced: train / val / test. Without the
files this raises FileNotFoundError and the run is deferred (mock mode).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.label_map import load_equivalence
from src.data.preprocess import common

# NIH ships only train_val_list.txt + test_list.txt (no official validation).
# We carve a patient-level validation split from the non-test images.
VAL_FRACTION = 0.1
SPLIT_SEED = 42


def _val_patients(patient_ids, frac: float, seed: int) -> set:
    unique = sorted(set(patient_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    return set(unique[: max(1, round(len(unique) * frac))])


def preprocess_nih(raw_root, out_root, images_dir=None, limit: int | None = None) -> Path:
    raw_root = Path(raw_root)
    label_csv = raw_root / "Data_Entry_2017.csv"
    if not label_csv.exists():
        raise FileNotFoundError(label_csv)
    images_dir = Path(images_dir) if images_dir else raw_root / "images"

    df = pd.read_csv(label_csv)
    if "View Position" in df.columns:
        df = df[df["View Position"] == "PA"]

    test_list = raw_root / "test_list.txt"
    test_ids = set(test_list.read_text().split()) if test_list.exists() else set()
    nih_map = load_equivalence().dataset_to_canonical("nih-cxr14")  # raw finding -> canonical
    applicable = set(nih_map.values())

    has_pid = "Patient ID" in df.columns
    def patient_of(row, name):
        return str(row["Patient ID"]) if has_pid else name.split("_")[0]

    nontest = df[~df["Image Index"].isin(test_ids)]
    nontest_pids = (nontest["Patient ID"].astype(str) if has_pid
                    else nontest["Image Index"].str.split("_").str[0])
    val_patients = _val_patients(nontest_pids, VAL_FRACTION, SPLIT_SEED)

    records = []
    for _, row in df.iterrows():
        name = row["Image Index"]
        src = images_dir / name
        if not src.exists():
            continue
        findings = str(row["Finding Labels"]).split("|")
        positives = {nih_map[f] for f in findings if f in nih_map}
        labels = {c: (1.0 if c in positives else 0.0) for c in applicable}
        if name in test_ids:
            split = "test"
        elif patient_of(row, name) in val_patients:
            split = "val"
        else:
            split = "train"
        records.append({
            "image_id": Path(name).stem,
            "src_image": src,
            "split": split,
            "labels": labels,
            "applicable": applicable,
        })
        if limit is not None and len(records) >= limit:
            break

    rows = common.preprocess_records(records, out_root, "nih-cxr14")
    out_csv = Path(out_root) / "labels" / "nih-cxr14.csv"
    common.verify_label_df(common.write_label_csv(rows, out_csv), "nih-cxr14")
    return out_csv
