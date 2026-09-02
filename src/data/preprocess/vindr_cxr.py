"""VinDr-CXR -> canonical preprocessing (generalization, flat only).

DICOM input; pydicom is imported lazily (not in base deps). Uses the PhysioNet
image-level labels; when multiple radiologist rows exist per image_id they are
majority-voted. Only the conditions mapped for vindr-cxr in
configs/label_equivalence.json apply; others are written NaN. VinDr ships train
and test sets but no validation, so a validation split is carved from train by
image_id. Splits produced: train / val / test (test only if the test file is
present).

Requires on disk:
  annotations/image_labels_train.csv  (and optionally image_labels_test.csv)
  train/<image_id>.dicom              (and test/<image_id>.dicom for the test set)

Without them this raises FileNotFoundError and the run is deferred (mock mode).
The DICOM read path is exercised only on real data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.label_map import load_equivalence
from src.data.preprocess import common


def _read_dicom(path) -> np.ndarray:
    import pydicom  # lazy: not a base dependency

    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)
    arr = arr * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))
    if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
        arr = arr.max() - arr
    return arr


# VinDr ships train (15,000) + test (3,000), no validation; scans are hash-named
# with no patient id, so a validation split is carved from train by image_id.
VAL_FRACTION = 0.1
SPLIT_SEED = 42


def preprocess_vindr(raw_root, out_root, limit: int | None = None) -> Path:
    raw_root = Path(raw_root)
    ann = raw_root / "annotations"
    train_csv = ann / "image_labels_train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(train_csv)

    vindr_map = load_equivalence().dataset_to_canonical("vindr-cxr")  # raw label -> canonical
    applicable = set(vindr_map.values())
    train_df = pd.read_csv(train_csv)

    ids = sorted(train_df["image_id"].unique())
    rng = np.random.default_rng(SPLIT_SEED)
    rng.shuffle(ids)
    val_ids = set(ids[: max(1, round(len(ids) * VAL_FRACTION))])

    records: list[dict] = []

    def add(frame, img_subdir, split_fn) -> bool:
        for image_id, group in frame.groupby("image_id"):
            src = raw_root / img_subdir / f"{image_id}.dicom"
            if not src.exists():
                continue
            labels: dict[str, float] = {}
            for s, canon in vindr_map.items():
                if s not in group.columns:
                    continue
                val = 1.0 if group[s].fillna(0).mean() >= 0.5 else 0.0
                labels[canon] = max(labels.get(canon, 0.0), val)
            records.append({
                "image_id": str(image_id),
                "src_image": _read_dicom(src),
                "split": split_fn(image_id),
                "labels": labels,
                "applicable": applicable,
            })
            if limit is not None and len(records) >= limit:
                return True
        return False

    stop = add(train_df, "train", lambda i: "val" if i in val_ids else "train")
    test_csv = ann / "image_labels_test.csv"
    if test_csv.exists() and not stop:
        add(pd.read_csv(test_csv), "test", lambda i: "test")

    rows = common.preprocess_records(records, out_root, "vindr-cxr", clip=True)
    out_csv = Path(out_root) / "labels" / "vindr-cxr.csv"
    common.verify_label_df(common.write_label_csv(rows, out_csv), "vindr-cxr")
    return out_csv
