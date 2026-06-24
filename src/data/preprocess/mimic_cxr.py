"""MIMIC-CXR-JPG -> canonical preprocessing (primary dataset).

Implements the standardized protocol steps for MIMIC-CXR-JPG. Requires the
PhysioNet files on disk:
  mimic-cxr-2.0.0-metadata.csv.gz   (ViewPosition, dicom_id, study_id, subject_id)
  mimic-cxr-2.0.0-split.csv.gz      (official train/validate/test)
  mimic-cxr-2.0.0-chexpert.csv.gz   (14 CheXpert labels)
  files/p<XX>/p<subject>/s<study>/<dicom_id>.jpg

Without these it raises FileNotFoundError and the run is deferred (mock mode).
The image + label + CSV work is shared via common.preprocess_records.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.label_map import load_equivalence
from src.data.preprocess import common

# Frontal-view priority: PA preferred over AP; laterals (rank >= 9) dropped.
_FRONTAL_PRIORITY = {"PA": 0, "AP": 1, "LL": 9, "LATERAL": 9, "LAO": 9, "RAO": 9}

# Sourced from configs/label_equivalence.json (not hardcoded); kept in sync with
# common.CHEXPERT_LABEL_MAP by the drift test.
_CHEXPERT_TO_CANONICAL = load_equivalence().dataset_to_canonical("mimic-cxr")


def _view_rank(view: str) -> int:
    return _FRONTAL_PRIORITY.get(str(view).upper(), 5)


def preprocess_mimic(raw_root, out_root, limit: int | None = None) -> Path:
    raw_root = Path(raw_root)
    split_csv = raw_root / "mimic-cxr-2.0.0-split.csv.gz"
    label_csv = raw_root / "mimic-cxr-2.0.0-chexpert.csv.gz"
    meta_csv = raw_root / "mimic-cxr-2.0.0-metadata.csv.gz"
    for f in (meta_csv, split_csv, label_csv):
        if not f.exists():
            raise FileNotFoundError(f)

    meta = pd.read_csv(meta_csv)
    splits = pd.read_csv(split_csv)
    labels = pd.read_csv(label_csv)
    df = meta.merge(splits, on=["dicom_id", "study_id", "subject_id"]).merge(
        labels, on=["subject_id", "study_id"]
    )

    records: list[dict] = []
    for (subj, study), group in df.groupby(["subject_id", "study_id"]):
        best = min(group.to_dict("records"), key=lambda r: _view_rank(r.get("ViewPosition")))
        if _view_rank(best.get("ViewPosition")) >= 9:
            continue  # study has no frontal view
        dicom_id = best["dicom_id"]
        image_id = f"p{subj}_s{study}_{dicom_id}"
        rel = Path("files") / f"p{str(subj)[:2]}" / f"p{subj}" / f"s{study}" / f"{dicom_id}.jpg"
        src = raw_root / rel
        if not src.exists():
            continue
        split = str(best["split"]).replace("validate", "val")
        raw_labels = {canon: best.get(col) for col, canon in _CHEXPERT_TO_CANONICAL.items()}
        records.append({
            "image_id": image_id,
            "src_image": src,
            "split": split,
            "labels": raw_labels,
            "applicable": None,  # all 14 conditions apply to MIMIC
        })
        if limit is not None and len(records) >= limit:
            break

    rows = common.preprocess_records(records, out_root, "mimic-cxr")
    out_csv = Path(out_root) / "labels" / "mimic-cxr.csv"
    df_out = common.write_label_csv(rows, out_csv)
    common.verify_label_df(df_out, "mimic-cxr")
    return out_csv
