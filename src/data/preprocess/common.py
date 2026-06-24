"""Shared preprocessing core used by every dataset preprocessor.

Produces the canonical on-disk layout from the standardized protocol:
  out_root/images/<dataset>/<image_id>.png   224x224x3 uint8
  out_root/labels/<dataset>.csv              4 id columns + one per canonical label

The dataset-specific modules (mimic_cxr.py, etc.) only build a list of simple
records and delegate the image + label + CSV work here, so the testable core
is shared and identical across datasets.

Record schema (one per output image):
  image_id   str, unique within the dataset
  src_image  path | PIL.Image | np.ndarray (raw pixels)
  split      "train" | "val" | "test"
  labels     dict canonical_name -> raw label value (1/0/-1/NaN)
  applicable set[str] of canonical names that apply, or None meaning all
             canonical labels
             (non-applicable conditions are written as NaN, never 0)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.data.label_space import CANONICAL_LABELS

CSV_COLUMNS = ["image_id", "image_path", "dataset", "split"] + CANONICAL_LABELS

# CheXpert NLP-labeler column -> canonical disease (lossy; see label_space). Only
# defensible finding->disease correspondences are kept. Shared by MIMIC-CXR-JPG and
# CheXpert+. Dropped (no clear clinical-diagnosis home): Consolidation, Enlarged
# Cardiomediastinum, Lung Opacity, Pleural Other, Support Devices, No Finding.
CHEXPERT_LABEL_MAP = {
    "Atelectasis": "Atelectasis",
    "Cardiomegaly": "Cardiomegaly",
    "Edema": "Pulmonary_Edema",
    "Fracture": "Chest_Trauma",
    "Lung Lesion": "Solitary_Pulmonary_Nodule",
    "Pleural Effusion": "Pleural_Effusion",
    "Pneumonia": "Pneumonia",
    "Pneumothorax": "Pneumothorax",
}


def to_uint8_clipped(arr: np.ndarray, low: float = 0.5, high: float = 99.5) -> np.ndarray:
    """Percentile-clip raw pixel intensities (DICOM-derived) to uint8 0-255."""
    a = arr.astype(np.float32)
    lo, hi = np.percentile(a, [low, high])
    a = np.clip(a, lo, hi)
    if hi > lo:
        a = (a - lo) / (hi - lo) * 255.0
    return a.astype(np.uint8)


def process_image(src, image_size: int = 224, clip: bool = False) -> Image.Image:
    """Load/convert a raw image to a 224x224x3 uint8 PIL image.

    Steps: grayscale, resize shortest side to 256 (bicubic), center-crop to
    image_size, replicate to 3 channels.
    """
    if isinstance(src, (str, Path)):
        img = Image.open(src)
    elif isinstance(src, np.ndarray):
        arr = to_uint8_clipped(src) if clip else src.astype(np.uint8)
        img = Image.fromarray(arr)
    else:
        img = src
    img = img.convert("L")

    w, h = img.size
    scale = 256 / min(w, h)
    img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)

    w, h = img.size
    left = (w - image_size) // 2
    top = (h - image_size) // 2
    img = img.crop((left, top, left + image_size, top + image_size))
    return img.convert("RGB")


def load_raw_image(path):
    """Flexible input router by file extension. DICOM (.dcm/.dicom) -> numpy
    array via pydicom (rescale slope/intercept, MONOCHROME1 inversion); JPEG/PNG
    -> PIL image. Returns whatever process_image accepts."""
    p = Path(path)
    if p.suffix.lower() in (".dcm", ".dicom"):
        import pydicom
        ds = pydicom.dcmread(str(p))
        arr = ds.pixel_array.astype(np.float32)
        arr = arr * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))
        if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
            arr = arr.max() - arr
        return arr
    return Image.open(p)


def process_path(path, image_size: int = 224) -> Image.Image:
    """Single flexible entry: any of DICOM / JPEG / PNG -> identical 224x224x3
    uint8 PIL output. DICOM gets percentile clipping; JPEG/PNG do not."""
    p = Path(path)
    is_dcm = p.suffix.lower() in (".dcm", ".dicom")
    return process_image(load_raw_image(p), image_size=image_size, clip=is_dcm)


def map_uncertain_value(v) -> float:
    """MIMIC/CheXpert mapping: -1 (uncertain) -> 0, NaN (not mentioned) -> 0,
    1 -> 1, 0 -> 0 (U-Zero policy)."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return 0.0
    v = float(v)
    if v == 1.0:
        return 1.0
    return 0.0


def preprocess_records(records, out_root, dataset: str, image_size: int = 224,
                       clip: bool = False) -> list[dict]:
    out_root = Path(out_root)
    img_dir = out_root / "images" / dataset
    img_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for rec in tqdm(records, desc=f"preprocess {dataset}", dynamic_ncols=True):
        img = process_image(rec["src_image"], image_size=image_size, clip=clip)
        rel = f"{dataset}/{rec['image_id']}.png"
        img.save(out_root / "images" / rel)

        row = {
            "image_id": rec["image_id"],
            "image_path": rel,
            "dataset": dataset,
            "split": rec["split"],
        }
        applicable = rec.get("applicable")
        raw = rec.get("labels", {})
        for c in CANONICAL_LABELS:
            if applicable is not None and c not in applicable:
                row[c] = float("nan")
            else:
                row[c] = map_uncertain_value(raw.get(c, float("nan")))
        rows.append(row)
    return rows


def write_label_csv(rows, out_path) -> pd.DataFrame:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    df.to_csv(out_path, index=False)
    return df


def verify_label_df(df: pd.DataFrame, dataset: str) -> None:
    missing = [c for c in CSV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{dataset}: label CSV missing columns {missing}")
    if df["image_id"].duplicated().any():
        raise ValueError(f"{dataset}: duplicate image_id present")
    for c in CANONICAL_LABELS:
        bad = [v for v in df[c].dropna().unique() if v not in (0.0, 1.0)]
        if bad:
            raise ValueError(f"{dataset}: column {c} has non-binary values {bad}")
