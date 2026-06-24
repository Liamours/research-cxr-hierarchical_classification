"""Ingest MIMIC-CXR or CheXpert from the archive-preprocessed layout into the
canonical preprocessed layout used by the training pipeline.

Source layout (archive-preprocessed):
  <dataset-root>/archive-preprocessed/<split>/p<patient>/s<study>/view1_frontal.png
  <dataset-root>/label/metadata_train.csv     (image_paths + CheXpert label cols)
  <dataset-root>/label/metadata_valid.csv
  [MIMIC only] <dataset-root>/label/mimic-cxr-2.0.0-chexpert.csv

Output layout (canonical preprocessed):
  <out-root>/images/<dataset>/<image_id>.png   224x224x3 uint8
  <out-root>/labels/<dataset>.csv              55-column canonical CSV

Resume-safe: skips existing PNGs.

Usage:
    uv run python src/script/run_ingest_metadata.py ^
        --dataset mimic-cxr ^
        --dataset-root E:/research-cxr/dataset/mimic-cxr ^
        --out-root E:/research-cxr/dataset/mimic-cxr/preprocessed

    uv run python src/script/run_ingest_metadata.py ^
        --dataset chexpert ^
        --dataset-root E:/research-cxr/dataset/chexpert ^
        --out-root E:/research-cxr/dataset/chexpert/preprocessed
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.data.label_space import CANONICAL_LABELS
from src.data.preprocess.common import (
    CHEXPERT_LABEL_MAP,
    CSV_COLUMNS,
    map_uncertain_value,
    verify_label_df,
    write_label_csv,
)

APPLICABLE = set(CHEXPERT_LABEL_MAP.values())  # 8 canonical diseases w/ signal


def archive_png(dataset_root: Path, stale_path: str) -> Path:
    """Convert stale metadata image_path to actual archive-preprocessed path.

    Stale:  'preprocessed\\PNG\\train\\p10000032\\s50414267\\view1_frontal.png'
    Actual: dataset_root/archive-preprocessed/train/p10000032/s50414267/view1_frontal.png
    """
    p = stale_path.replace("\\", "/").replace("preprocessed/PNG/", "")
    return dataset_root / "archive-preprocessed" / p


def to_rgb_224(src: Path, image_size: int = 224) -> Image.Image:
    """Load archive PNG (already 224x224 grayscale) -> 224x224x3 RGB.
    Falls back to resize+crop for any non-standard size."""
    img = Image.open(src).convert("L")
    if img.size != (image_size, image_size):
        w, h = img.size
        scale = 256 / min(w, h)
        img = img.resize(
            (max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC
        )
        w, h = img.size
        left, top = (w - image_size) // 2, (h - image_size) // 2
        img = img.crop((left, top, left + image_size, top + image_size))
    return img.convert("RGB")


def _build_row(image_id: str, dataset: str, split: str, raw_labels: dict) -> dict:
    row: dict = {
        "image_id": image_id,
        "image_path": f"{dataset}/{image_id}.png",
        "dataset": dataset,
        "split": split,
    }
    for c in CANONICAL_LABELS:
        if c not in APPLICABLE:
            row[c] = float("nan")
        else:
            row[c] = map_uncertain_value(raw_labels.get(c, float("nan")))
    return row


def ingest_mimic(dataset_root: Path, out_root: Path, limit: int | None) -> None:
    label_dir = dataset_root / "label"

    meta = pd.concat(
        [
            pd.read_csv(label_dir / "metadata_train.csv"),
            pd.read_csv(label_dir / "metadata_valid.csv"),
        ],
        ignore_index=True,
    )
    meta["study_id_int"] = meta["study_id"].str.lstrip("s").astype(int)

    chex = pd.read_csv(label_dir / "mimic-cxr-2.0.0-chexpert.csv")
    # index by study_id; handle rare duplicates by keeping first
    chex = chex.drop_duplicates(subset="study_id").set_index("study_id")

    dataset = "mimic-cxr"
    img_dir = out_root / "images" / dataset
    img_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    skipped = missing = errors = 0

    for _, row in tqdm(meta.iterrows(), total=len(meta), desc=f"ingest {dataset}"):
        if str(row.get("has_image", "")).lower() != "true":
            missing += 1
            continue

        try:
            paths: list[str] = json.loads(row["image_paths"])
        except (json.JSONDecodeError, TypeError):
            missing += 1
            continue

        frontal = [p for p in paths if "frontal" in Path(p).stem.lower()]
        if not frontal:
            missing += 1
            continue

        src = archive_png(dataset_root, frontal[0])
        if not src.exists():
            missing += 1
            continue

        view_stem = Path(frontal[0]).stem  # e.g. view1_frontal
        image_id = f"{row['patient_id']}_{row['study_id']}_{view_stem}"
        dest = img_dir / f"{image_id}.png"

        if not dest.exists():
            try:
                to_rgb_224(src).save(dest)
            except Exception as e:
                tqdm.write(f"ERROR {src}: {e}")
                errors += 1
                continue
        else:
            skipped += 1

        study_id_int = row["study_id_int"]
        raw_labels: dict = {}
        if study_id_int in chex.index:
            chex_row = chex.loc[study_id_int]
            for chex_col, canon in CHEXPERT_LABEL_MAP.items():
                raw_labels[canon] = chex_row.get(chex_col, float("nan"))

        split = str(row["split"]).replace("valid", "val")
        rows.append(_build_row(image_id, dataset, split, raw_labels))

        if limit is not None and len(rows) >= limit:
            break

    _finish(rows, out_root, dataset, skipped, missing, errors)


def ingest_chexpert(dataset_root: Path, out_root: Path, limit: int | None) -> None:
    label_dir = dataset_root / "label"

    meta = pd.concat(
        [
            pd.read_csv(label_dir / "metadata_train.csv"),
            pd.read_csv(label_dir / "metadata_valid.csv"),
        ],
        ignore_index=True,
    )

    CHEX_COLS = list(CHEXPERT_LABEL_MAP.keys())
    dataset = "chexpert"
    img_dir = out_root / "images" / dataset
    img_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    skipped = missing = errors = 0

    for _, row in tqdm(meta.iterrows(), total=len(meta), desc=f"ingest {dataset}"):
        if str(row.get("has_image", "")).lower() != "true":
            missing += 1
            continue

        try:
            paths: list[str] = json.loads(row["image_paths"])
        except (json.JSONDecodeError, TypeError):
            missing += 1
            continue

        frontal = [p for p in paths if "frontal" in Path(p).stem.lower()]
        if not frontal:
            missing += 1
            continue

        src = archive_png(dataset_root, frontal[0])
        if not src.exists():
            missing += 1
            continue

        view_stem = Path(frontal[0]).stem
        image_id = f"{row['patient_id']}_{row['study_id']}_{view_stem}"
        dest = img_dir / f"{image_id}.png"

        if not dest.exists():
            try:
                to_rgb_224(src).save(dest)
            except Exception as e:
                tqdm.write(f"ERROR {src}: {e}")
                errors += 1
                continue
        else:
            skipped += 1

        raw_labels = {
            CHEXPERT_LABEL_MAP[col]: row.get(col, float("nan"))
            for col in CHEX_COLS
            if col in row.index
        }

        split = str(row["split"]).replace("valid", "val")
        rows.append(_build_row(image_id, dataset, split, raw_labels))

        if limit is not None and len(rows) >= limit:
            break

    _finish(rows, out_root, dataset, skipped, missing, errors)


def _finish(rows, out_root, dataset, skipped, missing, errors):
    out_csv = out_root / "labels" / f"{dataset}.csv"
    df = write_label_csv(rows, out_csv)
    verify_label_df(df, dataset)
    splits = df["split"].value_counts().to_dict() if len(df) else {}
    annotated = [c for c in CANONICAL_LABELS if c in APPLICABLE]
    print(f"\n{dataset}: {len(df)} rows -> {out_csv}")
    print(f"  splits: {splits}")
    print(f"  annotated diseases ({len(annotated)}): {annotated}")
    print(f"  skipped (existing): {skipped}  |  missing/no-image: {missing}  |  errors: {errors}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["mimic-cxr", "chexpert"])
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--limit", type=int, default=None, help="cap rows (smoke test)")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    out_root = Path(args.out_root)

    if args.dataset == "mimic-cxr":
        ingest_mimic(dataset_root, out_root, args.limit)
    else:
        ingest_chexpert(dataset_root, out_root, args.limit)


if __name__ == "__main__":
    main()
