"""Ingest VinDr-PCXR into canonical preprocessed layout.

Label mapping (VINDR_PCXR_LABEL_MAP, OR logic for many-to-one):
  Bronchitis           -> Acute_Bronchitis
  Brocho-pneumonia     -> Pneumonia
  Bronchiolitis        -> Bronchiolitis
  Pneumonia            -> Pneumonia
  Pleuro-pneumonia     -> Pneumonia
  Diagphramatic hernia -> Diaphragmatic_Hernia
  Tuberculosis         -> Tuberculosis
  Mediastinal tumor    -> Mediastinal_Tumor
  Lung tumor           -> Lung_Cancer

Skipped (no canonical mapping):
  No finding, Other disease, Situs inversus,
  Congenital emphysema, CPAM, Hyaline membrane disease

Split: test/ -> test (official). train/ -> 90/10 random seed=42 -> train/val.

Usage:
    uv run python src/script/run_ingest_vindr_pcxr.py \\
        --dataset-root C:/rifqi/research-cxr-hierarchical_classification/dataset/vindr-pcxr \\
        --out-root C:/rifqi/research-cxr-hierarchical_classification/dataset/vindr-pcxr/preprocessed
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from tqdm import tqdm

from src.data.label_space import CANONICAL_LABELS, VINDR_PCXR_LABEL_MAP
from src.data.preprocess.common import (
    process_path,
    verify_label_df,
    write_label_csv,
)

DATASET = "vindr-pcxr"
APPLICABLE = set(VINDR_PCXR_LABEL_MAP.values())
VAL_FRACTION = 0.1
RANDOM_SEED = 42


def _load_labels(label_dir: Path) -> dict[str, dict[str, float]]:
    """Returns {image_id: {canonical_label: 0.0|1.0}} using OR logic."""
    result: dict[str, dict[str, float]] = {}
    for split_name, fname in [("train", "image_labels_train.csv"), ("test", "image_labels_test.csv")]:
        fpath = label_dir / fname
        df = pd.read_csv(fpath)
        for _, row in df.iterrows():
            iid = row["image_id"]
            canonical: dict[str, float] = {c: 0.0 for c in APPLICABLE}
            for raw_col, canon in VINDR_PCXR_LABEL_MAP.items():
                if raw_col in row and float(row[raw_col]) == 1.0:
                    canonical[canon] = 1.0
            result[iid] = canonical
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    out_root = Path(args.out_root)
    label_dir = dataset_root / "archive-labels"

    label_map = _load_labels(label_dir)
    print(f"Labels loaded: {len(label_map)} images")

    # build split assignment
    test_ids = {p.stem for p in (dataset_root / "raw" / "test").glob("*.dicom")}
    train_ids = sorted({p.stem for p in (dataset_root / "raw" / "train").glob("*.dicom")})
    random.seed(RANDOM_SEED)
    random.shuffle(train_ids)
    n_val = round(len(train_ids) * VAL_FRACTION)
    val_ids = set(train_ids[:n_val])
    train_ids_set = set(train_ids[n_val:])

    split_map: dict[str, str] = {}
    for iid in train_ids_set:
        split_map[iid] = "train"
    for iid in val_ids:
        split_map[iid] = "val"
    for iid in test_ids:
        split_map[iid] = "test"

    print(f"Split: train={len(train_ids_set)}  val={len(val_ids)}  test={len(test_ids)}")

    img_dir = out_root / "images" / DATASET
    img_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    skipped = missing = errors = 0

    all_ids = sorted(split_map.keys())
    for image_id in tqdm(all_ids, desc=f"ingest {DATASET}"):
        split = split_map[image_id]
        raw_subdir = "test" if split == "test" else "train"
        src = dataset_root / "raw" / raw_subdir / f"{image_id}.dicom"

        if not src.exists():
            missing += 1
            continue

        dest = img_dir / f"{image_id}.png"
        if not dest.exists():
            try:
                process_path(src).save(dest)
            except Exception as e:
                tqdm.write(f"ERROR {src}: {e}")
                errors += 1
                continue
        else:
            skipped += 1

        canon_vals = label_map.get(image_id, {})
        row: dict = {
            "image_id": image_id,
            "image_path": f"{DATASET}/{image_id}.png",
            "dataset": DATASET,
            "split": split,
        }
        for c in CANONICAL_LABELS:
            if c in APPLICABLE:
                row[c] = canon_vals.get(c, 0.0)
            else:
                row[c] = float("nan")
        rows.append(row)

        if args.limit and len(rows) >= args.limit:
            break

    out_csv = out_root / "labels" / f"{DATASET}.csv"
    df = write_label_csv(rows, out_csv)
    verify_label_df(df, DATASET)
    splits = df["split"].value_counts().to_dict()
    print(f"\n{DATASET}: {len(df)} rows -> {out_csv}")
    print(f"  splits: {splits}")
    print(f"  annotated: {sorted(APPLICABLE)}")
    print(f"  skipped (existing): {skipped}  missing: {missing}  errors: {errors}")


if __name__ == "__main__":
    main()
