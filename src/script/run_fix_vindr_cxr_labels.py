"""Fix VinDr-CXR label CSV: re-derive from raw/train.csv with majority vote (>=2/3 rads).

Problem: original ingest used vinbig_14class_manifest.csv (OR logic from previous project).
Fix: re-derive from raw bounding box annotations with proper majority vote aggregation.
Images already on disk — only label CSV regenerated.

Usage:
    uv run python src/script/run_fix_vindr_cxr_labels.py `
        --dataset-root E:/research-cxr/dataset/vindr-cxr
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.data.label_space import CANONICAL_LABELS
from src.data.preprocess.common import CSV_COLUMNS, write_label_csv, verify_label_df

# raw class_name -> canonical (from label_equivalence.json vindr-cxr entries)
# many-to-one: ILD + Pulmonary fibrosis -> ILD (OR after majority vote)
RAW_TO_CANONICAL: dict[str, str] = {
    "Atelectasis":       "Atelectasis",
    "Cardiomegaly":      "Cardiomegaly",
    "ILD":               "ILD",
    "Pulmonary fibrosis": "ILD",          # merged into ILD
    "Nodule/Mass":       "Solitary_Pulmonary_Nodule",
    "Pleural effusion":  "Pleural_Effusion",
    "Pneumothorax":      "Pneumothorax",
}
APPLICABLE = set(RAW_TO_CANONICAL.values())  # 6 canonical labels
VAL_FRACTION = 0.1
SEED = 42


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    args = ap.parse_args()

    root = Path(args.dataset_root)
    raw_csv = root / "raw" / "train.csv"
    img_dir = root / "preprocessed" / "images" / "vindr-cxr"
    out_csv = root / "preprocessed" / "labels" / "vindr-cxr.csv"

    print("Reading raw annotations...")
    raw = pd.read_csv(raw_csv)
    print(f"  {len(raw):,} rows, {raw['image_id'].nunique():,} images, "
          f"{raw['rad_id'].nunique()} radiologists")

    # majority vote per (image_id, class_name): positive if >=2 rads annotated it
    mapped = raw[raw["class_name"].isin(RAW_TO_CANONICAL)].copy()
    rad_counts = (
        mapped.groupby(["image_id", "class_name"])["rad_id"]
        .nunique()
        .reset_index(name="n_rads")
    )
    rad_counts["positive"] = (rad_counts["n_rads"] >= 2).astype(float)

    # pivot to wide: image_id -> {class_name: 0/1}
    wide = rad_counts.pivot_table(
        index="image_id", columns="class_name", values="positive", fill_value=0.0
    ).reset_index()

    # all 15k image_ids
    all_ids = sorted(raw["image_id"].unique())

    # train/val split (seed 42, 10% val)
    rng = np.random.default_rng(SEED)
    ids_shuffled = list(all_ids)
    rng.shuffle(ids_shuffled)
    n_val = max(1, round(len(ids_shuffled) * VAL_FRACTION))
    val_set = set(ids_shuffled[:n_val])

    rows = []
    skipped = 0
    for image_id in all_ids:
        png = img_dir / f"{image_id}.png"
        if not png.exists():
            skipped += 1
            continue

        split = "val" if image_id in val_set else "train"
        row: dict = {
            "image_id": image_id,
            "image_path": f"vindr-cxr/{image_id}.png",
            "dataset": "vindr-cxr",
            "split": split,
        }

        # get majority-vote labels for this image
        img_row = wide[wide["image_id"] == image_id]

        for canon in CANONICAL_LABELS:
            if canon not in APPLICABLE:
                row[canon] = float("nan")
                continue
            # find raw columns that map to this canonical
            raw_cols = [r for r, c in RAW_TO_CANONICAL.items() if c == canon]
            # OR across mapped raw classes (e.g. ILD OR Pulmonary fibrosis)
            val = 0.0
            for rc in raw_cols:
                if not img_row.empty and rc in img_row.columns:
                    val = max(val, float(img_row[rc].iloc[0]))
            row[canon] = val

        rows.append(row)

    if skipped:
        print(f"  Skipped {skipped} images not on disk")

    df = write_label_csv(rows, out_csv)
    verify_label_df(df, "vindr-cxr")

    print(f"\nWrote {len(df):,} rows -> {out_csv}")
    print(f"Splits: {df['split'].value_counts().to_dict()}")
    print("\nLabel counts (majority vote):")
    for c in APPLICABLE:
        print(f"  {c:<35} {int(df[c].sum()):>6,}")


if __name__ == "__main__":
    main()
