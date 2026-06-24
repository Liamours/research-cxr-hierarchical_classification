"""Merge all ingested per-dataset label CSVs into one combined CSV for training.

Reads configs/dataset_registry.json to find every dataset with status=ingested.
For each dataset:
  - Loads its preprocessed_label_csv
  - Rewrites image_path to absolute (so CxrClsDataset works with a single combined CSV
    regardless of each dataset living in its own root directory)
  - Filters out rows whose image file does not exist on disk

Output: a single combined.csv with the same 55-column schema, ready for training.

Usage:
    uv run python src/script/run_combine_datasets.py `
        --out E:/research-cxr/dataset/combined/combined.csv

    # exclude a dataset (e.g. vindr-pcxr is pediatric)
    uv run python src/script/run_combine_datasets.py `
        --out E:/research-cxr/dataset/combined/combined.csv `
        --exclude vindr-pcxr

    # dry run: print stats only, don't write
    uv run python src/script/run_combine_datasets.py `
        --out E:/research-cxr/dataset/combined/combined.csv --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from tqdm import tqdm

from src.data.preprocess.common import CSV_COLUMNS
from src.data.label_space import CANONICAL_LABELS

REGISTRY = Path(__file__).resolve().parents[2] / "configs" / "dataset_registry.json"


def load_and_absolutize(name: str, cfg: dict) -> pd.DataFrame:
    root = Path(cfg["root"])
    csv_path = root / cfg["preprocessed_label_csv"]
    img_base = root / cfg["preprocessed_image_dir"].rsplit("/", 1)[0]  # preprocessed/images/

    if not csv_path.exists():
        print(f"  SKIP {name}: CSV not found at {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)

    # rewrite image_path to absolute
    def to_abs(rel: str) -> str:
        p = Path(rel.replace("\\", "/"))
        if p.is_absolute():
            return str(p)
        return str(img_base / p)

    df["image_path"] = df["image_path"].map(to_abs)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="path to write combined.csv")
    ap.add_argument("--exclude", nargs="*", default=[], help="dataset names to skip")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-exist-check", action="store_true",
                    help="skip per-row image file existence check (faster on HDD)")
    args = ap.parse_args()

    registry = json.loads(REGISTRY.read_text())
    datasets = registry["datasets"]

    frames: list[pd.DataFrame] = []

    for name, cfg in datasets.items():
        if name in args.exclude:
            print(f"[{name}] excluded by --exclude")
            continue
        if cfg["status"] != "ingested":
            print(f"[{name}] status={cfg['status']} — skip")
            continue

        print(f"[{name}] loading...")
        df = load_and_absolutize(name, cfg)
        if df.empty:
            continue

        # verify image files exist, drop missing
        if not args.skip_exist_check:
            before = len(df)
            exists_mask = df["image_path"].map(lambda p: Path(p).exists())
            df = df[exists_mask].reset_index(drop=True)
            dropped = before - len(df)
            if dropped:
                print(f"  dropped {dropped} rows with missing image files")
        else:
            print(f"  (skipping file-existence check)")

        split_counts = df["split"].value_counts().to_dict()
        print(f"  rows={len(df)}  splits={split_counts}")

        # label coverage: which canonical labels have at least 1 positive
        coverage = [c for c in CANONICAL_LABELS if df[c].sum() > 0]
        print(f"  labels with positives ({len(coverage)}): {coverage}")

        frames.append(df)

    if not frames:
        print("No datasets loaded. Nothing to write.")
        return

    combined = pd.concat(frames, ignore_index=True)

    # Cross-dataset deduplication: same image_id from different datasets must not
    # appear in multiple splits. Policy: test > val > train (protect evaluation).
    _PRIO = {"test": 0, "val": 1, "train": 2}
    dup_ids = combined[combined.duplicated("image_id", keep=False)]["image_id"].unique()
    if len(dup_ids):
        combined["_split_prio"] = combined["split"].map(_PRIO).fillna(3)
        before = len(combined)
        combined = (
            combined.sort_values("_split_prio")
                    .drop_duplicates(subset="image_id", keep="first")
                    .drop(columns="_split_prio")
                    .reset_index(drop=True)
        )
        print(f"\n  Cross-dataset duplicates: {len(dup_ids)} image_ids, "
              f"dropped {before - len(combined)} rows (kept highest-priority split)")
    else:
        print("\n  No cross-dataset duplicate image_ids.")

    # ensure column order matches schema
    combined = combined[CSV_COLUMNS]

    print(f"\n{'='*60}")
    print(f"Combined rows:  {len(combined)}")
    print(f"Split breakdown: {combined['split'].value_counts().to_dict()}")
    print(f"Datasets: {combined['dataset'].value_counts().to_dict()}")

    # label coverage across all datasets
    covered = [c for c in CANONICAL_LABELS if combined[c].sum() > 0]
    uncovered = [c for c in CANONICAL_LABELS if combined[c].sum() == 0]
    print(f"\nLabels WITH signal ({len(covered)}): {covered}")
    print(f"Labels ALL-NaN   ({len(uncovered)}): {uncovered}")

    if args.dry_run:
        print("\n[dry-run] not writing CSV.")
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out, index=False)
    print(f"\nWrote {len(combined)} rows -> {out}")


if __name__ == "__main__":
    main()
