"""Ingest TBX11K into canonical preprocessed layout.

Label mapping:
  tb/      -> Tuberculosis=1.0
  sick/    -> Tuberculosis=0.0
  health/  -> Tuberculosis=0.0
  test/    -> skipped (unlabeled)
  extra/   -> skipped (files not present in zip)

Split from all_train.txt / all_val.txt (relative paths like tb/tb0005.png).

Usage:
    uv run python src/script/run_ingest_tbx11k.py \
        --dataset-root E:/research-cxr/dataset/tbx11k \
        --out-root E:/research-cxr/dataset/tbx11k/preprocessed
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tqdm import tqdm

from src.data.preprocess.common import (
    CSV_COLUMNS,
    process_path,
    verify_label_df,
    write_label_csv,
)
from src.data.label_space import CANONICAL_LABELS

APPLICABLE = {"Tuberculosis"}
DATASET = "tbx11k"


def _tb_label(rel_path: str) -> float | None:
    """Return Tuberculosis label from folder prefix. None = skip."""
    folder = rel_path.split("/")[0]
    if folder == "tb":
        return 1.0
    if folder in ("sick", "health"):
        return 0.0
    return None  # test/ or extra/ -> skip


def _load_split_map(label_dir: Path) -> dict[str, str]:
    """rel_path -> 'train' | 'val'"""
    split_map: dict[str, str] = {}
    for split, fname in [("train", "all_train.txt"), ("val", "all_val.txt")]:
        fpath = label_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(fpath)
        for line in fpath.read_text().splitlines():
            line = line.strip()
            if line:
                split_map[line] = split
    return split_map


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    out_root = Path(args.out_root)
    label_dir = dataset_root / "label"
    img_root = dataset_root / "raw" / "imgs"

    split_map = _load_split_map(label_dir)
    print(f"Split map: {sum(v=='train' for v in split_map.values())} train, "
          f"{sum(v=='val' for v in split_map.values())} val")

    img_dir = out_root / "images" / DATASET
    img_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    skipped = missing = errors = 0

    entries = sorted(split_map.items())
    for rel_path, split in tqdm(entries, desc=f"ingest {DATASET}"):
        tb_val = _tb_label(rel_path)
        if tb_val is None:
            skipped += 1
            continue

        src = img_root / rel_path.replace("/", "\\")
        if not src.exists():
            missing += 1
            continue

        image_id = Path(rel_path).stem
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

        row: dict = {
            "image_id": image_id,
            "image_path": f"{DATASET}/{image_id}.png",
            "dataset": DATASET,
            "split": split,
        }
        for c in CANONICAL_LABELS:
            row[c] = tb_val if c in APPLICABLE else float("nan")
        rows.append(row)

        if args.limit and len(rows) >= args.limit:
            break

    out_csv = out_root / "labels" / f"{DATASET}.csv"
    df = write_label_csv(rows, out_csv)
    verify_label_df(df, DATASET)
    splits = df["split"].value_counts().to_dict()
    print(f"\n{DATASET}: {len(df)} rows -> {out_csv}")
    print(f"  splits: {splits}")
    print(f"  annotated: {list(APPLICABLE)}")
    print(f"  skipped: {skipped}  missing: {missing}  errors: {errors}")


if __name__ == "__main__":
    main()
