"""Ingest COVIDx CXR-4 into canonical preprocessed layout.

Label mapping:
  positive -> COVID19_Pneumonia=1.0
  negative -> COVID19_Pneumonia=0.0
  all other 50 canonical labels -> NaN

Label file format (space-separated, 4 cols):
  patient_id  filename  class  source
  e.g.: 379 1e64990d1b40c1758a2aaa9c7f7a85_jumbo.jpeg negative cohen

Images are mixed JPEG + PNG; all output as PNG 224x224 RGB.

Usage:
    uv run python src/script/run_ingest_covidx.py \
        --dataset-root E:/research-cxr/dataset/covidx-cxr4 \
        --out-root E:/research-cxr/dataset/covidx-cxr4/preprocessed
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tqdm import tqdm

from src.data.preprocess.common import (
    process_path,
    verify_label_df,
    write_label_csv,
)
from src.data.label_space import CANONICAL_LABELS

APPLICABLE = {"COVID19_Pneumonia"}
DATASET = "covidx-cxr4"
SPLITS = ["train", "val", "test"]


def _load_split_labels(label_dir: Path) -> list[tuple[str, str, str, str]]:
    """Returns list of (split, patient_id, filename, class)."""
    entries = []
    for split in SPLITS:
        fpath = label_dir / f"{split}.txt"
        if not fpath.exists():
            raise FileNotFoundError(fpath)
        for line in fpath.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) >= 3:
                entries.append((split, parts[0], parts[1], parts[2]))
    return entries


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    out_root = Path(args.out_root)

    entries = _load_split_labels(dataset_root / "label")
    print(f"Entries: {len(entries)} total")
    for s in SPLITS:
        print(f"  {s}: {sum(1 for e in entries if e[0]==s)}")

    img_dir = out_root / "images" / DATASET
    img_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    skipped = missing = errors = 0

    for split, patient_id, filename, cls in tqdm(entries, desc=f"ingest {DATASET}"):
        src = dataset_root / "raw" / split / filename
        if not src.exists():
            missing += 1
            continue

        stem = Path(filename).stem
        safe_pid = patient_id.replace("/", "_").replace("\\", "_")
        image_id = f"{split}_{safe_pid}_{stem}"
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

        covid_val = 1.0 if cls == "positive" else 0.0
        row: dict = {
            "image_id": image_id,
            "image_path": f"{DATASET}/{image_id}.png",
            "dataset": DATASET,
            "split": split,
        }
        for c in CANONICAL_LABELS:
            row[c] = covid_val if c in APPLICABLE else float("nan")
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
    print(f"  skipped (existing): {skipped}  missing: {missing}  errors: {errors}")


if __name__ == "__main__":
    main()
