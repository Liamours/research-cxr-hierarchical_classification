"""Label-equivalence audit: map a dataset's raw label columns to the canonical
set and report coverage. Helps confirm name equivalence across datasets before
preprocessing.

    uv run python src/script/run_label_audit.py --raw-csv path/to/raw_labels.csv --dataset nih-cxr14

The raw CSV is read for its column names only; nothing is modified.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.data.label_map import load_equivalence


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-csv", required=True, help="raw dataset label CSV (read for column names only)")
    p.add_argument("--dataset", default=None, help="dataset key, e.g. chexpert | nih-cxr14 | vindr-cxr")
    p.add_argument("--equivalence", default="configs/label_equivalence.json")
    p.add_argument("--columns", nargs="*", default=None, help="override: explicit column names instead of CSV header")
    return p.parse_args()


def main():
    args = parse_args()
    equiv = load_equivalence(args.equivalence)
    cols = args.columns or list(pd.read_csv(args.raw_csv, nrows=0).columns)
    rep = equiv.coverage(cols, args.dataset)

    print(f"=== Label audit: dataset={args.dataset}  ({len(cols)} columns) ===")
    print(f"\nmapped ({len(rep['mapped'])}):")
    for raw, canon in rep["mapped"].items():
        print(f"  {raw:32s} -> {canon}")
    print(f"\nunmapped ({len(rep['unmapped'])}): {rep['unmapped']}")
    print(f"\ncanonical covered ({len(rep['canonical_covered'])}): {rep['canonical_covered']}")
    print(f"canonical MISSING ({len(rep['canonical_missing'])}): {rep['canonical_missing']}")


if __name__ == "__main__":
    main()
