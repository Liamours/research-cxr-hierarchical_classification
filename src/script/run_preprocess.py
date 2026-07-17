"""Preprocess a raw dataset into the canonical layout (one CLI for every dataset).

    uv run python src/script/run_preprocess.py --dataset nih-cxr14 \
        --raw-root data/raw/nih-cxr14 --out-root data/preprocessed

Raw files are read-only and never modified. Output is the standard layout:
    <out-root>/images/<dataset>/<image_id>.png   (224x224x3 uint8)
    <out-root>/labels/<dataset>.csv              (18-column canonical CSV)

Each dataset uses its adapter in src/data/preprocess/. The adapters raise
FileNotFoundError (with the missing path) if the expected raw files are absent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.data.preprocess import chexpert_plus, nih_cxr14, vindr_cxr

ADAPTERS = {
    "chexpert": chexpert_plus.preprocess_chexpert,
    "nih-cxr14": nih_cxr14.preprocess_nih,
    "vindr-cxr": vindr_cxr.preprocess_vindr,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=sorted(ADAPTERS))
    p.add_argument("--raw-root", required=True, help="directory holding the raw dataset")
    p.add_argument("--out-root", default="data/preprocessed")
    p.add_argument("--images-dir", default=None, help="NIH only: flattened images dir (default <raw-root>/images)")
    p.add_argument("--limit", type=int, default=None, help="cap number of studies (smoke runs)")
    return p.parse_args()


def main():
    args = parse_args()
    fn = ADAPTERS[args.dataset]
    kwargs = {"limit": args.limit}
    if args.dataset == "nih-cxr14" and args.images_dir:
        kwargs["images_dir"] = args.images_dir

    print(f"Preprocessing {args.dataset} from {args.raw_root} -> {args.out_root}")
    out_csv = fn(args.raw_root, args.out_root, **kwargs)
    df = pd.read_csv(out_csv)
    print(f"Wrote {len(df)} rows -> {out_csv}")
    print(f"Splits: {df['split'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
