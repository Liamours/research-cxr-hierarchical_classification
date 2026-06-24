"""Ingest a prior 14-class manifest + ImageNet-normalized .npy image set into the
Thread-2 canonical layout. Built for the E:\\research-cxr datasets whose images were
preprocessed (by the research-vlm_cxr Task_B pipeline) to (3,224,224) float32
ImageNet-normalized arrays, with a per-image label manifest.

It de-normalizes each .npy back to a 224x224x3 uint8 PNG and maps the manifest's
raw label columns to canonical diseases via configs/label_equivalence.json, then
writes the standard canonical output:
  <out-root>/images/<dataset>/<image_id>.png
  <out-root>/labels/<dataset>.csv

Streams one image at a time (low memory) and shows a tqdm progress bar.

  uv run python src/script/run_ingest_npy.py --dataset nih-cxr14 \
      --src-root E:/research-cxr/dataset/nih-cxr14 \
      --out-root E:/research-cxr/dataset/nih-cxr14/canonical
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.data.label_space import CANONICAL_LABELS
from src.data.label_map import load_equivalence
from src.data.preprocess import common

MANIFEST = {
    "nih-cxr14": "nih_14class_manifest.csv",
    "vindr-cxr": "vinbig_14class_manifest.csv",
}
SPLIT_DIRS = ("train", "val", "valid", "test")
_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)


def npy_to_uint8(arr: np.ndarray) -> np.ndarray:
    """(3,H,W) ImageNet-normalized float32 -> (H,W,3) uint8."""
    img = np.clip(arr * _STD + _MEAN, 0.0, 1.0)
    return (img.transpose(1, 2, 0) * 255.0).round().astype(np.uint8)


def _convert(npy: Path, dest: Path, attempts: int = 4) -> None:
    """Load .npy -> save PNG, retrying transient IO errors (e.g. a flaky external
    drive). Re-raises if it still fails after `attempts` (drive truly gone)."""
    for i in range(attempts):
        try:
            Image.fromarray(npy_to_uint8(np.load(npy))).save(dest)
            return
        except (OSError, ValueError):
            if i == attempts - 1:
                raise
            time.sleep(3)


def _find_npy(src_root: Path, split: str, stem: str) -> Path | None:
    cand = src_root / "preprocessed" / split / f"{stem}.npy"
    if cand.exists():
        return cand
    for sp in SPLIT_DIRS:  # fall back if the manifest split name != folder name
        c = src_root / "preprocessed" / sp / f"{stem}.npy"
        if c.exists():
            return c
    return None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=sorted(MANIFEST))
    p.add_argument("--src-root", required=True, help="dir holding label/ and preprocessed/")
    p.add_argument("--out-root", required=True, help="canonical output dir (images/ + labels/)")
    p.add_argument("--limit", type=int, default=None, help="process only N rows (smoke test)")
    return p.parse_args()


def main():
    args = parse_args()
    dataset = args.dataset
    src_root, out_root = Path(args.src_root), Path(args.out_root)

    manifest = pd.read_csv(src_root / "label" / MANIFEST[dataset])
    if args.limit is not None:
        manifest = manifest.head(args.limit)

    label_map = load_equivalence().dataset_to_canonical(dataset)  # raw column -> canonical
    present = {raw: canon for raw, canon in label_map.items() if raw in manifest.columns}
    applicable = set(present.values())  # only diseases this manifest actually annotates
    img_dir = out_root / "images" / dataset
    img_dir.mkdir(parents=True, exist_ok=True)

    rows, missing, skipped = [], 0, 0
    for _, r in tqdm(manifest.iterrows(), total=len(manifest), desc=f"ingest {dataset}", dynamic_ncols=True):
        stem = Path(str(r["image_id"])).stem
        split = str(r.get("split", "train"))
        split = "val" if split == "valid" else split
        dest = img_dir / f"{stem}.png"
        if dest.exists() and dest.stat().st_size > 0:   # resume: already converted
            skipped += 1
        else:
            npy = _find_npy(src_root, split, stem)
            if npy is None:
                missing += 1
                continue
            _convert(npy, dest)

        positives = {canon for raw, canon in present.items()
                     if common.map_uncertain_value(r.get(raw)) == 1.0}
        row = {"image_id": stem, "image_path": f"{dataset}/{stem}.png",
               "dataset": dataset, "split": split}
        for c in CANONICAL_LABELS:
            row[c] = (1.0 if c in positives else 0.0) if c in applicable else float("nan")
        rows.append(row)

    out_csv = out_root / "labels" / f"{dataset}.csv"
    common.verify_label_df(common.write_label_csv(rows, out_csv), dataset)
    sig = sorted(applicable)
    print(f"\n{dataset}: {len(rows)} rows -> {out_csv}  ({skipped} already existed, missing npy: {missing})")
    print(f"  annotated canonical diseases ({len(sig)}): {sig}")
    print(f"  splits: {pd.DataFrame(rows)['split'].value_counts().to_dict() if rows else '{}'}")


if __name__ == "__main__":
    main()
