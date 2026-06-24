"""Generate a small synthetic dataset in the canonical layout, so the full
pipeline (train / eval / inference / xai) can be run end to end with no real
data. For demos, CI, and smoke runs only -- the images are random noise with
weak injected label signal, not real radiographs.

    uv run python src/script/make_synthetic_dataset.py --out-root data/synthetic --n 80
    uv run python src/script/run_train.py --config configs/synthetic_demo.yaml

Output: <out-root>/images/synthetic/*.png (224x224x3) + <out-root>/labels/synthetic.csv
with patient-disjoint train/val/test splits.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.data.label_space import CANONICAL_LABELS
from src.data.preprocess import common


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", default="data/synthetic")
    p.add_argument("--n", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    n_train = int(args.n * 0.7)
    n_val = int(args.n * 0.15)

    records = []
    for i in range(args.n):
        split = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
        # weak signal: a brighter patch correlates with Pulmonary_Edema/Pneumonia positive
        img = rng.integers(20, 120, (90, 90), np.uint8)
        pos = bool(i % 2)
        if pos:
            img[30:60, 30:60] = rng.integers(180, 255, (30, 30), np.uint8)
        labels = {c: 0.0 for c in CANONICAL_LABELS}
        labels["Pulmonary_Edema"] = float(pos)
        labels["Pneumonia"] = float(pos)
        labels["Cardiomegaly"] = float((i % 3) == 0)
        records.append({"image_id": f"p{i:04d}_s1_d0", "src_image": img,
                        "split": split, "labels": labels, "applicable": None})

    rows = common.preprocess_records(records, args.out_root, "synthetic")
    out_csv = Path(args.out_root) / "labels" / "synthetic.csv"
    common.write_label_csv(rows, out_csv)
    import pandas as pd
    counts = pd.DataFrame(rows)["split"].value_counts().to_dict()
    print(f"Wrote {len(rows)} synthetic samples -> {out_csv}  splits={counts}")
    print(f"Images -> {Path(args.out_root) / 'images' / 'synthetic'}")


if __name__ == "__main__":
    main()
