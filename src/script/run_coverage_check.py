"""Check label coverage across ingested datasets (excluding MIMIC-CXR).

Reports positive count, applicable count, and prevalence per canonical label.
Classifies each label as trainable / sparse / no-signal.

Usage:
    uv run python src/script/run_coverage_check.py
    uv run python src/script/run_coverage_check.py --exclude mimic-cxr vindr-pcxr
    uv run python src/script/run_coverage_check.py --min-positives 500
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.data.label_space import CANONICAL_LABELS

REGISTRY = Path(__file__).resolve().parents[2] / "configs" / "dataset_registry.json"
DEFAULT_EXCLUDE = {"mimic-cxr"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", nargs="*", default=list(DEFAULT_EXCLUDE))
    ap.add_argument("--min-positives", type=int, default=100,
                    help="threshold for 'trainable' classification")
    args = ap.parse_args()

    registry = json.loads(REGISTRY.read_text())
    frames = {}

    for name, cfg in registry["datasets"].items():
        if name in args.exclude:
            print(f"[{name}] excluded")
            continue
        if cfg["status"] != "ingested":
            print(f"[{name}] status={cfg['status']} — skip")
            continue
        csv_path = Path(cfg["root"]) / cfg["preprocessed_label_csv"]
        if not csv_path.exists():
            print(f"[{name}] CSV missing: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        frames[name] = df
        print(f"[{name}] {len(df):,} rows")

    if not frames:
        print("No data loaded.")
        return

    combined = pd.concat(frames.values(), ignore_index=True)
    total = len(combined)
    print(f"\nTotal rows: {total:,}  |  datasets: {list(frames)}\n")

    # per-label stats
    rows = []
    for label in CANONICAL_LABELS:
        pos = int(combined[label].sum()) if label in combined.columns else 0
        applicable = int(combined[label].notna().sum())
        prev = pos / applicable * 100 if applicable > 0 else 0.0
        # which datasets contribute positives
        sources = [
            name for name, df in frames.items()
            if label in df.columns and df[label].sum() > 0
        ]
        rows.append((label, pos, applicable, prev, sources))

    rows.sort(key=lambda x: -x[1])

    # display
    trainable = [r for r in rows if r[1] >= args.min_positives]
    sparse    = [r for r in rows if 0 < r[1] < args.min_positives]
    zero      = [r for r in rows if r[1] == 0]

    print(f"{'Label':<42} {'Pos':>8} {'Applic':>8} {'Prev%':>7}  Sources")
    print("-" * 90)

    print(f"=== TRAINABLE (>={args.min_positives} positives): {len(trainable)} labels ===")
    for label, pos, app, prev, src in trainable:
        print(f"  {label:<40} {pos:>8,} {app:>8,} {prev:>6.1f}%  {src}")

    print(f"\n=== SPARSE (<{args.min_positives} positives): {len(sparse)} labels ===")
    for label, pos, app, prev, src in sparse:
        print(f"  {label:<40} {pos:>8,} {app:>8,} {prev:>6.1f}%  {src}")

    print(f"\n=== NO SIGNAL (0 positives): {len(zero)} labels ===")
    for label, pos, app, prev, src in zero:
        print(f"  {label:<40}   all-NaN across all datasets")

    print(f"\nSUMMARY")
    print(f"  Trainable  (>={args.min_positives} pos): {len(trainable)}")
    print(f"  Sparse     (<{args.min_positives} pos):  {len(sparse)}")
    print(f"  No signal  (0 pos):       {len(zero)}")
    print(f"  Total canonical:          {len(CANONICAL_LABELS)}")


if __name__ == "__main__":
    main()
