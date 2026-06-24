"""Verify dataset integrity against configs/dataset_registry.json.

Checks per ingested dataset:
  - Image count on disk matches expected
  - CSV row count matches expected
  - CSV columns = 55 (4 id + 51 canonical)
  - Split distribution matches expected
  - No duplicate image_ids
  - Label values are 0.0 / 1.0 / NaN only

Usage:
    uv run python src/script/run_verify_datasets.py
    uv run python src/script/run_verify_datasets.py --dataset chexpert
    uv run python src/script/run_verify_datasets.py --sample 100
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.data.preprocess.common import CSV_COLUMNS
from src.data.label_space import CANONICAL_LABELS

REGISTRY = Path(__file__).resolve().parents[2] / "configs" / "dataset_registry.json"
PASS = "\033[92m✅\033[0m"
FAIL = "\033[91m❌\033[0m"
WARN = "\033[93m⚠️\033[0m"
SKIP = "⏳"


def check_dataset(name: str, cfg: dict, sample: int | None) -> bool:
    root = Path(cfg["root"])
    ok = True

    if cfg["status"] != "ingested":
        print(f"  {SKIP} status={cfg['status']} — skip")
        return True

    img_dir = root / cfg["preprocessed_image_dir"]
    csv_path = root / cfg["preprocessed_label_csv"]

    # -- image count --
    if img_dir.exists():
        n_imgs = sum(1 for _ in img_dir.glob("*.png"))
        exp = cfg.get("expected_total_images")
        if exp is not None and n_imgs != exp:
            print(f"  {FAIL} images: {n_imgs} on disk, expected {exp}")
            ok = False
        else:
            print(f"  {PASS} images: {n_imgs}")
    else:
        print(f"  {FAIL} image dir missing: {img_dir}")
        ok = False

    # -- csv exists --
    if not csv_path.exists():
        print(f"  {FAIL} CSV missing: {csv_path}")
        return False

    df = pd.read_csv(csv_path, nrows=sample) if sample else pd.read_csv(csv_path)

    # -- row count --
    exp_rows = cfg.get("expected_total_rows")
    if exp_rows is not None:
        n_rows = len(df) if not sample else f"{len(df)} (sampled)"
        match = len(df) == exp_rows if not sample else True
        icon = PASS if match else FAIL
        print(f"  {icon} CSV rows: {n_rows}{'' if not sample else f' (full expected {exp_rows})'}")
        if not match:
            ok = False
    else:
        print(f"  {PASS} CSV rows: {len(df)} (no expectation set)")

    # -- columns --
    missing_cols = [c for c in CSV_COLUMNS if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in CSV_COLUMNS]
    if missing_cols:
        print(f"  {FAIL} missing columns: {missing_cols}")
        ok = False
    elif extra_cols:
        print(f"  {WARN} extra columns: {extra_cols}")
    else:
        print(f"  {PASS} columns: {len(df.columns)} (correct)")

    # -- splits --
    exp_splits = cfg.get("expected_splits")
    if exp_splits and not sample:
        actual_splits = df["split"].value_counts().to_dict()
        for split, exp_n in exp_splits.items():
            actual_n = actual_splits.get(split, 0)
            icon = PASS if actual_n == exp_n else FAIL
            print(f"  {icon} split {split}: {actual_n} (expected {exp_n})")
            if actual_n != exp_n:
                ok = False

    # -- duplicates --
    if not sample:
        n_dup = df["image_id"].duplicated().sum()
        if n_dup:
            print(f"  {FAIL} duplicate image_ids: {n_dup}")
            ok = False
        else:
            print(f"  {PASS} no duplicate image_ids")

    # -- label value sanity --
    bad_labels = []
    for c in CANONICAL_LABELS:
        if c not in df.columns:
            continue
        vals = set(df[c].dropna().unique())
        invalid = vals - {0.0, 1.0}
        if invalid:
            bad_labels.append(f"{c}={invalid}")
    if bad_labels:
        print(f"  {FAIL} invalid label values: {bad_labels[:5]}")
        ok = False
    else:
        annotated = [c for c in CANONICAL_LABELS if c in df.columns and df[c].notna().any()]
        print(f"  {PASS} label values OK  |  annotated ({len(annotated)}): {annotated}")

    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, help="check one dataset only")
    ap.add_argument("--sample", type=int, default=None, help="read only first N rows (fast smoke test)")
    args = ap.parse_args()

    registry = json.loads(REGISTRY.read_text())
    datasets = registry["datasets"]

    if args.dataset:
        if args.dataset not in datasets:
            print(f"Unknown dataset: {args.dataset}. Available: {list(datasets)}")
            sys.exit(1)
        datasets = {args.dataset: datasets[args.dataset]}

    results: dict[str, bool] = {}
    for name, cfg in datasets.items():
        print(f"\n{'='*50}\n{name}  [{cfg['status']}]")
        results[name] = check_dataset(name, cfg, args.sample)

    print(f"\n{'='*50}")
    all_ok = True
    for name, ok in results.items():
        cfg = registry["datasets"][name]
        if cfg["status"] != "ingested":
            print(f"  {SKIP} {name} (pending)")
        elif ok:
            print(f"  {PASS} {name}")
        else:
            print(f"  {FAIL} {name}")
            all_ok = False

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
