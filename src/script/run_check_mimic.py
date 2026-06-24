"""Reliable audit of MIMIC-CXR image completeness.

For every study in metadata (train + valid), regardless of has_image flag,
check whether view1_frontal.png exists on disk using the same path formula
as run_download_mimic.py:
    archive-preprocessed/{split}/p{subject_id}/s{study_id}/view1_frontal.png

Reports:
  - present:  file on disk
  - missing:  file not on disk
  - no breakdown by has_image flag — just ground truth vs disk

Usage:
    uv run python src/script/run_check_mimic.py \\
        --dataset-root E:/research-cxr/dataset/mimic-cxr
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from tqdm import tqdm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--save-missing", default=None,
                    help="write missing study_ids to this file")
    args = ap.parse_args()

    root = Path(args.dataset_root)
    label_dir = root / "label"
    archive_root = root / "archive-preprocessed"

    # official split: study_id (int) -> "train" | "validate" | "test"
    splits_df = pd.read_csv(label_dir / "mimic-cxr-2.0.0-split.csv")
    study_split: dict[int, str] = (
        splits_df.groupby("study_id")["split"].first().to_dict()
    )

    # official metadata: subject_id per study
    meta_df = pd.read_csv(label_dir / "mimic-cxr-2.0.0-metadata.csv",
                          usecols=["subject_id", "study_id"])
    study_subject: dict[int, int] = (
        meta_df.groupby("study_id")["subject_id"].first().to_dict()
    )

    # all studies we care about: from metadata_train + metadata_valid
    meta_train = pd.read_csv(label_dir / "metadata_train.csv",
                             usecols=["study_id"])
    meta_valid = pd.read_csv(label_dir / "metadata_valid.csv",
                             usecols=["study_id"])
    all_meta = pd.concat([meta_train, meta_valid], ignore_index=True)

    # strip leading 's' if present -> int
    def parse_sid(raw) -> int:
        s = str(raw).strip()
        return int(s.lstrip("s"))

    study_ids = [parse_sid(s) for s in all_meta["study_id"].unique()]
    print(f"Total unique studies in metadata: {len(study_ids)}")

    present = []
    missing = []
    no_subject = []
    no_split = []

    for sid in tqdm(study_ids, desc="checking disk"):
        subject_id = study_subject.get(sid)
        if subject_id is None:
            no_subject.append(sid)
            continue

        split = study_split.get(sid, "train").replace("validate", "valid")

        png = archive_root / split / f"p{subject_id}" / f"s{sid}" / "view1_frontal.png"
        if png.exists():
            present.append(sid)
        else:
            missing.append((sid, str(png)))

    total = len(study_ids)
    pct = 100 * len(present) / max(total, 1)

    print(f"\n{'='*60}")
    print(f"Total studies in metadata:    {total}")
    print(f"Present on disk:              {len(present)}  ({pct:.1f}%)")
    print(f"Missing from disk:            {len(missing)}")
    print(f"No subject_id in metadata:    {len(no_subject)}")
    print(f"No split in split.csv:        {no_split}")
    print(f"{'='*60}")

    if missing:
        print(f"\nFirst 10 missing:")
        for sid, path in missing[:10]:
            print(f"  study={sid}  expected={path}")

    if args.save_missing and missing:
        out = Path(args.save_missing)
        out.write_text("\n".join(str(s) for s, _ in missing))
        print(f"\nMissing study_ids saved to: {out}")


if __name__ == "__main__":
    main()
