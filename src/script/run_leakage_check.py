"""Protocol §11a — patient-level (and image-level) data leakage check.

Two checks:
  1. Image-level: no image_id appears in more than one split in the combined CSV.
  2. Dataset-level: within each source dataset, the same patient prefix does not
     appear in both train and test splits (where patient_id is inferrable from
     image_id via the naming convention of each dataset).

A non-zero overlap count BLOCKS training — it indicates a preprocessing error.

Usage:
    uv run python src/script/run_leakage_check.py
    uv run python src/script/run_leakage_check.py --csv C:/rifqi/research-cxr-hierarchical_classification/dataset/combined/combined.csv
    uv run python src/script/run_leakage_check.py --csv ... --per-dataset-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd


# --------------------------------------------------------------------------
# Per-dataset patient-ID extractors.
# Returns a patient identifier string given an image_id, or None if unknown.
# --------------------------------------------------------------------------

def _nih_patient_id(image_id: str) -> str | None:
    # NIH: 00000001_000.png  -> patient 00000001
    parts = image_id.split("_")
    return parts[0] if len(parts) >= 2 else None


def _chexpert_patient_id(image_id: str) -> str | None:
    # CheXpert+: patient64541_study1_view1_frontal -> patient64541
    parts = image_id.split("_")
    return parts[0] if parts and parts[0].startswith("patient") else None


def _vindr_patient_id(image_id: str) -> str | None:
    # VinDr-CXR: image IDs are study-level UUIDs; no per-patient grouping
    # available from image_id alone — skip patient check.
    return None


def _covidx_patient_id(image_id: str) -> str | None:
    # COVIDx-CXR4: IDs are per-image; patient linkage not in CSV.
    return None


def _tbx_patient_id(image_id: str) -> str | None:
    # TBX11K: IDs are per-image filenames; no patient structure.
    return None


DATASET_PATIENT_FN = {
    "nih-cxr14":    _nih_patient_id,
    "chexpert":     _chexpert_patient_id,
    "vindr-cxr":    _vindr_patient_id,
    "vindr-pcxr":   _vindr_patient_id,
    "covidx-cxr4":  _covidx_patient_id,
    "tbx11k":       _tbx_patient_id,
}


def check_image_leakage(df: pd.DataFrame) -> dict:
    """Check no image_id appears in more than one split."""
    groups = df.groupby("image_id")["split"].agg(lambda x: set(x))
    leaked = groups[groups.apply(lambda s: len(s) > 1)]
    return {
        "total_images": len(df),
        "unique_image_ids": len(groups),
        "leaked_ids": int(len(leaked)),
        "examples": list(leaked.index[:5]) if len(leaked) > 0 else [],
        "passes": len(leaked) == 0,
    }


def check_patient_leakage(df: pd.DataFrame, dataset: str, patient_fn) -> dict:
    """Check no patient appears in both train and test splits for one dataset."""
    sub = df[df["dataset"] == dataset].copy()
    if len(sub) == 0:
        return {"dataset": dataset, "skipped": True, "reason": "no rows"}

    sub["_patient"] = sub["image_id"].map(patient_fn)
    sub = sub[sub["_patient"].notna()]
    if len(sub) == 0:
        return {"dataset": dataset, "skipped": True, "reason": "patient ID not inferrable from image_id"}

    train_patients = set(sub[sub["split"] == "train"]["_patient"])
    test_patients  = set(sub[sub["split"] == "test"]["_patient"])
    val_patients   = set(sub[sub["split"] == "val"]["_patient"])

    overlap_train_test = train_patients & test_patients
    overlap_val_test   = val_patients & test_patients
    overlap_train_val  = train_patients & val_patients

    return {
        "dataset": dataset,
        "skipped": False,
        "train_patients": len(train_patients),
        "val_patients":   len(val_patients),
        "test_patients":  len(test_patients),
        "overlap_train_test": len(overlap_train_test),
        "overlap_val_test":   len(overlap_val_test),
        "overlap_train_val":  len(overlap_train_val),
        "examples_train_test": list(overlap_train_test)[:3],
        "passes": len(overlap_train_test) == 0 and len(overlap_val_test) == 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="C:/rifqi/research-cxr-hierarchical_classification/dataset/combined/combined.csv")
    ap.add_argument("--per-dataset-only", action="store_true")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        sys.exit(1)

    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path, usecols=["image_id", "dataset", "split"], low_memory=False)
    print(f"  {len(df):,} rows  splits={dict(df['split'].value_counts())}")

    all_pass = True

    # 1. Image-level check
    if not args.per_dataset_only:
        print("\n=== Check 1: image_id uniqueness across splits ===")
        r = check_image_leakage(df)
        status = "PASS" if r["passes"] else "FAIL"
        print(f"  {status}  leaked_ids={r['leaked_ids']}  unique_ids={r['unique_image_ids']}")
        if not r["passes"]:
            print(f"  Examples: {r['examples']}")
            all_pass = False

    # 2. Per-dataset patient-level check
    print("\n=== Check 2: patient-level split integrity per dataset ===")
    datasets = df["dataset"].unique()
    for ds in sorted(datasets):
        fn = DATASET_PATIENT_FN.get(ds)
        if fn is None:
            print(f"  [{ds}] no patient extractor registered — skip")
            continue
        r = check_patient_leakage(df, ds, fn)
        if r.get("skipped"):
            print(f"  [{ds}] SKIP — {r['reason']}")
            continue
        status = "PASS" if r["passes"] else "FAIL"
        print(f"  [{ds}] {status}  "
              f"train={r['train_patients']} val={r['val_patients']} test={r['test_patients']}  "
              f"train∩test={r['overlap_train_test']} val∩test={r['overlap_val_test']}")
        if not r["passes"]:
            print(f"    examples: {r['examples_train_test']}")
            all_pass = False

    print(f"\n{'='*50}")
    if all_pass:
        print("LEAKAGE CHECK: ALL PASS — safe to train")
    else:
        print("LEAKAGE CHECK: FAILURES DETECTED — fix before training")
        sys.exit(1)


if __name__ == "__main__":
    main()
