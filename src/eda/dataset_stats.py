"""Exploratory data analysis for a canonical label CSV.

Runs six checks and prints tables (no files written unless figures are
explicitly requested):
  1. per-split sample counts
  2. per-condition class distribution (positive / negative / not-applicable)
  3. multi-label co-occurrence (top positive condition pairs)
  4. patient-level split-leakage check
  5. image integrity (size, channels, constant/blank images)
  6. label value sanity (only 0 / 1 / NaN)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.data.label_space import CANONICAL_LABELS


def split_counts(df: pd.DataFrame) -> pd.DataFrame:
    if "split" not in df.columns:
        return pd.DataFrame({"split": ["(none)"], "count": [len(df)]})
    return df.groupby("split").size().reset_index(name="count")


def class_distribution(df: pd.DataFrame, conditions: list[str] | None = None) -> pd.DataFrame:
    conditions = conditions or CANONICAL_LABELS
    rows = []
    for c in conditions:
        col = df[c]
        pos = int((col == 1).sum())
        neg = int((col == 0).sum())
        na = int(col.isna().sum())
        applicable = pos + neg
        rate = round(pos / applicable, 4) if applicable else float("nan")
        rows.append({"condition": c, "positive": pos, "negative": neg,
                     "not_applicable": na, "pos_rate": rate})
    return pd.DataFrame(rows)


def cooccurrence(df: pd.DataFrame, conditions: list[str] | None = None,
                 top_k: int = 10) -> pd.DataFrame:
    conditions = conditions or CANONICAL_LABELS
    mat = (df[conditions] == 1).astype(int).to_numpy()
    co = mat.T @ mat  # C x C positive co-occurrence counts
    pairs = []
    for i in range(len(conditions)):
        for j in range(i + 1, len(conditions)):
            pairs.append({"a": conditions[i], "b": conditions[j], "count": int(co[i, j])})
    out = pd.DataFrame(pairs).sort_values("count", ascending=False).head(top_k)
    return out.reset_index(drop=True)


def _patient_ids(df: pd.DataFrame, dataset: str) -> tuple[pd.Series, str]:
    if "patient_id" in df.columns:
        return df["patient_id"].astype(str), "patient"
    if dataset in ("mimic-cxr", "chexpert"):
        # image_id starts with the patient token, e.g. "p10000032_s..." / "patient00001_..."
        return df["image_id"].astype(str).str.split("_").str[0], "patient"
    return df["image_id"].astype(str), "image"


def split_leakage(df: pd.DataFrame, dataset: str) -> dict:
    if "split" not in df.columns:
        return {"granularity": "n/a", "n_leaked": 0, "leaked": []}
    pids, granularity = _patient_ids(df, dataset)
    tmp = pd.DataFrame({"pid": pids, "split": df["split"].values})
    per = tmp.groupby("pid")["split"].nunique()
    leaked = per[per > 1].index.tolist()
    return {"granularity": granularity, "n_leaked": len(leaked), "leaked": leaked[:20]}


def image_integrity(df: pd.DataFrame, image_root: Path, image_size: int = 224,
                    max_check: int = 5000) -> dict:
    image_root = Path(image_root)
    n = min(len(df), max_check)
    bad_size = bad_channels = constant = missing = 0
    for rel in tqdm(df["image_path"].head(n), desc="image integrity", dynamic_ncols=True):
        p = image_root / str(rel).replace("\\", "/")
        if not p.exists():
            missing += 1
            continue
        arr = np.array(Image.open(p))
        if arr.shape[:2] != (image_size, image_size):
            bad_size += 1
        if arr.ndim != 3 or arr.shape[2] != 3:
            bad_channels += 1
        elif int(arr.min()) == int(arr.max()):
            constant += 1
    return {"checked": n, "missing": missing, "bad_size": bad_size,
            "bad_channels": bad_channels, "constant_or_blank": constant}


def label_sanity(df: pd.DataFrame, conditions: list[str] | None = None) -> dict:
    conditions = conditions or CANONICAL_LABELS
    offenders = {}
    for c in conditions:
        bad = [v for v in df[c].dropna().unique() if v not in (0.0, 1.0)]
        if bad:
            offenders[c] = bad
    return {"ok": not offenders, "offenders": offenders}


def run_eda(cfg, max_image_check: int = 5000) -> dict:
    df = pd.read_csv(Path(cfg.data.label_csv))
    dataset = cfg.data.dataset

    sc = split_counts(df)
    cd = class_distribution(df)
    co = cooccurrence(df)
    leak = split_leakage(df, dataset)
    integ = image_integrity(df, cfg.data.image_root, cfg.data.image_size, max_image_check)
    sanity = label_sanity(df)

    print(f"\n=== EDA: {dataset}  ({len(df)} rows) ===")
    print("\n[split counts]")
    print(sc.to_string(index=False))
    print("\n[class distribution]")
    print(cd.to_string(index=False))
    print("\n[top co-occurring positive pairs]")
    print(co.to_string(index=False))
    print(f"\n[split leakage]  granularity={leak['granularity']}  "
          f"leaked_{leak['granularity']}s={leak['n_leaked']}")
    if leak["n_leaked"]:
        print(f"  LEAK: {leak['leaked']}")
    print(f"\n[image integrity]  {integ}")
    print(f"\n[label sanity]  ok={sanity['ok']}"
          + ("" if sanity["ok"] else f"  offenders={sanity['offenders']}"))

    return {"split_counts": sc, "class_distribution": cd, "cooccurrence": co,
            "leakage": leak, "integrity": integ, "label_sanity": sanity}
