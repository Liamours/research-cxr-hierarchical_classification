"""Multi-label CXR classification dataset.

Reads the standardized 55-column label CSV (image_id, image_path, dataset,
split, + 51 canonical conditions). Label values are 0, 1, or NaN, where NaN
means the condition was not annotated by that dataset. NaN is never treated
as a confirmed negative: it is carried as 0 in the label tensor but flagged 0
in label_mask so the loss and metrics ignore it.

Init is fully vectorized (numpy column ops, no iterrows). The optional
skip_missing_check skips per-row Path.exists() calls — use for verified CSVs
where missing-file checks were already run at ingest time.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.label_space import CANONICAL_LABELS
from src.data.segmentation import apply_mask_conditioning


class CxrClsDataset(Dataset):
    def __init__(
        self,
        label_csv: str | Path,
        image_root: str | Path,
        split: str | None,
        transform,
        conditions: list[str] | None = None,
        seg_provider=None,
        seg_method: str = "concat_channel",
        skip_missing_check: bool = False,
    ):
        self.image_root = Path(image_root)
        self.transform = transform
        self.conditions = conditions or CANONICAL_LABELS
        self.seg_provider = seg_provider
        self.seg_method = seg_method

        df = pd.read_csv(Path(label_csv), low_memory=False)
        if split is not None and "split" in df.columns:
            df = df[df["split"] == split].reset_index(drop=True)

        # Resolve image paths (vectorized string op, then per-path absolutization)
        raw = df["image_path"].astype(str).str.replace("\\", "/", regex=False)
        abs_paths = [
            Path(p) if Path(p).is_absolute() else self.image_root / p
            for p in raw
        ]

        # Existence filter
        if not skip_missing_check:
            keep_mask = np.array([os.path.exists(p) for p in abs_paths], dtype=bool)
            self.skipped_missing = int((~keep_mask).sum())
            keep_idx = np.where(keep_mask)[0]
        else:
            self.skipped_missing = 0
            keep_idx = np.arange(len(df))

        # Vectorized label + mask extraction (N × C float32, NaN → 0 + mask=0)
        present_cols = [c for c in self.conditions if c in df.columns]
        missing_cols = [c for c in self.conditions if c not in df.columns]

        if present_cols:
            raw_labels = df[present_cols].values.astype(np.float32)    # (N, len(present))
            raw_mask   = (~np.isnan(raw_labels)).astype(np.float32)
            raw_labels  = np.nan_to_num(raw_labels, nan=0.0)
        else:
            raw_labels = np.zeros((len(df), 0), dtype=np.float32)
            raw_mask   = np.zeros((len(df), 0), dtype=np.float32)

        # Re-order into full conditions list (missing cols are all-zero / all-masked)
        col_map = {c: i for i, c in enumerate(present_cols)}
        full_labels = np.zeros((len(df), len(self.conditions)), dtype=np.float32)
        full_mask   = np.zeros((len(df), len(self.conditions)), dtype=np.float32)
        for out_j, c in enumerate(self.conditions):
            if c in col_map:
                in_j = col_map[c]
                full_labels[:, out_j] = raw_labels[:, in_j]
                full_mask[:, out_j]   = raw_mask[:, in_j]
            # missing_cols: full_labels[:, out_j] = 0, full_mask[:, out_j] = 0 (already)

        image_ids = df["image_id"].astype(str).tolist()

        self.records: list[dict] = [
            {
                "image_id":   image_ids[i],
                "image_path": abs_paths[i],
                "labels":     full_labels[i],   # np.float32 array, converted in __getitem__
                "label_mask": full_mask[i],
            }
            for i in keep_idx
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        img = Image.open(rec["image_path"]).convert("RGB")
        pixel_values = self.transform(img)
        if self.seg_provider is not None:
            seg = self.seg_provider.get(rec["image_id"])
            if seg is None:
                seg = torch.zeros((1, pixel_values.shape[1], pixel_values.shape[2]))
            pixel_values = apply_mask_conditioning(pixel_values, seg, self.seg_method)
        return {
            "pixel_values": pixel_values,
            "labels":       torch.from_numpy(rec["labels"]),
            "label_mask":   torch.from_numpy(rec["label_mask"]),
            "image_id":     rec["image_id"],
        }
