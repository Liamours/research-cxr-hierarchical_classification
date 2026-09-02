"""A8: NIH box provenance. The manuscript's localization section (main.tex
Sec. IV-B) states the un-rescaled NIH bounding-box source file was
unavailable and treats the resulting IoU/AP numbers as provisional. It is
not unavailable: BBox_List_2017.csv ships inside the torchxrayvision
package data (a dependency this project already uses), in the same
1024x1024 coordinate space as the released NIH images.

This script persists a project-native copy of that raw file, recomputes
correct 224x224-space boxes using the project's own documented
preprocessing (src/data/preprocess/common.py: resize shortest side to
256, center-crop 224 -> scale 0.25, crop offset 16 on both axes, exactly
matching how the training images themselves were made), and prints a
diagnostic comparing the correct boxes against the previously-committed
(and undocumented) dataset/nih-cxr14/preprocessed/labels/nih_bboxes_224.csv.

That diagnostic identifies the old file's actual error: it scaled each
box by 224 / OriginalImage[Width or Height] (the pre-NIH-release scan
size from Data_Entry_2017.csv, e.g. 2992x2991 for the first bbox image)
instead of the fixed 1024 the box coordinates and the released PNGs
actually use. Confirmed by exact reproduction, not inference: applying
that formula to the raw boxes reproduces every stored coordinate in
nih_bboxes_224.csv to 5+ decimal places.

    uv run python src/script/run_fix_nih_bboxes.py
"""
from __future__ import annotations

import csv
import gzip
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torchxrayvision as xrv

ORIGINAL_SIZE = 1024.0
SCALE = 256.0 / ORIGINAL_SIZE  # 0.25
CROP_OFFSET = 16.0             # (256 - 224) // 2, both axes (square post-resize)

NIH_DIR = Path(r"C:\rifqi\research-cxr-hierarchical_classification\dataset\nih-cxr14\preprocessed\labels")
RAW_OUT = NIH_DIR / "nih_bbox_list_2017_raw.csv"
OLD_FILE = NIH_DIR / "nih_bboxes_224.csv"
FIXED_OUT = NIH_DIR / "nih_bboxes_224_corrected.csv"


def _package_bbox_path() -> Path:
    return Path(xrv.__file__).resolve().parent / "data" / "BBox_List_2017.csv.gz"


def load_raw_boxes() -> list[dict]:
    with gzip.open(_package_bbox_path(), "rt") as f:
        r = csv.reader(f)
        next(r)  # header: Image Index,Finding Label,Bbox [x,y,w,h],,,
        rows = []
        for row in r:
            rows.append({
                "image_id": row[0], "label": row[1],
                "x": float(row[2]), "y": float(row[3]),
                "w": float(row[4]), "h": float(row[5]),
            })
    return rows


def persist_raw_copy(rows: list[dict]) -> None:
    RAW_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(RAW_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "label", "x", "y", "w", "h"])
        for r in rows:
            w.writerow([r["image_id"], r["label"], r["x"], r["y"], r["w"], r["h"]])


def correct_224_box(r: dict) -> tuple[float, float, float, float]:
    x1 = r["x"] * SCALE - CROP_OFFSET
    y1 = r["y"] * SCALE - CROP_OFFSET
    x2 = (r["x"] + r["w"]) * SCALE - CROP_OFFSET
    y2 = (r["y"] + r["h"]) * SCALE - CROP_OFFSET
    return x1, y1, x2, y2


def write_corrected(rows: list[dict]) -> None:
    with open(FIXED_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "label", "x1", "y1", "x2", "y2"])
        for r in rows:
            x1, y1, x2, y2 = correct_224_box(r)
            w.writerow([r["image_id"], r["label"], x1, y1, x2, y2])


def diagnose_old_file(rows: list[dict]) -> None:
    if not OLD_FILE.exists():
        print(f"old file not found at {OLD_FILE}, skipping diagnostic")
        return
    with open(OLD_FILE) as f:
        old_rows = list(csv.DictReader(f))
    if len(old_rows) != len(rows):
        print(f"row count mismatch: old={len(old_rows)} raw={len(rows)}, skipping diagnostic")
        return
    correct_areas, old_areas = [], []
    for raw, old in zip(rows, old_rows):
        assert raw["image_id"] == old["image_id"], (raw["image_id"], old["image_id"])
        x1, y1, x2, y2 = correct_224_box(raw)
        correct_areas.append((x2 - x1) * (y2 - y1) / (224 * 224) * 100)
        ox1, oy1, ox2, oy2 = (float(old["x1"]), float(old["y1"]),
                              float(old["x2"]), float(old["y2"]))
        old_areas.append((ox2 - ox1) * (oy2 - oy1) / (224 * 224) * 100)
    n = len(rows)
    print(f"old file (buggy scale): mean area {sum(old_areas)/n:.2f}%  "
          f"min {min(old_areas):.2f}%  max {max(old_areas):.2f}%")
    print(f"corrected (scale 0.25, crop 16): mean area {sum(correct_areas)/n:.2f}%  "
          f"min {min(correct_areas):.2f}%  max {max(correct_areas):.2f}%")


def main() -> None:
    rows = load_raw_boxes()
    print(f"loaded {len(rows)} raw boxes from {_package_bbox_path()}")
    persist_raw_copy(rows)
    print(f"persisted raw copy -> {RAW_OUT}")
    write_corrected(rows)
    print(f"wrote corrected 224-space boxes -> {FIXED_OUT}")
    diagnose_old_file(rows)


if __name__ == "__main__":
    main()
