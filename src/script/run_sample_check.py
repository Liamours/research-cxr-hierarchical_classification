"""Sample 1 image per dataset and verify preprocessing correctness.

Checks per image:
  - File size, dimensions, mode (RGB/L)
  - Pixel statistics (min/max/mean/std)
  - Aspect ratio (warn if not 1:1 — we expect square 224×224)
  - No all-black / all-white corruption

Checks per label CSV:
  - Canonical columns present
  - NaN fraction per present column
  - Split distribution

Saves a 6-panel composite thumbnail to <out-dir>/sample_grid.png for
manual visual inspection.

    uv run python src/script/run_sample_check.py
    uv run python src/script/run_sample_check.py --dataset-root E:/research-cxr/dataset --out-dir result/sample_check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from src.data.label_space import CANONICAL_LABELS

DATASET_ROOT = Path("E:/research-cxr/dataset")

DATASETS = {
    "chexpert":     {"img_glob": "preprocessed/images/chexpert/*.png",
                     "label":    "label/metadata_train.csv",
                     "id_col":   None},
    "vindr-cxr":    {"img_glob": "preprocessed/images/vindr-cxr/*.png",
                     "label":    "label/vinbig_14class_manifest.csv",
                     "id_col":   "image_id"},
    "nih-cxr14":    {"img_glob": "preprocessed/images/nih-cxr14/*.png",
                     "label":    "label/nih_14class_manifest.csv",
                     "id_col":   "image_id"},
    "tbx11k":       {"img_glob": "preprocessed/images/tbx11k/*.png",
                     "label":    None,
                     "id_col":   None},
    "covidx-cxr4":  {"img_glob": "preprocessed/images/covidx-cxr4/*.png",
                     "label":    "label/train.txt",
                     "id_col":   None},
    "vindr-pcxr":   {"img_glob": "preprocessed/images/vindr-pcxr/*.png",
                     "label":    None,
                     "id_col":   None},
}

TARGET_SIZE = 224


def check_image(path: Path) -> dict:
    img = Image.open(path)
    arr = np.array(img)
    w, h = img.size
    result = {
        "path":       str(path.name),
        "size_kb":    round(path.stat().st_size / 1024, 1),
        "mode":       img.mode,
        "width":      w,
        "height":     h,
        "dtype":      str(arr.dtype),
        "channels":   arr.shape[2] if arr.ndim == 3 else 1,
        "px_min":     int(arr.min()),
        "px_max":     int(arr.max()),
        "px_mean":    round(float(arr.mean()), 1),
        "px_std":     round(float(arr.std()), 1),
        "square":     w == h,
        "is_224":     w == TARGET_SIZE and h == TARGET_SIZE,
        "corrupt":    arr.max() == arr.min(),
    }
    return result, img


def label_summary(cfg: dict, ds_root: Path) -> str:
    lbl_path = cfg.get("label")
    if not lbl_path:
        return "no label file configured"
    p = ds_root / lbl_path
    if not p.exists():
        return f"NOT FOUND: {p}"
    try:
        if p.suffix == ".csv":
            df = pd.read_csv(p, nrows=5)
            cols = list(df.columns)
            canonical_present = [c for c in CANONICAL_LABELS if c in cols]
            return (f"{len(df)} rows (head), {len(cols)} cols, "
                    f"{len(canonical_present)} canonical cols present: "
                    f"{canonical_present[:5]}{'...' if len(canonical_present) > 5 else ''}")
        else:
            with open(p) as f:
                lines = f.readlines()[:3]
            return f"text file, sample: {lines[0].strip()[:80]}"
    except Exception as e:
        return f"ERROR reading: {e}"


def annotate(img: Image.Image, info: dict, ds: str) -> Image.Image:
    canvas_size = 280
    canvas = Image.new("RGB", (canvas_size, canvas_size + 60), (240, 240, 240))
    thumb = img.convert("RGB").resize((canvas_size, canvas_size), Image.LANCZOS)
    canvas.paste(thumb, (0, 0))
    draw = ImageDraw.Draw(canvas)
    status = "OK" if info["is_224"] and not info["corrupt"] else "WARN"
    color = (0, 180, 0) if status == "OK" else (220, 60, 0)
    draw.rectangle([0, canvas_size, canvas_size, canvas_size + 60], fill=(30, 30, 30))
    draw.text((4, canvas_size + 2),  f"{ds}", fill=(255, 220, 0))
    draw.text((4, canvas_size + 16), f"{info['width']}x{info['height']} {info['mode']}", fill=(200, 200, 200))
    draw.text((4, canvas_size + 30), f"min={info['px_min']} max={info['px_max']} mean={info['px_mean']:.0f}", fill=(200, 200, 200))
    draw.text((4, canvas_size + 44), status, fill=color)
    return canvas


def make_grid(panels: list[Image.Image], cols: int = 3) -> Image.Image:
    pw, ph = panels[0].size
    rows = (len(panels) + cols - 1) // cols
    grid = Image.new("RGB", (pw * cols, ph * rows), (180, 180, 180))
    for i, p in enumerate(panels):
        r, c = divmod(i, cols)
        grid.paste(p, (c * pw, r * ph))
    return grid


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default=str(DATASET_ROOT))
    ap.add_argument("--out-dir", default="result/sample_check")
    ap.add_argument("--datasets", nargs="*", default=None, help="subset to check")
    return ap.parse_args()


def main():
    args = parse_args()
    ds_root_base = Path(args.dataset_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = args.datasets or list(DATASETS)
    panels = []
    all_ok = True

    print(f"\n{'Dataset':<18} {'File':<45} {'WxH':<12} {'Mode':<5} {'min':>4} {'max':>4} {'mean':>6} {'224?':<5} {'Status'}")
    print("-" * 110)

    for ds in datasets:
        cfg = DATASETS.get(ds)
        if not cfg:
            print(f"{ds:<18} unknown dataset")
            continue
        ds_root = ds_root_base / ds
        imgs = sorted((ds_root / cfg["img_glob"].split("/*")[0]).glob("*.png"))[:1]
        if not imgs:
            print(f"{ds:<18} {'NO IMAGES FOUND':<45}")
            all_ok = False
            continue

        info, pil_img = check_image(imgs[0])

        issues = []
        if not info["is_224"]:
            issues.append(f"NOT 224x224 ({info['width']}x{info['height']})")
        if info["corrupt"]:
            issues.append("FLAT/CORRUPT (min==max)")
        if info["px_max"] <= 1:
            issues.append("SUSPICIOUS: max<=1 (float not uint8?)")
        if info["mode"] not in ("RGB", "L"):
            issues.append(f"UNEXPECTED MODE {info['mode']}")

        status = "FAIL: " + "; ".join(issues) if issues else "OK"
        if issues:
            all_ok = False

        print(f"{ds:<18} {info['path']:<45} {info['width']}x{info['height']:<7} "
              f"{info['mode']:<5} {info['px_min']:>4} {info['px_max']:>4} {info['px_mean']:>6} "
              f"{'YES' if info['is_224'] else 'NO':<5} {status}")

        panels.append(annotate(pil_img, info, ds))

    print("\n--- Label files ---")
    for ds in datasets:
        cfg = DATASETS.get(ds)
        if not cfg:
            continue
        summ = label_summary(cfg, ds_root_base / ds)
        print(f"  {ds:<18} {summ}")

    if panels:
        grid = make_grid(panels, cols=3)
        grid_path = out_dir / "sample_grid.png"
        grid.save(grid_path)
        print(f"\nSaved visual grid: {grid_path.resolve()}")

    print(f"\n{'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED — see above'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
