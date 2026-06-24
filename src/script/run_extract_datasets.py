"""Extract TBX11K and COVIDx CXR-4 zips into project-consistent directory layout.

TBX11K layout:
  TBX11K/imgs/    -> dataset/tbx11k/raw/imgs/
  TBX11K/lists/   -> dataset/tbx11k/label/
  TBX11K/annotations/ -> dataset/tbx11k/label/annotations/
  (code/, README, PDF, teaser skipped)

COVIDx CXR-4 layout:
  train/, val/, test/  -> dataset/covidx-cxr4/raw/train|val|test/
  train.txt, val.txt, test.txt -> dataset/covidx-cxr4/label/

Usage:
    uv run python src/script/run_extract_datasets.py --dataset tbx11k
    uv run python src/script/run_extract_datasets.py --dataset covidx-cxr4
    uv run python src/script/run_extract_datasets.py --dataset all
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from tqdm import tqdm


DATASET_ROOT = Path("E:/research-cxr/dataset")


def _extract_zip(
    zip_path: Path,
    mappings: list[tuple[str, Path]],
    skip_prefixes: list[str] | None = None,
) -> None:
    """Extract zip entries, remapping prefixes to target dirs.

    mappings: list of (zip_prefix, target_dir) — first match wins.
    skip_prefixes: entries starting with these are skipped entirely.
    """
    skip_prefixes = skip_prefixes or []

    with zipfile.ZipFile(zip_path) as zf:
        entries = [e for e in zf.infolist() if not e.is_dir()]
        print(f"  {len(entries)} files in {zip_path.name}")

        for entry in tqdm(entries, desc=f"extract {zip_path.stem[:20]}"):
            name = entry.filename.replace("\\", "/")

            if any(name.startswith(sp) for sp in skip_prefixes):
                continue

            target_dir: Path | None = None
            rel: str | None = None
            for prefix, dest in mappings:
                if name.startswith(prefix):
                    rel = name[len(prefix):]
                    if not rel:  # exact filename match (e.g. "train.txt")
                        rel = Path(name).name
                    target_dir = dest
                    break

            if target_dir is None or not rel:
                continue

            out = target_dir / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.exists():
                continue
            with zf.open(entry) as src, open(out, "wb") as dst:
                dst.write(src.read())


def extract_tbx11k() -> None:
    zip_path = DATASET_ROOT / "TBX11K.zip"
    base = DATASET_ROOT / "tbx11k"

    mappings = [
        ("TBX11K/imgs/",        base / "raw" / "imgs"),
        ("TBX11K/lists/",       base / "label"),
        ("TBX11K/annotations/", base / "label" / "annotations"),
    ]
    skip = ["TBX11K/code/", "TBX11K/README.md", "TBX11K/teaser.jpg", "TBX11K/TBX11K_CVPR2020.pdf"]

    print("Extracting TBX11K...")
    _extract_zip(zip_path, mappings, skip)

    # quick count
    imgs = list((base / "raw" / "imgs").rglob("*.png"))
    lbls = list((base / "label").glob("*.txt"))
    anns = list((base / "label" / "annotations").glob("*")) if (base / "label" / "annotations").exists() else []
    print(f"  images : {len(imgs)}")
    print(f"  labels : {lbls}")
    print(f"  annotations: {len(anns)} files")


def extract_covidx() -> None:
    zip_path = DATASET_ROOT / "archive (1).zip"
    base = DATASET_ROOT / "covidx-cxr4"

    # Determine if zip has top-level folder or not by peeking
    with zipfile.ZipFile(zip_path) as zf:
        sample = zf.namelist()[:5]
    print(f"  COVIDx zip sample: {sample}")

    # Try to detect if there's a top-level directory wrapper
    top_dirs = set()
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist()[:200]:
            parts = name.replace("\\", "/").split("/")
            if parts:
                top_dirs.add(parts[0])
    top_dirs = {d for d in top_dirs if d}
    print(f"  COVIDx top-level entries: {sorted(top_dirs)[:10]}")

    # Build mappings depending on whether there's a wrapper dir
    # Common Kaggle patterns: flat (train/, val/) or wrapped (archive/train/)
    image_splits = {"train", "val", "test"}
    label_files = {"train.txt", "val.txt", "test.txt"}

    def make_mappings(prefix: str) -> list[tuple[str, Path]]:
        m = []
        for split in image_splits:
            m.append((f"{prefix}{split}/", base / "raw" / split))
        for lf in label_files:
            m.append((f"{prefix}{lf}", base / "label"))
        return m

    # If only one top dir and it's not train/val/test, it's a wrapper
    non_split_tops = top_dirs - image_splits - label_files
    if non_split_tops and not (image_splits & top_dirs):
        wrapper = sorted(non_split_tops)[0] + "/"
        print(f"  Using wrapper prefix: '{wrapper}'")
        mappings = make_mappings(wrapper)
    else:
        mappings = make_mappings("")

    print("Extracting COVIDx CXR-4 (28.9 GB — may take 10-30 min)...")
    _extract_zip(zip_path, mappings)

    for split in ["train", "val", "test"]:
        d = base / "raw" / split
        if d.exists():
            n = sum(1 for _ in d.iterdir())
            print(f"  {split}: {n} files")
    for lf in label_files:
        p = base / "label" / lf
        if p.exists():
            with open(p) as f:
                lines = sum(1 for _ in f)
            print(f"  {lf}: {lines} lines")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["tbx11k", "covidx-cxr4", "all"])
    args = ap.parse_args()

    if args.dataset in ("tbx11k", "all"):
        extract_tbx11k()
    if args.dataset in ("covidx-cxr4", "all"):
        extract_covidx()


if __name__ == "__main__":
    main()
