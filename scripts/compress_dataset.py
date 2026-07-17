#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

from tqdm import tqdm

DATASET_ROOT = Path(r"E:\research-cxr\dataset")
DATASETS = ["chexpert", "covidx-cxr4", "nih-cxr14", "tbx11k", "vindr-cxr", "vindr-pcxr"]
DEFLATE_EXTS = frozenset({".csv", ".json", ".txt"})
WINRAR = Path(r"C:\Program Files\WinRAR\Rar.exe")


def _label_sources() -> list[Path]:
    candidates = [DATASET_ROOT / "combined"] + [
        DATASET_ROOT / d / "preprocessed" / "labels" for d in DATASETS
    ]
    return [p for p in candidates if p.exists()]


def archive_labels(out: Path) -> None:
    sources = _label_sources()
    if not sources:
        sys.exit("No label sources found.")
    files = [
        f
        for src in sources
        for f in sorted(src.rglob("*") if src.is_dir() else [src])
        if f.is_file()
    ]
    total_raw = total_cmp = 0
    with zipfile.ZipFile(out, "w", allowZip64=True) as zf:
        for f in tqdm(files, unit="file", dynamic_ncols=True):
            arcname = f.relative_to(DATASET_ROOT.parent).as_posix()
            zf.write(f, arcname, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
            info = zf.getinfo(arcname)
            total_raw += info.file_size
            total_cmp += info.compress_size
            pct = 100 * (1 - info.compress_size / max(info.file_size, 1))
            tqdm.write(f"  {f.name}  {info.file_size/1e6:.1f} MB → {info.compress_size/1e6:.1f} MB  ({pct:.0f}% saved)")

    with zipfile.ZipFile(out) as zf:
        bad = zf.testzip()
    if bad:
        raise RuntimeError(f"Corrupt entry: {bad}")

    pct = 100 * (1 - total_cmp / max(total_raw, 1))
    print(f"\nDone.  {total_raw/1e6:.1f} MB → {total_cmp/1e6:.1f} MB  ({pct:.1f}% smaller)")
    print(f"Archive: {out.stat().st_size/1e6:.1f} MB  |  Integrity: OK")


def archive_full_winrar(out: Path, threads: int) -> None:
    if not WINRAR.exists():
        sys.exit(f"WinRAR not found at {WINRAR}")

    total = sum(
        1 for d in [DATASET_ROOT] for f in d.rglob("*") if f.is_file()
    )
    print(f"Total files: {total:,}")

    cmd = [
        str(WINRAR), "a", "-r", "-ep1", "-m0", f"-mt{threads}",
        str(out), str(DATASET_ROOT) + "\\",
    ]

    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace") as proc:
        with tqdm(total=total, unit="file", dynamic_ncols=True) as pbar:
            for line in proc.stdout:
                line = line.strip()
                if line.startswith("Adding") or line.startswith("Creating"):
                    pbar.update(1)

    if proc.returncode not in (0, 1):
        raise RuntimeError(f"WinRAR exited with code {proc.returncode}")
    print(f"\nDone. Archive: {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATASET_ROOT.parent / "dataset_archive"))
    ap.add_argument("--full", action="store_true", help="Full dataset via WinRAR (images + labels, ~15 GB)")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    if args.full:
        out = Path(args.out).with_suffix(".rar")
        print(f"Archiving (full, WinRAR) → {out}\n")
        archive_full_winrar(out, args.threads)
    else:
        out = Path(args.out).with_suffix(".zip")
        print(f"Archiving (labels only) → {out}\n")
        archive_labels(out)


if __name__ == "__main__":
    main()
