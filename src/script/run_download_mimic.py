"""Download missing MIMIC-CXR-JPG frontal images from PhysioNet v2.1.0.

Strategy:
  - Persistent cache (.mimic_cache.json) makes disk-check instant on restart.
  - First run scans disk and builds cache; every subsequent run loads it in <1s.
  - wget handles auth (no session expiry); Python handles path logic + preprocessing.
  - ThreadPoolExecutor downloads in parallel; cache updates are thread-safe.
  - Cache flushed every 100 downloads and on exit — crash-safe (files on disk are ground truth).
  - --rebuild-cache forces a full disk rescan (use if files were added/removed externally).

Usage:
    # spot-check 10 images
    uv run python src/script/run_download_mimic.py `
        --archive-root E:/research-cxr/dataset/mimic-cxr/archive-preprocessed `
        --label-dir E:/research-cxr/dataset/mimic-cxr/label `
        --username mrda --limit 10

    # full run
    uv run python src/script/run_download_mimic.py `
        --archive-root E:/research-cxr/dataset/mimic-cxr/archive-preprocessed `
        --label-dir E:/research-cxr/dataset/mimic-cxr/label `
        --username mrda --workers 1

    # force rebuild cache (after external file changes)
    uv run python src/script/run_download_mimic.py `
        --archive-root E:/research-cxr/dataset/mimic-cxr/archive-preprocessed `
        --label-dir E:/research-cxr/dataset/mimic-cxr/label `
        --username mrda --rebuild-cache
"""
from __future__ import annotations

import atexit
import argparse
import getpass
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from PIL import Image
from tqdm import tqdm

BASE_URL = "https://physionet.org/files/mimic-cxr-jpg/2.1.0/files"
VP_RANK = {"PA": 0, "AP": 1}
WGET_TIMEOUT = 300
CACHE_FLUSH_EVERY = 100


class DownloadCache:
    """Thread-safe persistent set of downloaded study_ids."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._present: set[int] = set()
        self._dirty_count = 0
        self._load()
        atexit.register(self._flush)

    def _load(self) -> None:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text())
                self._present = set(data.get("present", []))
            except Exception:
                self._present = set()

    def _flush(self) -> None:
        with self._lock:
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"present": list(self._present), "version": 1}))
            tmp.replace(self._path)
            self._dirty_count = 0

    def is_present(self, study_id: int) -> bool:
        return study_id in self._present

    def mark_present(self, study_id: int) -> None:
        with self._lock:
            self._present.add(study_id)
            self._dirty_count += 1
        if self._dirty_count >= CACHE_FLUSH_EVERY:
            self._flush()

    def __len__(self) -> int:
        return len(self._present)

    def rebuild(self, archive_root: Path) -> None:
        """
        Single sequential rglob scan instead of N random stat() calls.
        On external HDD: 2-5 min vs 30 min for 224k individual Path.exists().
        Extracts study_id from path: archive_root/{split}/p{subj}/s{study}/view1_frontal.png
        """
        print("Building cache via sequential rglob scan (first run only — restarts will be instant)...")
        present: set[int] = set()
        for png in tqdm(
            archive_root.rglob("view1_frontal.png"),
            desc="rglob scan",
            unit="file",
        ):
            study_dir = png.parent.name  # "s{study_id}"
            if study_dir.startswith("s"):
                try:
                    present.add(int(study_dir[1:]))
                except ValueError:
                    pass
        with self._lock:
            self._present = present
        self._flush()
        print(f"Cache built: {len(present)} studies present on disk.")


def build_url(subject_id: int, study_id: int, dicom_id: str) -> str:
    prefix = str(subject_id)[:2]
    return f"{BASE_URL}/p{prefix}/p{subject_id}/s{study_id}/{dicom_id}.jpg"


def load_and_process(path: Path, image_size: int = 224) -> Image.Image:
    img = Image.open(path).convert("L")
    if img.size == (image_size, image_size):
        return img
    w, h = img.size
    scale = 256 / min(w, h)
    img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)
    w, h = img.size
    left, top = (w - image_size) // 2, (h - image_size) // 2
    return img.crop((left, top, left + image_size, top + image_size))


def download_one(
    url: str,
    out_png: Path,
    dicom_id: str,
    username: str,
    password: str,
    cache: DownloadCache,
    study_id: int,
) -> str:
    """Download one JPG via wget, preprocess to PNG, update cache. Returns 'ok' or error."""
    if out_png.exists():
        cache.mark_present(study_id)
        return "skip"

    out_png.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_png.parent / f".tmp_{dicom_id}.jpg"

    try:
        result = subprocess.run(
            [
                "wget", "-q", "-c",
                f"--user={username}",
                f"--password={password}",
                f"--timeout={WGET_TIMEOUT}",
                "-O", str(tmp),
                url,
            ],
            capture_output=True,
            timeout=WGET_TIMEOUT + 10,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace").strip()
            return f"wget rc={result.returncode}: {stderr[-200:]}"

        img = load_and_process(tmp)
        img.save(out_png)
        cache.mark_present(study_id)
        return "ok"
    except Exception as e:
        return f"exception: {e}"
    finally:
        if tmp.exists():
            tmp.unlink()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive-root", required=True)
    ap.add_argument("--label-dir", required=True)
    ap.add_argument("--username", required=True)
    ap.add_argument("--password", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--rebuild-cache", action="store_true",
                    help="Force full disk rescan and rebuild cache")
    args = ap.parse_args()

    password = args.password or getpass.getpass(f"PhysioNet password for {args.username}: ")

    archive_root = Path(args.archive_root)
    label_dir = Path(args.label_dir)
    cache_path = archive_root / ".mimic_cache.json"

    # load metadata
    splits_df = pd.read_csv(label_dir / "mimic-cxr-2.0.0-split.csv")
    study_split: dict[int, str] = splits_df.groupby("study_id")["split"].first().to_dict()

    meta_df = pd.read_csv(label_dir / "mimic-cxr-2.0.0-metadata.csv")
    frontal = meta_df[meta_df["ViewPosition"].isin(["PA", "AP"])].copy()
    frontal["vp_rank"] = frontal["ViewPosition"].map(
        lambda v: VP_RANK.get(str(v).upper(), 9)
    )
    best_per_study = (
        frontal.sort_values("vp_rank")
        .groupby("study_id")
        .first()
        .reset_index()[["study_id", "subject_id", "dicom_id"]]
    )

    # build full study list with expected paths
    all_studies = []
    for _, row in best_per_study.iterrows():
        sid = int(row["study_id"])
        subj = int(row["subject_id"])
        split = study_split.get(sid, "train").replace("validate", "valid")
        out_png = archive_root / split / f"p{subj}" / f"s{sid}" / "view1_frontal.png"
        all_studies.append((sid, subj, str(row["dicom_id"]), out_png))

    # load or build cache
    cache = DownloadCache(cache_path)

    if args.rebuild_cache or not cache_path.exists():
        cache.rebuild(archive_root)
    else:
        print(f"Cache loaded: {len(cache)} studies known present (instant start)")

    # filter to missing
    missing = [
        (sid, subj, dicom_id, out_png)
        for sid, subj, dicom_id, out_png in all_studies
        if not cache.is_present(sid)
    ]

    if args.limit:
        missing = missing[: args.limit]

    print(f"Missing: {len(missing)}  |  workers: {args.workers}")

    if not missing:
        print("All studies present. Done.")
        return

    downloaded = skipped = errors = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                download_one,
                build_url(subj, sid, dicom_id),
                out_png,
                dicom_id,
                args.username,
                password,
                cache,
                sid,
            ): sid
            for sid, subj, dicom_id, out_png in missing
        }

        with tqdm(total=len(futures), desc="download+preprocess") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result == "ok":
                    downloaded += 1
                elif result == "skip":
                    skipped += 1
                else:
                    sid = futures[future]
                    tqdm.write(f"ERROR study={sid}: {result}")
                    errors += 1
                pbar.update(1)
                pbar.set_postfix(ok=downloaded, skip=skipped, err=errors)

    print(f"\nDone.  downloaded={downloaded}  skipped={skipped}  errors={errors}")
    print(f"Cache saved: {len(cache)} studies present")


if __name__ == "__main__":
    main()
