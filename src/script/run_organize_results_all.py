"""Copy every 2026-08-30-batch condition's weights/predictions/metrics into
the project's established layout (weights/, metrics/, inference_results/),
the same pattern run_verify_against_manuscript.py already used for the
seed42 verification pair -- so every trained condition is discoverable the
same way, not just sitting under result/<dated-run>/.

    uv run python src/script/run_organize_results_all.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.script._conditions import _RUNS, available, run_dir

PROJECT_ROOT = Path(r"C:\rifqi\research-cxr-hierarchical_classification")

# name -> label suffix, matching the existing "<backbone>-<condition>-<tag>-<date>" style.
_LABELS = {
    "flat_seed42":         None,  # already copied by run_verify_against_manuscript.py
    "hierarchical_seed42": None,
    "flat_seed43":         "densenet121_xrv-flat-seed43-260830",
    "hierarchical_seed43": "densenet121_xrv-hierarchical-seed43-260830",
    "flat_seed44":         "densenet121_xrv-flat-seed44-260830",
    "hierarchical_seed44": "densenet121_xrv-hierarchical-seed44-260830",
    "resnet50_flat_seed42":         "resnet50_xrv-flat-seed42-260830",
    "resnet50_hierarchical_seed42": "resnet50_xrv-hierarchical-seed42-260830",
    "hierarchical_soft_seed42":     "densenet121_xrv-hierarchical_soft-seed42-260830",
}


def organize(name: str, label: str) -> None:
    src = run_dir(name)
    weights_dst = PROJECT_ROOT / "weights" / f"classification-{label}"
    metrics_dst = PROJECT_ROOT / "metrics" / f"classification-{label}"
    infer_dst = PROJECT_ROOT / "inference_results" / f"predictions-{label}"
    for d in (weights_dst, metrics_dst, infer_dst):
        d.mkdir(parents=True, exist_ok=True)

    for fname in ("config.yaml", "run.log", "train_log.csv", "model_summary.txt", "events.jsonl", "metrics.json"):
        s = src / fname
        if s.exists():
            shutil.copy2(s, weights_dst / fname)
    ckpt_src = src / "checkpoints"
    if ckpt_src.exists():
        shutil.copytree(ckpt_src, weights_dst / "checkpoints", dirs_exist_ok=True)

    for split in ("val", "test"):
        m = src / f"eval_metrics_{split}.json"
        if m.exists():
            shutil.copy2(m, metrics_dst / m.name)
        p = src / "predictions" / f"{split}.csv"
        if p.exists():
            shutil.copy2(p, infer_dst / p.name)

    print(f"{name} -> {label}")


def main() -> None:
    ready = available()
    for name in _RUNS:
        label = _LABELS[name]
        if label is None:
            print(f"{name}: already organized (verify-260830)")
            continue
        if name not in ready:
            print(f"{name}: skipping, training not finished")
            continue
        organize(name, label)
    print("\nDone. See weights/, metrics/, inference_results/ for the classification-<label> folders.")


if __name__ == "__main__":
    main()
