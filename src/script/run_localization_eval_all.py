"""Grad-CAM localization, all 9 conditions from the 2026-08-30 batch, not
just the original flat/hierarchical pair (run_localization_eval.py, kept
as-is for that one). Same corrected boxes, same method; reuses that
script's box-loading and per-condition evaluation directly rather than
duplicating them.

    uv run python src/script/run_localization_eval_all.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from src.script._conditions import _RUNS, available, run_dir
from src.script.run_localization_eval import evaluate_condition, load_test_boxes

OUT_DIR = Path(r"C:\rifqi\research-cxr-hierarchical_classification\analyses\localization_eval_all-260901")


def main() -> None:
    device = torch.device("cpu")  # same reasoning as run_localization_eval.py
    test_boxes = load_test_boxes()
    n_images = test_boxes["image_id"].nunique()
    print(f"{len(test_boxes)} usable boxes across {n_images} test images")

    ready = available()  # only conditions with saved predictions, i.e. training done
    results = {}
    for name in _RUNS:
        if name not in ready:
            print(f"skipping {name}: training not finished")
            continue
        d = run_dir(name)
        print(f"\n--- {name} ({d.name}) ---")
        results[name] = evaluate_condition(name, str(d), test_boxes, device)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # per-condition "records" (added for run_localization_eval.py's Fig. 5
    # bucketing) are a side channel, not needed for this summary file.
    results_summary = {name: {"per_finding": r["per_finding"], "macro": r["macro"]}
                        for name, r in results.items()}
    with open(OUT_DIR / "localization_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"n_boxes": len(test_boxes), "n_images": int(n_images), "results": results_summary},
                   f, indent=2)

    print(f"\n{'Condition':<32}{'n boxes':>9}{'IoU':>8}{'IoBB':>8}{'Pointing':>10}{'AP@0.1':>9}"
          f"{'AP@0.25':>9}{'AP@0.5':>9}")
    print("-" * 96)
    for name, r in results.items():
        m = r["macro"]
        print(f"{name:<32}{m['n_boxes']:>9}{m['iou']:>8.4f}{m['iobb']:>8.4f}{100*m['pointing']:>9.1f}%"
              f"{m['ap@0.1']:>9.4f}{m['ap@0.25']:>9.4f}{m['ap@0.5']:>9.4f}")

    print(f"\nWritten: {OUT_DIR / 'localization_metrics.json'}")


if __name__ == "__main__":
    main()
