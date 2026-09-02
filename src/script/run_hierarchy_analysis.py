"""Hierarchy analysis over already-evaluated runs (no re-inference).

Computes the hierarchy violation rate from each run's predictions CSV and
tabulates per-class AUROC / AP across runs from each eval_metrics JSON, so
flat vs hierarchical (and lambda sweeps) can be compared directly.

    uv run python src/script/run_hierarchy_analysis.py --split test \
        result/<flat_run> result/<hier_run> ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.data.label_space import CANONICAL_LABELS
from src.evaluate.metrics import hierarchy_violation_rate


def _probs_matrix(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path)
    return np.stack([df[f"prob_{c}"].to_numpy() for c in CANONICAL_LABELS], axis=1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", nargs="+", help="run directories to compare")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--threshold", type=float, default=0.5)
    return p.parse_args()


def main():
    args = parse_args()
    runs = []
    for d in args.run_dirs:
        d = Path(d)
        js = json.loads((d / f"eval_metrics_{args.split}.json").read_text(encoding="utf-8"))
        probs = _probs_matrix(d / "predictions" / f"{args.split}.csv")
        hcv = hierarchy_violation_rate(probs, CANONICAL_LABELS, args.threshold)
        runs.append({"name": d.name, "summary": js["summary"], "report": js["report"], "hcv": hcv})

    print(f"\n=== condition-level [{args.split}] ===")
    print(f"{'run':<52}{'AUROC':>8}{'mAP':>8}{'AURC':>8}{'ECE':>8}{'HCV%':>8}")
    for r in runs:
        s = r["summary"]
        print(f"{r['name'][:51]:<52}{s['auroc_macro']:>8.4f}{s['map_macro']:>8.4f}"
              f"{s['aurc_macro']:>8.4f}{s['ece']:>8.4f}{r['hcv']['rate'] * 100:>8.3f}")

    print(f"\n=== per-class AUROC (labels with signal) ===")
    labels = [c for c in CANONICAL_LABELS
              if not np.isnan(runs[0]["report"]["auroc"]["per_class"].get(c, float("nan")))]
    print(f"{'label':<28}" + "".join(f"{r['name'].split('__')[-1][:11]:>12}" for r in runs))
    for c in labels:
        row = "".join(f"{r['report']['auroc']['per_class'].get(c, float('nan')):>12.4f}" for r in runs)
        print(f"{c:<28}{row}")

    print(f"\n=== per-edge violation rate (fraction of samples) ===")
    edges = list(runs[0]["hcv"]["per_edge"].keys())
    print(f"{'edge':<44}" + "".join(f"{r['name'].split('__')[-1][:11]:>12}" for r in runs))
    for e in edges:
        row = "".join(f"{r['hcv']['per_edge'].get(e, float('nan')):>12.4f}" for r in runs)
        print(f"{e:<44}{row}")


if __name__ == "__main__":
    main()
