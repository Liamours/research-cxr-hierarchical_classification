"""Full metric report over already-saved predictions (no GPU re-inference).

Reads each run's predictions/<split>.csv and recomputes the complete metric
family via evaluate_predictions: AUROC, mAP, AURC, ECE, hierarchy violation,
and the threshold family (precision, recall/sensitivity, specificity, accuracy,
F1, balanced accuracy, MCC; per-class + macro/micro/weighted + subset accuracy).
Writes eval_metrics_full_<split>.json into each run dir.

    uv run python src/script/run_test_metrics.py --split test result/<run> [result/<run> ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch

from src.data.label_space import CANONICAL_LABELS
from src.evaluate.evaluator import evaluate_predictions, _summary

METRICS = ["auroc", "f1", "ece", "map", "aurc", "hcv", "clf"]


def _load(csv_path: Path):
    df = pd.read_csv(csv_path)
    probs = torch.tensor(np.stack([df[f"prob_{c}"].to_numpy() for c in CANONICAL_LABELS], axis=1))
    targets = torch.tensor(np.stack([df[f"label_{c}"].to_numpy() for c in CANONICAL_LABELS], axis=1))
    mask = torch.tensor(np.stack([df[f"mask_{c}"].to_numpy() for c in CANONICAL_LABELS], axis=1))
    return probs.float(), targets.float(), mask.float()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--threshold", type=float, default=0.5)
    return p.parse_args()


def main():
    args = parse_args()
    rows = []
    for d in args.run_dirs:
        d = Path(d)
        probs, targets, mask = _load(d / "predictions" / f"{args.split}.csv")
        report = evaluate_predictions(probs, targets, mask, conditions=CANONICAL_LABELS,
                                      threshold=args.threshold, metrics=METRICS)
        summary = _summary(report)
        (d / f"eval_metrics_full_{args.split}.json").write_text(
            json.dumps({"split": args.split, "summary": summary, "report": report},
                       indent=2, default=str), encoding="utf-8")
        rows.append((d.name, summary))

    cols = ["auroc_macro", "auroc_micro", "map_macro", "aurc_macro", "ece",
            "f1_macro", "f1_micro", "precision_macro", "recall_macro",
            "specificity_macro", "accuracy_macro", "balanced_accuracy_macro",
            "mcc_macro", "subset_accuracy", "hcv_rate"]
    print(f"\n=== full metrics [{args.split}] ===")
    hdr = f"{'metric':<26}" + "".join(f"{n.split('__')[-1][:13]:>15}" for n, _ in rows)
    print(hdr)
    print("-" * len(hdr))
    for c in cols:
        line = f"{c:<26}"
        for _, s in rows:
            v = s.get(c, float("nan"))
            line += f"{v:>15.4f}" if isinstance(v, float) and v == v else f"{'n/a':>15}"
        print(line)
    print(f"\nwrote eval_metrics_full_{args.split}.json into each run dir")


if __name__ == "__main__":
    main()
