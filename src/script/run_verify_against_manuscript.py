"""Compare a fresh run's eval_metrics_test.json against the manuscript's
reported Table II numbers, and copy its predictions/metrics into the
established weights/ + metrics/ + inference_results/ layout.

    uv run python src/script/run_verify_against_manuscript.py --run result/20260828_densenet121_xrv__flat --label densenet121_xrv-flat-verify-260828
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(r"C:\rifqi\research-cxr-hierarchical_classification")

# main.tex Table II, tab:overall (flat column) -- the manuscript's reported
# numbers. f1_macro corrected 2026-08-29 from 0.1381/0.1293: those were a
# 27-label macro that silently included Lung_Cancer/Pleural_Empyema at
# F1=0.0 each, contradicting the manuscript's own stated 25-label scope
# (main.tex lines 338-341); see src/evaluate/metrics.py's per_class_f1 fix
# and context/task/draft.md. All other rows unaffected by that fix.
MANUSCRIPT_TABLE_II = {
    "flat": {
        "auroc_macro": 0.8529, "auroc_micro": 0.9262, "map_macro": 0.2614,
        "aurc_macro": 0.0120, "aurc_flat": 0.0079, "ece": 0.0059,
        "hcv_rate": 0.3695, "f1_macro": 0.1491, "f1_micro": 0.4898,
        "mcc_macro": 0.3116, "balanced_accuracy_macro": 0.5557,
    },
    "hierarchical": {
        "auroc_macro": 0.8528, "auroc_micro": 0.9241, "map_macro": 0.2610,
        "aurc_macro": 0.0210, "aurc_flat": 0.0084, "ece": 0.0097,
        "hcv_rate": 0.3382, "f1_macro": 0.1397, "f1_micro": 0.4327,
        "mcc_macro": 0.2839, "balanced_accuracy_macro": 0.5498,
    },
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path, help="run dir, e.g. result/20260828_densenet121_xrv__flat")
    ap.add_argument("--condition", required=True, choices=["flat", "hierarchical"])
    ap.add_argument("--label", required=True, help="dest folder suffix, e.g. densenet121_xrv-flat-verify-260828")
    args = ap.parse_args()

    run_dir = args.run
    metrics_json = run_dir / "eval_metrics_test.json"
    if not metrics_json.exists():
        print(f"MISSING: {metrics_json} -- auto-eval did not complete (matches the original flat run's "
              f"known MemoryError history?). Run manually:\n"
              f"  uv run --extra cu128 python src/script/run_evaluate.py --config configs/densenet121_xrv__{args.condition}.yaml "
              f"--checkpoint {run_dir}/checkpoints/best_val_auroc_macro.pt --split test")
        return

    js = json.loads(metrics_json.read_text(encoding="utf-8"))
    summary = js["summary"]
    ref = MANUSCRIPT_TABLE_II[args.condition]

    print(f"\n=== {args.condition}: fresh run vs. manuscript Table II ===")
    print(f"{'metric':<28}{'manuscript':>12}{'this run':>14}{'delta':>10}")
    print("-" * 64)
    for k, want in ref.items():
        got = summary.get(k)
        if got is None:
            print(f"{k:<28}{'--':>12}{'MISSING':>14}")
            continue
        delta = got - want
        flag = "" if abs(delta) < 0.01 else "  <-- check"
        print(f"{k:<28}{want:>12.4f}{got:>14.4f}{delta:>+10.4f}{flag}")

    # copy into the established layout
    weights_dst = PROJECT_ROOT / "weights" / f"classification-{args.label}"
    metrics_dst = PROJECT_ROOT / "metrics" / f"classification-{args.label}"
    infer_dst = PROJECT_ROOT / "inference_results" / f"predictions-{args.label}"
    for d in (weights_dst, metrics_dst, infer_dst):
        d.mkdir(parents=True, exist_ok=True)

    for name in ("config.yaml", "run.log", "train_log.csv", "model_summary.txt", "events.jsonl", "metrics.json"):
        src = run_dir / name
        if src.exists():
            shutil.copy2(src, weights_dst / name)
    ckpt_src = run_dir / "checkpoints"
    if ckpt_src.exists():
        shutil.copytree(ckpt_src, weights_dst / "checkpoints", dirs_exist_ok=True)

    for split in ("val", "test"):
        m = run_dir / f"eval_metrics_{split}.json"
        if m.exists():
            shutil.copy2(m, metrics_dst / m.name)
        p = run_dir / "predictions" / f"{split}.csv"
        if p.exists():
            shutil.copy2(p, infer_dst / p.name)

    print(f"\nCopied into:\n  {weights_dst}\n  {metrics_dst}\n  {infer_dst}")


if __name__ == "__main__":
    main()
