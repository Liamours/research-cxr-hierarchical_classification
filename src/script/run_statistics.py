"""Post-hoc statistical analysis over grid run results.

Reads predictions CSVs from run directories, computes bootstrap 95% CIs for
all headline metrics, and optionally runs pairwise AURC and McNemar comparisons.

Usage:
    # CIs for a single condition
    uv run python src/script/run_statistics.py --run-dir result/20260622_densenet121_xrv__seg-off__uq-off --split val

    # Pairwise comparison: fallback (condition A) vs flat abstention (condition B)
    uv run python src/script/run_statistics.py \\
        --run-dir  result/20260622_densenet121_xrv__seg-off__uq-mc_dropout \\
        --compare  result/20260622_densenet121_xrv__seg-off__uq-off \\
        --split val

    # Full grid: CIs for all 8 conditions + primary hypothesis test
    uv run python src/script/run_statistics.py --grid configs/grid_base.yaml --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.label_space import CANONICAL_LABELS
from src.evaluate.statistics import (
    bootstrap_metric_ci,
    compare_aurc,
    load_predictions,
    mcnemar_comparison,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, help="single run directory")
    p.add_argument("--compare", type=Path, help="second run dir for pairwise comparison")
    p.add_argument("--grid", type=Path, help="grid_base.yaml: run analysis for all 8 conditions")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--out", type=Path, help="write JSON to this path (default: print)")
    return p.parse_args()


def ci_for_run(run_dir: Path, split: str, n_boot: int = 1000) -> dict:
    pred_csv = run_dir / "predictions" / f"{split}.csv"
    if not pred_csv.exists():
        return {"error": f"predictions not found: {pred_csv}"}
    probs, targets, mask, _ = load_predictions(pred_csv)
    return bootstrap_metric_ci(probs, targets, mask, CANONICAL_LABELS, n_boot=n_boot)


def main():
    args = parse_args()
    result: dict = {}

    if args.grid:
        from src.config.experiment_config import ExperimentConfig
        from src.experiment.grid import build_grid
        base = ExperimentConfig.from_yaml(args.grid)
        configs = build_grid(base)
        result["conditions"] = {}
        for cfg in configs:
            name = cfg.experiment.name
            print(f"  bootstrap CI: {name}")
            result["conditions"][name] = ci_for_run(cfg.run_dir(), args.split, args.n_boot)

        # Primary hypothesis: hierarchical+mc_dropout vs flat+none (same seg state)
        def _get(seg, ls, uq):
            for c in configs:
                if c.seg.enabled == seg and c.label.label_structure == ls and c.uq.method == uq:
                    return c
            return None

        for seg in (False, True):
            seg_tag = "seg-on" if seg else "seg-off"
            ca = _get(seg, "hierarchical", "mc_dropout")
            cb = _get(seg, "flat", "none")
            if ca and cb:
                pa_csv = ca.run_dir() / "predictions" / f"{args.split}.csv"
                pb_csv = cb.run_dir() / "predictions" / f"{args.split}.csv"
                if pa_csv.exists() and pb_csv.exists():
                    pa, ta, ma, _ = load_predictions(pa_csv)
                    pb, tb, mb, _ = load_predictions(pb_csv)
                    print(f"  AURC comparison ({seg_tag}): {ca.experiment.name} vs {cb.experiment.name}")
                    result[f"aurc_comparison_{seg_tag}"] = compare_aurc(
                        pa, ta, ma, pb, tb, mb,
                        n_boot=args.n_boot,
                        label=(ca.experiment.name, cb.experiment.name),
                    )
                    if pa.shape[0] == pb.shape[0]:
                        result[f"mcnemar_{seg_tag}"] = mcnemar_comparison(
                            pa, ta, ma, pb, tb, mb,
                            label=(ca.experiment.name, cb.experiment.name),
                        )

    elif args.run_dir:
        result["ci"] = ci_for_run(args.run_dir, args.split, args.n_boot)
        if args.compare:
            pred_a = args.run_dir / "predictions" / f"{args.split}.csv"
            pred_b = args.compare / "predictions" / f"{args.split}.csv"
            if pred_a.exists() and pred_b.exists():
                pa, ta, ma, _ = load_predictions(pred_a)
                pb, tb, mb, _ = load_predictions(pred_b)
                result["aurc_comparison"] = compare_aurc(
                    pa, ta, ma, pb, tb, mb,
                    n_boot=args.n_boot,
                    label=(args.run_dir.name, args.compare.name),
                )
                if pa.shape[0] == pb.shape[0]:
                    result["mcnemar"] = mcnemar_comparison(
                        pa, ta, ma, pb, tb, mb,
                        label=(args.run_dir.name, args.compare.name),
                    )

    out_str = json.dumps(result, indent=2, default=str)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(out_str, encoding="utf-8")
        print(f"Written: {args.out}")
    else:
        print(out_str)


if __name__ == "__main__":
    main()
