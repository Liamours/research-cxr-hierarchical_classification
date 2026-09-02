"""Shared registry of all trained conditions from the 2026-08-30 revision
batch, for the post-training analysis scripts (tuned thresholds, source
stratification, significance tests). Not a CLI entry point itself.

Each entry's val/test CSVs come straight from `_final_eval`'s own
`predictions/{val,test}.csv` (src/evaluate/evaluator.py::save_predictions),
so this only reads already-computed predictions -- no GPU, no retraining.
"""
from __future__ import annotations

from pathlib import Path

RESULT_ROOT = Path(r"C:\rifqi\research-cxr-hierarchical_classification\repo\research-cxr-hierarchical_classification\result")

# name -> (experiment.name, backbone, label_structure, seed). Deliberately
# NOT the dated run-dir prefix (cfg.run_dir() stamps it with date.today() at
# launch time) -- a run launched after midnight lands in a different-dated
# folder than one launched before, which silently broke this exact lookup
# once already when the calendar rolled from Aug 30 to Sep 1 mid-session.
# csv_paths() below globs for the folder instead of assuming its date.
_RUNS = {
    "flat_seed42":         ("densenet121_xrv__flat__seed42", "densenet121_xrv", "flat", 42),
    "hierarchical_seed42": ("densenet121_xrv__hierarchical__seed42", "densenet121_xrv", "hierarchical", 42),
    "flat_seed43":         ("densenet121_xrv__flat__seed43", "densenet121_xrv", "flat", 43),
    "hierarchical_seed43": ("densenet121_xrv__hierarchical__seed43", "densenet121_xrv", "hierarchical", 43),
    "flat_seed44":         ("densenet121_xrv__flat__seed44", "densenet121_xrv", "flat", 44),
    "hierarchical_seed44": ("densenet121_xrv__hierarchical__seed44", "densenet121_xrv", "hierarchical", 44),
    "resnet50_flat_seed42":         ("resnet50_xrv__flat__seed42", "resnet50_xrv", "flat", 42),
    "resnet50_hierarchical_seed42": ("resnet50_xrv__hierarchical__seed42", "resnet50_xrv", "hierarchical", 42),
    "hierarchical_soft_seed42":     ("densenet121_xrv__hierarchical_soft__seed42", "densenet121_xrv", "hierarchical_soft", 42),
}

# Natural pairwise comparisons -- what actually answers a reviewer question,
# not all-vs-all. (name_a, name_b, question)
COMPARISONS = [
    ("flat_seed42", "hierarchical_seed42", "primary ablation, seed 42"),
    ("flat_seed43", "hierarchical_seed43", "primary ablation, seed 43"),
    ("flat_seed44", "hierarchical_seed44", "primary ablation, seed 44"),
    ("resnet50_flat_seed42", "resnet50_hierarchical_seed42", "backbone ablation (R1 pt.5)"),
    ("hierarchical_seed42", "hierarchical_soft_seed42", "hard vs. soft penalty (R1 pt.2/pt.4, R2 pt.3)"),
]


def _find_run_dir(experiment_name: str) -> Path | None:
    """Match `{any date}_{experiment_name}` exactly -- glob, don't assume
    which date it landed on. Picks the newest if more than one date somehow
    matches (e.g. a rerun the next day)."""
    matches = sorted(RESULT_ROOT.glob(f"*_{experiment_name}"))
    return matches[-1] if matches else None


def run_dir(name: str) -> Path | None:
    """The run's own result directory (holds config.yaml, checkpoints/,
    predictions/) -- for callers that need the checkpoint itself, e.g.
    Grad-CAM, not just the already-computed predictions CSVs."""
    experiment_name, *_ = _RUNS[name]
    return _find_run_dir(experiment_name)


def csv_paths(name: str) -> tuple[Path, Path] | tuple[None, None]:
    d = run_dir(name)
    if d is None:
        return None, None
    p = d / "predictions"
    return p / "val.csv", p / "test.csv"


def available() -> dict[str, tuple[Path, Path]]:
    """Only conditions whose val+test predictions actually exist on disk."""
    out = {}
    for name in _RUNS:
        val_csv, test_csv = csv_paths(name)
        if val_csv is not None and val_csv.exists() and test_csv.exists():
            out[name] = (val_csv, test_csv)
    return out


def missing() -> list[str]:
    return [name for name in _RUNS if name not in available()]
