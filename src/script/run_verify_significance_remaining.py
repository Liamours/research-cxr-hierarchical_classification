"""Extend the manuscript's significance testing to the 6 metrics still
marked "not tested" in the primary results table (2026-09-02, from draft
session): AUROC micro, mAP macro, AURC macro, F1 micro, MCC macro,
balanced accuracy macro. Already tested elsewhere: AUROC macro, AURC
(micro/flat), F1 macro (run_verify_significance.py), HCV rate, ECE
(run_verify_significance_hcv_ece.py). Same seeded paired bootstrap
(1,000 resamples, seed 42), same seed42 predictions, same _compare_metric
machinery -- no new methodology.

Each metric is tested in whichever direction its own point estimate
favors (same convention run_verify_significance_hcv_ece.py used for HCV:
flip which condition is "A" per metric rather than always forcing "flat
better"). As it turns out flat's point estimate leads on all 6 here, so
every row below tests flat-vs-hierarchical -- the per-metric flip support
stays in case a future metric needs it, not because this run used it.

    uv run python src/script/run_verify_significance_remaining.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.label_space import CANONICAL_LABELS
from src.evaluate.metrics import (
    _macro, compute_confusion_metrics, micro_auroc, micro_f1, per_class_ap,
)
from src.evaluate.selective import per_class_aurc
from src.evaluate.statistics import _compare_metric, load_predictions

FLAT_CSV = Path(r"C:\rifqi\research-cxr-hierarchical_classification\inference_results\predictions-densenet121_xrv-flat-260719\test.csv")
HIER_CSV = Path(r"C:\rifqi\research-cxr-hierarchical_classification\inference_results\predictions-densenet121_xrv-hierarchical-260718\test.csv")


def _map_macro(p, t, m):
    return _macro(per_class_ap(p, t, m, CANONICAL_LABELS))


def _aurc_macro(p, t, m):
    return _macro(per_class_aurc(p, t, m, CANONICAL_LABELS))


def _mcc_macro(p, t, m):
    return compute_confusion_metrics(p, t, m, CANONICAL_LABELS)["macro"]["mcc"]


def _bal_acc_macro(p, t, m):
    return compute_confusion_metrics(p, t, m, CANONICAL_LABELS)["macro"]["balanced_accuracy"]


def main() -> None:
    pf, tf, mf, _ = load_predictions(FLAT_CSV)
    ph, th, mh, _ = load_predictions(HIER_CSV)

    # (metric_fn, name, higher_is_better, point-estimate-favored condition as "a")
    tests = [
        ("auroc_micro",             micro_auroc,      True,  "flat"),
        ("map_macro",               _map_macro,       True,  "flat"),
        ("aurc_macro",              _aurc_macro,      False, "flat"),
        ("f1_micro",                micro_f1,         True,  "flat"),
        ("mcc_macro",               _mcc_macro,       True,  "flat"),
        ("balanced_accuracy_macro", _bal_acc_macro,   True,  "flat"),
    ]

    for name, fn, higher_is_better, favored in tests:
        if favored == "flat":
            pa, ta, ma, pb, tb, mb = pf, tf, mf, ph, th, mh
            label = ("flat", "hierarchical")
        else:
            pa, ta, ma, pb, tb, mb = ph, th, mh, pf, tf, mf
            label = ("hierarchical", "flat")

        result = _compare_metric(fn, name, pa, ta, ma, pb, tb, mb,
                                  higher_is_better=higher_is_better, label=label)
        print(f"=== {name}: {label[0]} vs {label[1]} (favored direction, per point estimate) ===")
        print(f"  {label[0]}={result[f'{name}_a']['estimate']}  {label[1]}={result[f'{name}_b']['estimate']}"
              f"  delta={result['delta_a_minus_b']:+.4f}  ci95={result['delta_ci_95']}"
              f"  p={result['p_value_h1_a_better']}")


if __name__ == "__main__":
    main()
