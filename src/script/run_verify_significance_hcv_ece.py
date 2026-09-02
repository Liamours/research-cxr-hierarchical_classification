"""A7: extend the manuscript's significance testing to hierarchy violation
rate (HCV) and calibration (ECE) -- both reported in Table II but never
statistically tested (R1 pt.8). Reuses the same seeded bootstrap machinery
already verified against the manuscript's existing tests.

    uv run python src/script/run_verify_significance_hcv_ece.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.label_space import CANONICAL_LABELS
from src.evaluate.calibration import expected_calibration_error
from src.evaluate.metrics import hierarchy_violation_rate
from src.evaluate.statistics import _compare_metric, load_predictions

FLAT_CSV = Path(r"C:\rifqi\research-cxr-hierarchical_classification\inference_results\predictions-densenet121_xrv-flat-260719\test.csv")
HIER_CSV = Path(r"C:\rifqi\research-cxr-hierarchical_classification\inference_results\predictions-densenet121_xrv-hierarchical-260718\test.csv")


def _hcv(p, t, m):
    return hierarchy_violation_rate(p, CANONICAL_LABELS, 0.5)["rate"]


def main() -> None:
    pf, tf, mf, _ = load_predictions(FLAT_CSV)
    ph, th, mh, _ = load_predictions(HIER_CSV)

    print("=== Hierarchy violation rate (H1: flat > hierarchical, i.e. hier has fewer violations) ===")
    hcv = _compare_metric(_hcv, "hcv", pf, tf, mf, ph, th, mh, higher_is_better=True, label=("flat", "hierarchical"))
    print(hcv)
    print(f"  manuscript: 36.95% -> 33.82%, reported as an improvement, never tested. This bootstrap: p={hcv['p_value_h1_a_better']}")

    print("\n=== ECE (H1: flat < hierarchical, i.e. flat is better calibrated) ===")
    ece = _compare_metric(expected_calibration_error, "ece", pf, tf, mf, ph, th, mh, higher_is_better=False, label=("flat", "hierarchical"))
    print(ece)
    print(f"  manuscript: 0.0059 vs 0.0097, reported as flat-better, never tested. This bootstrap: p={ece['p_value_h1_a_better']}")


if __name__ == "__main__":
    main()
