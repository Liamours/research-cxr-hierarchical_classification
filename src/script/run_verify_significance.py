"""Re-run the manuscript's significance tests (Sec III-A / Table II) against
the real, confirmed-genuine test predictions, and diff against the reported
values. No training, no GPU -- pure numpy over already-saved predictions CSVs.

    uv run python src/script/run_verify_significance.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluate.statistics import (
    compare_aurc, compare_auroc_macro, compare_f1_macro,
    load_predictions, mcnemar_comparison,
)

FLAT_CSV = Path(r"C:\rifqi\research-cxr-hierarchical_classification\inference_results\predictions-densenet121_xrv-flat-260719\test.csv")
HIER_CSV = Path(r"C:\rifqi\research-cxr-hierarchical_classification\inference_results\predictions-densenet121_xrv-hierarchical-260718\test.csv")

# main.tex reported values, for the diff.
REPORTED = {
    "auroc_macro_p": 0.309,
    "f1_macro_p": 0.0001,       # reported as p<0.0001
    "aurc_p": 0.0001,           # reported as p<0.0001 (AURC micro)
    "mcnemar_flat_wins": 2228,
    "mcnemar_hier_wins": 543,
    "mcnemar_p": 0.0001,        # reported as p<0.0001
    "accuracy_flat": 0.9467,
    "accuracy_hier": 0.9430,
}


def main() -> None:
    pf, tf, mf, _ = load_predictions(FLAT_CSV)
    ph, th, mh, _ = load_predictions(HIER_CSV)
    print(f"flat: {pf.shape}  hierarchical: {ph.shape}\n")

    auroc = compare_auroc_macro(pf, tf, mf, ph, th, mh, label=("flat", "hierarchical"))
    f1 = compare_f1_macro(pf, tf, mf, ph, th, mh, label=("flat", "hierarchical"))
    aurc = compare_aurc(pf, tf, mf, ph, th, mh, label=("flat", "hierarchical"))
    mcn = mcnemar_comparison(pf, tf, mf, ph, th, mh, label=("flat", "hierarchical"))

    print("=== AUROC macro (H1: flat > hierarchical) ===")
    print(auroc)
    print(f"  reported p=0.309 -> got p={auroc['p_value_h1_a_better']}"
          f"  {'MATCH' if abs(auroc['p_value_h1_a_better'] - REPORTED['auroc_macro_p']) < 0.02 else 'CHECK'}")

    print("\n=== F1 macro (H1: flat > hierarchical) ===")
    print(f1)
    print(f"  reported p<0.0001 -> got p={f1['p_value_h1_a_better']}"
          f"  {'MATCH' if f1['p_value_h1_a_better'] < 0.001 else 'CHECK'}")

    print("\n=== AURC (H1: flat < hierarchical, lower risk) ===")
    print(aurc)
    print(f"  reported p<0.0001 -> got p={aurc['p_value_h1_a_lt_b']}"
          f"  {'MATCH' if aurc['p_value_h1_a_lt_b'] < 0.001 else 'CHECK'}")

    print("\n=== McNemar (flat vs hierarchical, threshold 0.5) ===")
    print(mcn)


if __name__ == "__main__":
    main()
