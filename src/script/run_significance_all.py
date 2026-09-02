"""Item 5: statistical tests (paired bootstrap on AUROC macro, F1 macro,
AURC; McNemar on threshold-0.5 decisions) for every natural comparison in
the 2026-08-30 batch: flat vs. hierarchical at each seed, the backbone
ablation, and hard vs. soft penalty. Same method and same functions as
run_verify_significance.py (the manuscript's original A-series test, kept
as-is for the two originally reported checkpoints); this one loops over
the comparisons in src/script/_conditions.py::COMPARISONS instead.

    uv run python src/script/run_significance_all.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluate.statistics import (
    compare_aurc, compare_auroc_macro, compare_f1_macro,
    load_predictions, mcnemar_comparison,
)
from src.script._conditions import COMPARISONS, available


def main() -> None:
    conds = available()
    for name_a, name_b, question in COMPARISONS:
        if name_a not in conds or name_b not in conds:
            print(f"\n=== {question} ({name_a} vs {name_b}) === SKIPPED, predictions not ready")
            continue
        _, test_a = conds[name_a]
        _, test_b = conds[name_b]
        pa, ta, ma, _ = load_predictions(test_a)
        pb, tb, mb, _ = load_predictions(test_b)

        print(f"\n=== {question}: {name_a} vs {name_b} ===")
        auroc = compare_auroc_macro(pa, ta, ma, pb, tb, mb, label=(name_a, name_b))
        f1 = compare_f1_macro(pa, ta, ma, pb, tb, mb, label=(name_a, name_b))
        aurc = compare_aurc(pa, ta, ma, pb, tb, mb, label=(name_a, name_b))
        mcn = mcnemar_comparison(pa, ta, ma, pb, tb, mb, label=(name_a, name_b))

        print(f"  AUROC macro: {name_a}={auroc['auroc_macro_a']['estimate']:.4f} "
              f"{name_b}={auroc['auroc_macro_b']['estimate']:.4f}  "
              f"delta={auroc['delta_a_minus_b']:+.4f}  p={auroc['p_value_h1_a_better']}")
        print(f"  F1 macro:    {name_a}={f1['f1_macro_a']['estimate']:.4f} "
              f"{name_b}={f1['f1_macro_b']['estimate']:.4f}  "
              f"delta={f1['delta_a_minus_b']:+.4f}  p={f1['p_value_h1_a_better']}")
        print(f"  AURC:        {name_a}={aurc['aurc_a']['estimate']:.4f} "
              f"{name_b}={aurc['aurc_b']['estimate']:.4f}  "
              f"delta={aurc['delta_a_minus_b']:+.4f}  p={aurc['p_value_h1_a_lt_b']}")
        print(f"  McNemar: {name_a} wins {mcn['label_a_wins']}, {name_b} wins {mcn['label_b_wins']}, "
              f"p={mcn['p_value_two_tailed']}  "
              f"acc {mcn[f'{name_a}_acc']:.4f} vs {mcn[f'{name_b}_acc']:.4f}")


if __name__ == "__main__":
    main()
