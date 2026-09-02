"""Edge co-annotation coverage matrix for the HBCE hierarchy.

For each parent-child edge in src/data/hierarchy.py, counts how many pooled-
corpus rows have BOTH labels annotated (non-NaN) -- the only rows that can
contribute gradient to that edge's HBCE penalty term -- broken down by
contributing source dataset. Reads only dataset/combined/combined.csv.

    uv run python src/script/run_hierarchy_coverage.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.data.hierarchy import PARENT_CHILD_EDGES

COMBINED_CSV = Path(r"C:\rifqi\research-cxr-hierarchical_classification\dataset\combined\combined.csv")


def main() -> None:
    df = pd.read_csv(COMBINED_CSV)
    total = len(df)
    print(f"Corpus: {total:,} rows, {df['dataset'].nunique()} datasets\n")

    rows = []
    for parent, child in PARENT_CHILD_EDGES:
        parent_ann = df[parent].notna()
        child_ann = df[child].notna()
        co_ann = parent_ann & child_ann
        co_pos_child = co_ann & (df[child] == 1)
        sources = sorted(df.loc[co_ann, "dataset"].unique().tolist())
        rows.append({
            "edge": f"{parent} -> {child}",
            "parent_ann": int(parent_ann.sum()),
            "child_ann": int(child_ann.sum()),
            "co_annotated": int(co_ann.sum()),
            "co_annotated_pct": co_ann.sum() / total * 100,
            "co_pos_child": int(co_pos_child.sum()),
            "sources": sources,
        })

    active = [r for r in rows if r["co_annotated"] > 0]
    print(f"{len(active)} / {len(rows)} edges have >=1 co-annotated row\n")

    print(f"{'edge':<44}{'parent_ann':>11}{'child_ann':>11}{'co_ann':>9}{'co_ann%':>9}{'co_pos':>8}  sources")
    print("-" * 130)
    for r in rows:
        print(f"{r['edge']:<44}{r['parent_ann']:>11,}{r['child_ann']:>11,}"
              f"{r['co_annotated']:>9,}{r['co_annotated_pct']:>8.3f}%{r['co_pos_child']:>8,}  {r['sources']}")


if __name__ == "__main__":
    main()
