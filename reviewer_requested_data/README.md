# Reviewer-requested release data

Per iBioMed 2026 review point R2.8: "Releasing the label harmonization map,
split CSV, preprocessing code, and all evaluation predictions is essential
because the contribution depends strongly on corpus construction."

- Preprocessing code and the label harmonization map (`src/data/label_map.py`
  and the rest of `src/data/`) are in `../src/`, part of the normal repo tree.
- `combined.csv.gz` is the pooled 527,745-image corpus: all 51 canonical
  finding labels plus the train/val/test split assignment used throughout
  the paper (80/10/10, multi-label iterative stratification). The original
  export also carried each image's absolute local filesystem path
  (`C:\rifqi\...`); that column has been dropped here, since it doesn't
  resolve on any other machine and it exposed a local username. Reconstruct
  a path from `image_id` and `dataset` if needed:
  `dataset/<dataset>/preprocessed/images/<dataset>/<image_id>.png`.
- `predictions/` has per-condition test and validation set predictions
  (13 conditions: the primary flat/hierarchical comparison at 3 seeds,
  the ResNet50 backbone ablation, the soft-penalty condition, and a few
  verification/reproduction re-runs from the review round).

## Before committing: this collides with the repo's own .gitignore

The parent `.gitignore` excludes `*.csv` globally and `/dataset/` at the
root -- deliberate defaults for the normal case, but this folder is the
explicit exception the reviewer asked for. Either:

```bash
git add -f reviewer_requested_data/
```

or add a negation rule near the top of `.gitignore`:

```
!/reviewer_requested_data/
!/reviewer_requested_data/**
```

## Size and format

Every file here is gzip-compressed (`.csv.gz`). Read directly with
`pandas.read_csv("combined.csv.gz")` or any tool that auto-detects gzip;
no separate decompression step needed. Load a file with the leaked-path
column already removed, don't regenerate from the old export without also
dropping `image_path`.

| | Before | After |
|---|--:|--:|
| `combined.csv` | 150.7 MB | 5.3 MB |
| `predictions/` (22 files) | 1080.8 MB | 152.4 MB |
| **Total** | **1231.5 MB** | **157.7 MB** |

Largest single file is 6.6 MB, well under GitHub's 100 MB hard limit and
its 50 MB warning threshold. A plain `git push` works; **Git LFS is not
needed** for this data (the free tier's 1 GB storage quota wouldn't have
covered the original 1.2 GB anyway).
