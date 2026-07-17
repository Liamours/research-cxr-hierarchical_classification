# Experiment Summary (bridge doc — not paper prose)

Last updated: 2026-07-16. Working notes only: raw numbers + caveats, updated
as runs land. Promote settled numbers into `draft/sections/` once the
experiment set stabilizes (see "Open items" below for what's still moving).

## Status

| Condition | Run dir | Status |
|---|---|---|
| flat (MaskedBCE) | `result/20260712_densenet121_xrv__flat` | complete: 15/15 epochs, val+test evaluated |
| hierarchical (HBCE, lam=0.5) | `result/20260712_densenet121_xrv__hierarchical` | complete: 15/15 epochs, val+test evaluated |
| BAFL | `result/20260712_densenet121_xrv__seg-off__uq-off__bafl` | not trained — killed before any checkpoint saved |

`result/20260716_densenet121_xrv__flat` is not a model: no checkpoints, no
config.yaml. It's a `RunLogger` side effect from re-running the flat model's
test-split eval on 2026-07-16 with `--run-dir` pointing elsewhere (the logger
always writes to `cfg.run_dir()` = today's date, independent of `--run-dir`).
That's also why the flat model's `eval_metrics_test.json` mtime is 2026-07-16
even though training finished 2026-07-12 (first test-eval attempt crashed with
`MemoryError`; retried successfully today).

## Architecture (identical for flat and hierarchical)

- Backbone: DenseNet121, TorchXRayVision weights `densenet121-res224-all`
- Head: `Dropout(0.2) -> Linear(1024 -> 51)`, raw logits
- Input: `(batch, 1, 224, 224)` grayscale, XRV-normalized `[-1024, 1024]`
- Params: 7,018,309 total (backbone 6,966,034 + head 52,275); 2.755 GMACs
- Freeze schedule: epoch 1 backbone frozen, epoch >=2 unfrozen; differential
  LR backbone=1e-5 / head=1e-4
- Loss: `MaskedBCELoss` (flat, lam=0.0) vs `HBCELoss` (hierarchical, lam=0.5,
  13 IS-A edges — penalty confirmed non-zero every epoch, 0.0002 -> 0.0009)

## Dataset

`dataset/combined/combined.csv`, frozen since 2026-07-12 01:27 (unchanged
through both trainings and the 07-16 re-eval). 527,745 images, 7 datasets,
51 canonical labels (27 with real signal, 24 always NaN-masked).

| dataset | train | val | test | total |
|---|--:|--:|--:|--:|
| chexpert | 161,920 | 12,890 | 12,864 | 187,674 |
| covidx-cxr4 | 71,690 | 6,644 | 6,483 | 84,817 |
| nih-cxr14 | 92,105 | 10,017 | 9,998 | 112,120 |
| padchest | 88,349 | 11,166 | 11,099 | 110,614 |
| tbx11k | 6,720 | 840 | 840 | 8,400 |
| vindr-cxr | 11,996 | 1,499 | 1,500 | 14,995 |
| vindr-pcxr | 7,300 | 913 | 912 | 9,125 |
| **total** | **440,080** | **43,969** | **43,696** | **527,745** |

Split: multi-label iterative stratification (seed 42, 80/10/10),
`src/script/run_resplit_stratified.py`. Patient/study-level for chexpert,
nih-cxr14, covidx-cxr4, padchest; per-image for tbx11k, vindr-cxr,
vindr-pcxr (no recoverable patient key in those three).

## Results (test split, macro over 27 signal-bearing labels)

| condition | AUROC | F1 | mAP | ECE | AURC | HCV |
|---|--:|--:|--:|--:|--:|--:|
| flat | 0.6954 | 0.1173 | 0.1999 | 0.0044 | 0.0145 | 0.3927 |
| hierarchical | 0.6872 | 0.1107 | 0.1986 | 0.0055 | 0.0170 | 0.3653 |

Full 25-label per-class AUROC/F1/mAP/AURC in each run's `eval_metrics_test.json`.

## Code paths

| Step | Entry point | Core logic |
|---|---|---|
| Config | `configs/densenet121_xrv__{flat,hierarchical}.yaml` | `src/config/experiment_config.py::ExperimentConfig` |
| Train | `src/script/run_train.py` | `src/train/trainer.py::MultiLabelTrainer`, loss via `src/train/losses.py::build_loss` |
| Eval | `src/script/run_evaluate.py --split {val,test}` | `src/evaluate/evaluator.py::evaluate_model` |

## Open items (why this isn't in draft/sections/ yet)

- Single seed (42) for both conditions — no CI, no repeat runs. Differences
  of ~0.01 AUROC may be noise (same caveat as the earlier 4-condition grid).
- BAFL condition has no trained model yet.
- Here, hierarchical has *worse* ECE than flat (0.0055 vs 0.0044) and *lower*
  HCV (0.3653 vs 0.3927) — opposite calibration direction from the earlier
  6-dataset grid (`analysis_export/SUMMARY.md`, pre-padchest/pre-resplit),
  where hierarchical improved ECE monotonically with lambda. Different
  dataset pool (padchest added, full resplit) — not yet reconciled or
  explained, needs a second look before writing any calibration claim.
- Dataset has been resplit multiple times this project (pre-padchest,
  pre-resplit, pre-stratified backups all exist under
  `dataset/combined/backups/`) — confirm no other resplit is planned before
  freezing these numbers into the paper.
