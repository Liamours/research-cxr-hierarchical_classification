# Experiment Details

Generated: 2026-07-16 22:22

Exact experimental setup behind `CLASSIFICATION_RESULTS-260716-2218.md`,
`LOCALIZATION_RESULTS-260716-2218.md`, and the accompanying Excel/YAML exports.
All values pulled directly from the run configs, training logs, and checkpoint
metadata -- not recalled from memory.

## Corpus

- `dataset/combined/combined.csv`, 527,745 images, 7 datasets (chexpert, covidx-cxr4,
  nih-cxr14, padchest, tbx11k, vindr-cxr, vindr-pcxr). MIMIC-CXR excluded (never ingested).
- 27 of 51 canonical labels have signal (see `DATASET_ACCESS.md` for full per-label counts).
- Split: multi-label iterative stratification (Sechidis 2011), group-level where a
  patient/study key is recoverable, 80/10/10, seed 42. Train 440,080 / Val 43,969 / Test 43,696.

## Model

- Backbone: DenseNet121, TorchXRayVision weights `densenet121-res224-all` (pretrained on
  NIH+PadChest+CheXpert+MIMIC+OpenI+RSNA, CXR-domain).
- Head: `Dropout(0.2) -> Linear(1024, 51)`, sigmoid multi-label output.
- Total params: 7,018,309 (backbone 6,966,034 + head 52,275). 2.755 GMACs @ 224x224.
- Input: 1-channel grayscale, XRV normalization to [-1024, 1024].

## Two conditions trained

| | Flat | Hierarchical |
|---|---|---|
| Config | `configs/densenet121_xrv__flat.yaml` | `configs/densenet121_xrv__hierarchical.yaml` |
| Loss | MaskedBCELoss | HBCELoss (lambda=0.5) |
| label.lam | 0.0 | 0.5 |
| Result dir | `result/20260712_densenet121_xrv__flat/` | `result/20260712_densenet121_xrv__hierarchical/` |

Both otherwise identical: same backbone, same data, same training protocol, same seed --
architecture and hyperparameters held constant so the only variable is the loss function.

## Training protocol (both conditions)

| Setting | Value |
|---|---|
| Optimizer | AdamW, weight_decay=0.01 |
| LR schedule | OneCycleLR, cosine anneal, warmup_steps=1000 (capped at 30% of total steps) |
| Differential LR | backbone 1e-5, head 1e-4 |
| Backbone freeze | epoch 1 only (freeze_backbone_epochs=1) |
| Precision | bf16 autocast on CUDA |
| Batch size | 32 |
| Grad accum steps | 1 |
| Grad clip (max norm) | 1.0 |
| Epochs (max / early-stop patience) | 15 / 3 (on val_loss) |
| Augmentation (train only) | hflip=False, rotation=+/-10deg, translate=0.05, scale_jitter=0.05, brightness=0.1, contrast=0.1 |
| Seed | 42 |
| DataLoader workers | 12, pin_memory=True, persistent_workers=True |

## Actual training outcome

| | Flat | Hierarchical |
|---|--:|--:|
| Epochs run | 15 (full budget, early stop never triggered) | 15 (full budget) |
| Best epoch (min val_loss) | 13 (val_loss=0.1918) | 15 (val_loss=0.2032) |
| Checkpoint used for eval | `best_val_auroc_macro.pt`, epoch 14 (val_auroc_macro=0.6735) | `best_val_auroc_macro.pt`, epoch 14 (val_auroc_macro=0.6731) |
| HBCE penalty at final epoch | n/a (flat loss has no penalty term) | 0.0009 (non-zero -- confirms the HBCE child-mask fix from earlier this session is active) |

Note: `best_val_auroc_macro.pt` (used for all reported metrics) is selected by a different
criterion than `best_val_loss` (lowest val_loss), so the "best epoch" by loss and the epoch
whose weights were actually evaluated can differ -- that's expected, not an inconsistency.

## Evaluation

- Metrics computed: `auroc, f1, ece, map, aurc, hcv, clf` (AUROC, F1, Expected Calibration
  Error, mean Average Precision, Area Under Risk-Coverage Curve, hierarchy violation rate,
  full confusion family incl. precision/recall/specificity/accuracy/balanced_accuracy/MCC).
- Threshold for all threshold-based metrics: 0.5 (not tuned per-label).
- Both val and test evaluated; val is optimistic (checkpoint was selected on it), test is the
  honest held-out number.
- Predictions saved per model per split: `result/<run>/predictions/{val,test}.csv`.

## Localization sub-experiment

- Ground truth: `dataset/nih-cxr14/preprocessed/labels/nih_bboxes_224.csv` (NIH's official
  bounding-box release, 984 boxes / 880 images / 8 categories, already rescaled to 224x224).
  Original un-rescaled source not present locally -- rescale correctness not independently
  re-verified (see `LOCALIZATION_RESULTS-260716-2218.md` for the specific concern re:
  Cardiomegaly box sizes).
- Label mapping: 6 of 8 NIH categories map to canonical labels (Atelectasis, Pleural_Effusion,
  Cardiomegaly, Pneumonia, Pneumothorax, Solitary_Pulmonary_Nodule). Infiltration dropped
  (no canonical home).
- Bbox ground truth cross-referenced against OUR stratified re-split (not NIH's original
  split label): 100 usable boxes on val, 118 on test, 643 on train (train not processed).
- Method: Grad-CAM (last DenseNet121 conv block, `src/xai/gradcam.py`) for the ground-truth
  label's class index -> 224x224 heatmap. Predicted box = bounding box of the largest
  connected component above the 90th percentile of the heatmap's own pixel values (percentile
  thresholding chosen over a fixed fraction-of-max threshold because the latter produced
  degenerate whole-image boxes on low-confidence predictions -- verified on a real example).
- Metrics: IoU, IoBB (intersection / GT-box-area), Pointing Game (heatmap peak inside GT box),
  and detection-style AP@{0.1, 0.25, 0.5} (score = model confidence, hit = IoU>=threshold).

## Deliverables produced this session (dated 2026-07-16)

| File | Contents |
|---|---|
| `CLASSIFICATION_RESULTS-260716-2218.md` | Overall + per-label AUROC/AP tables, val & test |
| `LOCALIZATION_RESULTS-260716-2218.md` | IoU/IoBB/Pointing Game + mAP@0.1/0.25/0.5 tables |
| `CLASSIFICATION_TABLES_val-260716-2225.xlsx` | Per-image predictions (both models) + ground truth, val split, 27 signal labels |
| `CLASSIFICATION_TABLES_test-260716-2225.xlsx` | Same, test split |
| `LOCALIZATION_TABLES_val-260716-2230.xlsx` | Per-image GT bbox, predicted bbox, IoU/IoBB/pointing-game/confidence, heatmap path (both models), val split |
| `LOCALIZATION_TABLES_test-260716-2230.xlsx` | Same, test split |
| `localization_bboxes_val-260716-2230.yaml` | Predicted + GT bbox coordinates, structured by image/label/model |
| `localization_bboxes_test-260716-2230.yaml` | Same, test split |
| `EXPERIMENT_DETAILS-260716-2222.md` | This file |

Heatmap overlay PNGs (436 total: 218 images x 2 models) are archived at
`document/HEATMAPS-260716-2232.zip` (internal structure: `<flat|hierarchical>/<val|test>/
<image_id>__<label>.png`). The `heatmap_path` field in both YAML files is relative to this
zip's extracted root, e.g. `flat/test/00029817_009__Atelectasis.png`.
