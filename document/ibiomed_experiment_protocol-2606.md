iBioMed Experiment Protocol (pre-registration)
status: frozen design; backbone fixed to DenseNet121-XRV; 8-condition grid running.
mechanism under study: Uncertainty-Gated Coarse Fallback
backbone: DenseNet121 (TorchXRayVision, densenet121-res224-all, CXR-pretrained).
grid: 3-factor (seg × label_structure × UQ) = 8 conditions.
Authoritative design: configs/grid_base.yaml + configs/grid/*.yaml

Purpose
This protocol fixes the datasets, splits, models, metrics, statistics, and
decision criteria before running, so results cannot be shaped after the fact.
It also lists the shortcut and leakage checks that must pass before any number
is trusted. Anything not fixed here (only the uncertainty thresholds) is tuned
on validation only; the test split is untouched until final evaluation.


1. Hypotheses
Primary: at matched coverage, uncertainty-gated coarse fallback achieves lower
  selective risk (higher accuracy on committed predictions) than flat abstention
  on multi-label chest X-ray, measured by area under the risk-coverage curve
  (AURC). Stated as: AURC_fallback < AURC_flat with non-overlapping confidence
  intervals.
Secondary 1: hierarchical multi-label training (HBCE) does not reduce flat AUROC
  versus flat BCE training by more than a small margin (non-inferiority within
  0.01 macro AUROC).
Secondary 2: organ segmentation conditioning (seg-on vs seg-off) does not degrade
  AUROC by more than 0.01; report any gain separately.
Secondary 3: Grad-CAM activation mass falls predominantly within the lung field
  on correct positives (localization sanity); no formal hypothesis.


2. Datasets and access
Training data: combined pool of 6 datasets (NIH ChestX-ray14, CheXpert+,
  VinDr-CXR, VinDr-PCXR, TBX11K, COVIDx-CXR4) via a single canonical 55-column
  CSV (combined.csv, 417,136 images). Multi-source pooling is permitted because
  masked BCE loss supervises each image only on the labels its source dataset
  provides; unannotated labels stay NaN-masked and receive no gradient. This
  differs from naive label-pooling: source-specific label subsets are never mixed
  or imputed, so cross-source label confusion is structurally prevented.
  See context/research-cxr_datasets-2606.md for individual dataset details.
Bounding boxes for localization: NIH BBox_List_2017 (8 diseases, ~984 images).


3. Splits
Patient-level only. For NIH use the official train_val_list.txt and
test_list.txt; carve validation from the train+val list by patient id so no
patient crosses train/val. Test split is never read until final evaluation.
Pre-run gate: the leakage test (no patient in more than one split) must pass.


4. Preprocessing (frozen)
Per the standardized protocol: frontal view (PA preferred), grayscale, resize
shorter side to 256, center-crop 224, replicate to 3 channels, save 8-bit PNG.
Normalization at load: TorchXRayVision convention, pixel range [-1024, 1024].
Uncertain and not-mentioned labels mapped to 0 (U-Zero). No CLAHE.
Augmentation (train only): small affine (rotation 10 deg, translate 0.05, scale
0.95-1.05) + mild brightness/contrast 0.1; horizontal flip OFF (laterality).
Annotated label counts per ingested dataset (from configs/dataset_registry.json):
  NIH ChestX-ray14 : 10 labels  CheXpert+     : 8 labels
  VinDr-CXR        :  6 labels  VinDr-PCXR    : 7 labels
  TBX11K           :  1 label   COVIDx-CXR4   : 1 label
The remaining labels (up to 51) are NaN-masked for each dataset.


5. Label space and hierarchy
Clinical target list grounded in Indonesian authority (PDPI/PNPK/KKI) — see
context/research-cxr_canonical-diseases-2606.md (51 diseases, flat — no grouping).
Canonical label set: src/data/label_space.py CANONICAL_LABELS (51 entries).
Combined training CSV: 55 columns (image_id, image_path, dataset, split + 51
labels); 17 labels have signal across the 6 ingested datasets, 34 are always
NaN-masked. Cross-dataset name mapping from configs/label_equivalence.json.
For each image only its source dataset's applicable labels participate in loss
and metrics; inapplicable labels stay NaN-masked (masked BCE).
Annotated label counts per ingested dataset (from configs/dataset_registry.json):
  NIH ChestX-ray14 : 10 labels  CheXpert+     : 8 labels
  VinDr-CXR        :  6 labels  VinDr-PCXR    : 7 labels
  TBX11K           :  1 label   COVIDx-CXR4   : 1 label


6. Experiment matrix
Backbone fixed: DenseNet121-XRV (TorchXRayVision, CXR-pretrained). Not a variable.
3-factor ablation grid (2^3 = 8 conditions):
  seg            : {off, on}   — organ segmentation mask conditioning (CheXmask-U)
  label_structure: {flat, hierarchical} — BCE vs HBCE loss
  uq             : {none, mc_dropout}  — single-pass vs MC Dropout (T=30)
Core comparison for the primary hypothesis: hierarchical + mc_dropout condition
vs flat + none condition, coarse fallback AURC vs flat abstention AURC.
Run via configs/grid/*.yaml with run_train then run_evaluate; run_grid aggregates.


7. Models and training (frozen hyperparameters)
Backbone: DenseNet121 (TorchXRayVision, weights=densenet121-res224-all, CXR-pretrained).
Single flat head: linear 1024→51 + sigmoid. Dropout p=0.2 before head (reused
for MC Dropout at inference). No separate coarse-group head.
AdamW, differential learning rate backbone 1e-5 / head 1e-4, cosine annealing with
1000-step linear warm-up, bf16 on CUDA, gradient accumulation (effective bs 32),
grad clip 1.0, backbone frozen epoch 1, max 15 epochs, early stop patience 3 on
validation loss. Seed 42.
Flat loss: masked BCE (NaN entries excluded from numerator and denominator).
Hierarchical loss: L_HBCE = L_BCE + 0.5 * sum_{(g,c) in edges} ReLU(p_c-0.5)*ReLU(0.5-p_g)
  where (g,c) are the 13 IS-A edges in src/data/label_space.py HIERARCHY.
  Penalty is non-zero only for hierarchically inconsistent predictions.
Four checkpoints saved (last, best val loss, best val macro AUROC, best val macro
F1); best val macro AUROC is the model used for test evaluation.


8. Metrics
Discrimination: macro and micro AUROC, macro and micro F1 (mask-aware).
Calibration: Expected Calibration Error and reliability diagram.
Selective: risk-coverage curve and AURC; hierarchical risk-coverage with the
  ancestor rule (a coarse prediction is correct if any child is truly present);
  coarse fallback vs flat abstention at matched coverage.
Hierarchy: hierarchical consistency (fraction of positive fine predictions whose
  parent is also positive).
Localization: pointing game hit rate and IoU on the bounding-box subset.
Report metrics as decimals to four places.


9. Sanity targets (gate before trusting any result)
DenseNet121-XRV flat macro AUROC on NIH test is expected in 0.80-0.86 (CheXNet
reported 0.843; TorchXRayVision reproductions ~0.85). Decision: below ~0.75
indicates a bug (stop and debug); above ~0.92 indicates likely leakage (stop and
audit). These are go/no-go gates, not results.


10. Statistical analysis
Report 95 percent confidence intervals from bootstrap resampling (1000
resamples) for all headline metrics. Compare AUROC with DeLong's test; compare
thresholded predictions with McNemar's test. The primary hypothesis requires the
coarse-fallback AURC interval to sit below the flat-abstention AURC interval.
No claim is made without an interval.


11. Shortcut and leakage pre-run checks (must pass before main runs)
a. Patient-level leakage test passes (no shared patient across splits).
b. Multi-source pooling integrity: confirm NaN-masked labels receive no gradient
   (masked BCE denominator excludes NaN entries). Cross-source label confusion
   is structurally prevented; verify no inapplicable column is accidentally set
   to 0 instead of NaN in any ingested CSV.
c. After the baseline trains, Grad-CAM activation mass falls predominantly
   inside the lung field on a sample of correct positives (qualitative + IoU on
   the bbox subset). If activation sits on borders/text/devices, flag shortcut.
d. Quick source/view probe: confirm no trivial separability by view position or
   image size that the model could exploit. If external data is later pooled by
   mistake, this is where it would show.


12. Uncertainty protocol
MC Dropout, T = 30 forward passes, evaluation and inference only (training is a
single deterministic pass). Before using uncertainty for the fallback gate,
verify on validation that incorrect predictions carry higher predictive variance
than correct ones; if not, report this and reconsider MC Dropout (Deep Ensemble
is the fallback method if compute allows). The fallback threshold tau is selected
on validation only.


13. Decision criteria for the contribution
Accept (report as positive): coarse fallback shows lower AURC than flat
  abstention with non-overlapping 95 percent intervals on the NIH test split,
  with the localization check (11c) not indicating shortcut behavior.
Null (report honestly): if intervals overlap, report that coarse fallback did
  not significantly beat flat abstention on this data; the comparison and
  protocol are still a contribution at this venue.
Either outcome is publishable at iBioMed; the protocol prevents reframing a null
  as a win.


14. Reproducibility and compute
Seed 42 fixed across random, numpy, torch, cudnn. Every run snapshots its config
and logs per-epoch metrics, events, and final metrics. Code and configs released.
Hardware: RTX 4050 (6 GB) or 5070 (12 GB); cfg.data.loading auto-downgrades
gpu/ram caching to lazy if it will not fit. bf16 + gradient accumulation keep
memory within budget.


15. Execution order
1. Confirm dataset selection (see context/research-cxr_datasets-2606.md).
   Planned: NIH ChestX-ray14 (+ BBox list) to data/raw.
2. Preprocess selected dataset (src/data/preprocess) to the standard layout.
3. run_label_audit + run_eda: confirm label mapping coverage, class
   distribution, image integrity, and the leakage gate (section 11a).
4. Phase 1: train flat condition (seg-off, flat, uq-none); confirm sanity band
   (section 9) before continuing.
5. Phase 2: hierarchical training (seg-off, hierarchical, uq-none); check
   non-inferiority vs flat (Secondary 1).
6. Phase 3: MC Dropout + coarse fallback (seg-off, hierarchical, mc_dropout);
   uncertainty sanity (section 12); produce risk-coverage and AURC fallback vs
   flat (primary hypothesis).
7. Run remaining 5 grid conditions (seg-on variants); Secondary 2 seg analysis.
8. Localization (section 11c) and, if available, external-dataset evaluation.
8. Statistics (section 10), then write Results.
