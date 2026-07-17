# Localization Results (Grad-CAM vs NIH bounding boxes)

Generated: 2026-07-16 22:18

Evaluates whether Grad-CAM heatmaps from the trained classifiers actually point at the
pathology, using the NIH ChestX-ray14 official bounding-box release as ground truth.

## Setup

- **Ground truth source:** `dataset/nih-cxr14/preprocessed/labels/nih_bboxes_224.csv`
  (NIH's official bbox release, already resized to 224x224 to match our preprocessed images).
  984 boxes total, 880 unique images, 8 NIH categories.
- **Label mapping:** 6 of 8 NIH categories map to our canonical labels (Atelectasis,
  Pleural_Effusion [from Effusion], Cardiomegaly, Pneumonia, Pneumothorax,
  Solitary_Pulmonary_Nodule [from Mass+Nodule]). "Infiltration" has no canonical home and was
  excluded (123 boxes dropped).
- **Split cross-reference:** bbox image_ids were joined against our own stratified re-split
  (not NIH's original split), since our re-split reshuffled which NIH images land in test.
  Result: 135 train, 117 val, **135 boxes on test** before filtering to mapped labels;
  **118 boxes (114 unique images) on test after filtering to the 6 mapped labels.**
- **Models evaluated:** Flat and Hierarchical (same checkpoints as `CLASSIFICATION_RESULTS.md`).
- **Method:** Grad-CAM on the last DenseNet121 conv block for the ground-truth label's
  class_idx -> 224x224 heatmap. Bounding box extracted by thresholding at the 90th percentile
  of the heatmap's own pixel values (top 10% most-activated pixels), then taking the bounding
  box of the largest connected component. Percentile thresholding was used instead of a fixed
  fraction-of-max threshold because the latter produced degenerate whole-image boxes when the
  model's classification confidence for a label was low (verified on a real example: model
  confidence 0.044 for Atelectasis on image 00021007_000 produced a CAM whose 50%-of-max
  threshold covered the entire 224x224 image).

## IoU / IoBB / Pointing Game

IoU = intersection over union. IoBB = intersection over ground-truth-box area (standard in CXR
localization literature since Grad-CAM boxes tend to be larger/more diffuse than tight GT boxes).
Pointing Game = is the heatmap's single peak pixel inside the ground-truth box.

### Flat model

| Label | n | IoU | IoBB | Pointing Game |
|---|--:|--:|--:|--:|
| Atelectasis | 19 | 0.0053 | 0.1800 | 0.0% |
| Cardiomegaly | 11 | 0.0239 | 0.1138 | 0.0% |
| Pleural_Effusion | 30 | 0.0122 | 0.1118 | 3.3% |
| Pneumonia | 12 | 0.0130 | 0.2559 | 0.0% |
| Pneumothorax | 20 | 0.0083 | 0.0500 | 0.0% |
| Solitary_Pulmonary_Nodule | 26 | 0.0012 | 0.1397 | 0.0% |
| **Overall (macro avg)** | **118** | **0.0107** | **0.1419** | **0.6%** |

### Hierarchical model

| Label | n | IoU | IoBB | Pointing Game |
|---|--:|--:|--:|--:|
| Atelectasis | 19 | 0.0050 | 0.1265 | 0.0% |
| Cardiomegaly | 11 | 0.0248 | 0.1106 | 0.0% |
| Pleural_Effusion | 30 | 0.0143 | 0.1160 | 3.3% |
| Pneumonia | 12 | 0.0000 | 0.0000 | 0.0% |
| Pneumothorax | 20 | 0.0142 | 0.0882 | 0.0% |
| Solitary_Pulmonary_Nodule | 26 | 0.0008 | 0.0769 | 0.0% |
| **Overall (macro avg)** | **118** | **0.0099** | **0.0864** | **0.6%** |

## Detection-style mAP

AP computed per label: score = model's classification confidence (sigmoid probability) for
that label on that image, hit = IoU >= threshold vs. the ground-truth box. mAP = macro average
of AP across the 6 labels. Standard object-detection thresholds used: 0.1, 0.25, 0.5.

### Flat model

| Label | n | AP@0.1 | AP@0.25 | AP@0.5 |
|---|--:|--:|--:|--:|
| Atelectasis | 19 | 0.0000 | 0.0 | 0.0 |
| Cardiomegaly | 11 | 1.0000 | 0.0 | 0.0 |
| Pleural_Effusion | 30 | 0.0627 | 0.0 | 0.0 |
| Pneumonia | 12 | 0.0000 | 0.0 | 0.0 |
| Pneumothorax | 20 | 0.0556 | 0.0 | 0.0 |
| Solitary_Pulmonary_Nodule | 26 | 0.0000 | 0.0 | 0.0 |
| **mAP (macro)** | **118** | **0.1864** | **0.0000** | **0.0000** |

### Hierarchical model

| Label | n | AP@0.1 | AP@0.25 | AP@0.5 |
|---|--:|--:|--:|--:|
| Atelectasis | 19 | 0.0000 | 0.0 | 0.0 |
| Cardiomegaly | 11 | 0.4500 | 0.0 | 0.0 |
| Pleural_Effusion | 30 | 0.0845 | 0.0 | 0.0 |
| Pneumonia | 12 | 0.0000 | 0.0 | 0.0 |
| Pneumothorax | 20 | 0.0804 | 0.0 | 0.0 |
| Solitary_Pulmonary_Nodule | 26 | 0.0000 | 0.0 | 0.0 |
| **mAP (macro)** | **118** | **0.1025** | **0.0000** | **0.0000** |

## Reading it honestly

- **AP@0.25 and AP@0.5 are exactly zero for every label, both models.** Not one of the 118
  predicted boxes reached even IoU>=0.25 with its ground-truth box. This is a hard floor, not
  just a low average.
- **AP@0.1's only real signal is Cardiomegaly** (1.0000 flat, 0.4500 hierarchical), driven by a
  handful of hits ranked highly by model confidence. With n=11 this is a fragile estimate, not a
  robust one -- a perfect AP on 11 samples should not be over-interpreted.
- **Localization is effectively not usable as-is for either model** at standard detection
  thresholds. The classifiers work for image-level prediction (see `CLASSIFICATION_RESULTS.md`)
  but their Grad-CAM heatmaps do not reliably point at the labeled pathology region.
- **Open data-quality concern, not resolved:** the ground-truth Cardiomegaly boxes in
  `nih_bboxes_224.csv` span only 1.2-6.2% of image area, which is small relative to typical
  published NIH bbox statistics for Cardiomegaly (a large, central finding -- the visible
  cardiac silhouette). The original, un-rescaled NIH bbox file and raw images are not present
  locally, so the "_224" rescale that produced this file could not be independently re-verified.
  If that rescale has a bug, true IoU/mAP could be higher than reported here. Recommend checking
  this file's provenance before treating these numbers as a definitive localization benchmark.
- **Pneumonia scored exactly 0.0 IoU/IoBB for hierarchical across all 12 images** -- a flat zero,
  not just low, suggesting the hierarchical model rarely produces a confident, non-degenerate
  CAM for Pneumonia specifically. Worth a follow-up look if pursuing this further.

## Reproduction

Scripts used (not part of the repo, kept in the session scratchpad):
`localization_eval.py` (IoU/IoBB/Pointing Game) and `localization_map.py` (mAP). Both load the
real trained checkpoints and the real NIH bbox ground truth; no synthetic data was used.
