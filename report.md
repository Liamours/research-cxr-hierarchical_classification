# Flat Multi-Label Chest X-Ray Classification — Project Report

## Status (be explicit)

This repository is code-complete and tested on synthetic data. It has not yet
been run on a real dataset, so there are no real accuracy numbers in this report.
Every component (preprocessing, data loading, training, evaluation, inference,
uncertainty, explainability) is verified end to end with dummy data through a
committed test suite. The next step is to run the pre-registered experiment
protocol on NIH ChestX-ray14.

(An earlier version of this project did chest X-ray report generation. That code
was moved out of the repository; this report describes the current
classification system.)

## Overview

The system performs flat multi-label classification of frontal chest radiographs
into the official Indonesian clinical-authority disease set (51 diseases grounded
in PDPI/PNPK/KKI guidelines), and adds
two things on top of a plain classifier:

1. a comparison of a convolutional backbone (DenseNet121) and a transformer
   backbone (DeiT-Base) under one identical pipeline;
2. uncertainty estimation (Monte Carlo Dropout) and Grad-CAM explainability used
   to support, not replace, the accuracy numbers.

## Datasets

Training uses a combined pool of 6 datasets (NIH ChestX-ray14, CheXpert+,
VinDr-CXR, VinDr-PCXR, TBX11K, COVIDx-CXR4) via a single 55-column canonical
CSV (417,136 images). See `configs/dataset_registry.json` for per-dataset
image counts and annotated label sets. Multi-source pooling is enabled by
masked BCE: each image is
supervised only on its source dataset's applicable labels; unannotated labels
stay NaN-masked and receive no gradient. Cross-dataset label names are
reconciled through `configs/label_equivalence.json`, guarded by a test that
keeps it in sync with the preprocessing maps.

## Preprocessing

One canonical output for every input: frontal view selected (PA preferred),
grayscale, shorter side resized to 256, center-cropped to 224, replicated to
three channels, saved as 8-bit PNG. DICOM, JPEG, and PNG inputs are all handled
(DICOM gets rescale, MONOCHROME1 inversion, and percentile clipping).
Normalization is applied at load time and is backbone specific (ImageNet for
DeiT, TorchXRayVision for DenseNet121). Uncertain and not-mentioned labels are
mapped to 0 (U-Zero). Splits are patient-level.

## Models and training

DenseNet121 (TorchXRayVision, chest-X-ray pretrained) and DeiT-Base (ImageNet
pretrained), each with a flat multi-label head (51 sigmoid outputs). Training
uses AdamW with differential learning
rates (backbone 1e-5, head 1e-4), one-cycle cosine schedule, bfloat16 mixed
precision, gradient accumulation, a one-epoch backbone freeze, and early
stopping. Four checkpoints are saved per run (last, best validation loss, best
validation macro AUROC, best validation macro F1).

## Method: flat multi-label classification with uncertainty

A shared backbone feeds one linear head producing 51 logits, each followed by a
sigmoid. Masked binary cross-entropy supervises only the labels a dataset
provides; diseases a dataset does not annotate are masked out (no gradient), so
the 51-label head trains correctly even when each source covers a different
subset. At inference, Monte Carlo Dropout (dropout kept active over N passes)
gives a predictive mean and variance per disease; the variance is reported as an
uncertainty signal and is sanity-checked (wrong predictions should be more
uncertain than correct ones). The comparison of interest is DenseNet121 versus
DeiT-Base under one identical pipeline.

## Evaluation

Discrimination (macro/micro AUROC and F1, mask-aware), calibration (Expected
Calibration Error and reliability diagrams), selective prediction (risk-coverage
curve and area under it), and localization (Grad-CAM with pointing game and IoU
on the bounding-box subset). Evaluation runs on any split and saves per-image
predictions to a per-split CSV.

## Configuration, logging, reproducibility

Every experiment is one YAML config consumed by `ExperimentConfig`. The cross-
dataset label equivalence is an external data file, not code. Each run fixes
seed 42, snapshots its config, and logs per-epoch metrics, a structured event
stream, and final metrics. Data loading is device-aware: it supports lazy, RAM,
and GPU caching and auto-downgrades if a cache will not fit, so the same config
runs on an RTX 4050 (6 GB) and a 5070 (12 GB).

## How to run

See `README.md` for commands. In short: preprocess, run EDA, train, evaluate,
infer, and produce Grad-CAM overlays, all through the scripts in `src/script/`.

## Tests

`uv run pytest` runs the fast suite; `uv run pytest -m slow` runs training and
loader tests. The suite covers config, data, preprocessing, models, losses,
metrics, uncertainty, and explainability, plus
behavioral checks (overfit a single batch, determinism, gradient flow, data
leakage detection, no-refit at evaluation).
