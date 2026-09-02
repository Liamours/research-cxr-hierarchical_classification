# research-cxr-hierarchical_classification

Multi-label chest X-ray classification over 51 Indonesian clinical-authority
diseases, pooled across 7 public datasets. DenseNet121 (TorchXRayVision
pretrained) is the only trained backbone -- DeiT-Base was in the original
two-backbone plan and dropped 2026-06-22 (`master_plan-260622.md`). Monte
Carlo Dropout uncertainty at inference, Grad-CAM explainability.

This is the codebase behind "Label Co-Annotation Limits Hierarchy-Consistency
Regularization in Pooled Multi-Label Chest Radiograph Classification," IEEE
iBioMed 2026 (paper 1571332442). Status: Accepted Pending TPC Approval,
decision received 2026-08-28 — the first-phase acceptance. The post-review
revision round is tracked in `draft/revision_1/revision_plan.md` (top-level
project directory, one level above this repo/ checkout). Trained and
evaluated on the real pooled seven-dataset corpus reported in the paper
(`draft/revision_1/main.tex`), not the synthetic-data smoke-test state the
older `report.md` describes.

Note: the Python package name is still `research-cxr` (see `pyproject.toml`,
unchanged since the 2026-07 directory rename -- cosmetic mismatch, not a
functional issue); the project directory and repository name is
`research-cxr-hierarchical_classification`. This checkout
(`repo/research-cxr-hierarchical_classification/`) is a plain file copy under
the real project directory, not a live git clone -- see `context/directory.md`
in the parent project directory for the full layout.

## Setup

```bash
uv sync --extra cu128   # RTX 4050, RTX 5070, recent NVIDIA (Ada + Blackwell)
uv sync --extra cu126   # older CUDA driver
uv sync --extra cpu     # no GPU / CI

uv run pytest           # fast test suite
uv run pytest -m slow   # training + loader tests
```

## Layout

```
src/
  config/       ExperimentConfig (one YAML drives a run)
  data/         label space, equivalence, transforms, dataset, loader, preprocess/
  model/        backbones (DenseNet121 only, DeiT dropped 2026-06-22), head, classifier, mc_dropout
  train/        losses (masked BCE), trainer
  evaluate/     metrics, calibration, selective prediction, evaluator
  inference/    predictor
  xai/          Grad-CAM, localization metrics
  script/       command-line entry points
configs/        experiment YAMLs, label_equivalence.json, dataset_registry.json
tests/          pytest suite
```

## Workflow

Data is not included in this checkout. The real pooled corpus (527,745 rows,
7 datasets) lives at `dataset/combined/combined.csv` in the top-level project
directory (one level above this repo/ checkout), preprocessed images under
`dataset/<name>/preprocessed/images/<name>/` per dataset -- see
`context/dataset_access.md` there. Each committed training config's
`data.label_csv` points at that file with an absolute path (fixed
2026-08-28 after a directory rename broke it; see `context/directory.md`).

Each dataset here was actually ingested via its own dedicated script
(`run_ingest_npy.py`, `run_ingest_covidx.py`, `run_ingest_tbx11k.py`,
`run_ingest_vindr_pcxr.py`, `run_ingest_metadata.py`; PadChest needed extra
DICOM + Spanish-report handling, see `dataset/padchest/preprocessing/`), not
the generic `run_preprocess.py` below -- that entry point exists and works,
but was not how any of the 7 pooled datasets were actually built; its
`--out-root` default (`data/preprocessed`) does not match the real
`dataset/<name>/preprocessed/` convention used everywhere else, so override
it explicitly if you use it.

```bash
# generic preprocess raw -> canonical (224x224x3 PNG + one column per canonical disease)
# override --out-root to match the dataset/<name>/preprocessed/ convention
uv run python src/script/run_preprocess.py --dataset nih-cxr14 \
    --raw-root <path-to-raw> --out-root dataset/nih-cxr14/preprocessed

# train (saves last / best-AUROC / best-loss / best-F1 checkpoints)
uv run python src/script/run_train.py --config configs/densenet121_xrv__flat__seed42.yaml

# evaluate a split; writes metrics + per-image predictions
uv run python src/script/run_evaluate.py --config <cfg> --split test

# single-image prediction (MC-Dropout if uq.method=mc_dropout in the config)
uv run python src/script/run_inference.py --config <cfg> --checkpoint <ckpt> --image <img>

# Grad-CAM overlay
uv run python src/script/run_xai.py --config <cfg> --checkpoint <ckpt> --image <img>
```

Each run writes to `result/<date>_<name>/`: config snapshot, logs, metrics,
checkpoints, predictions, xai overlays.

## Config

One YAML per run, nothing hardcoded (`experiment`, `data`, `aug`, `model`,
`seg`, `label`, `uq`, `training`, `eval`, `xai`, `paths`, `logging`; unknown
keys are rejected). Cross-dataset label mapping lives in
`configs/label_equivalence.json`, so adding a dataset needs no code change.
`data.loading` (`lazy` / `ram` / `gpu`) auto-downgrades to fit the device.
