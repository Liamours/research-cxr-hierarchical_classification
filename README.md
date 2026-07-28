# research-cxr

Multi-label chest X-ray classification over 51 Indonesian clinical-authority
diseases. DenseNet121 and DeiT-Base backbones, Monte Carlo Dropout uncertainty,
Grad-CAM explainability.

Status: code-complete, tested end to end on synthetic data. See `report.md`.

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
  model/        backbones (DenseNet121, DeiT), head, classifier, mc_dropout
  train/        losses (masked BCE), trainer
  evaluate/     metrics, calibration, selective prediction, evaluator
  inference/    predictor
  xai/          Grad-CAM, localization metrics
  script/       command-line entry points
configs/        experiment YAMLs, label_equivalence.json, dataset_registry.json
tests/          pytest suite
```

## Workflow

Data is not included. Place a raw dataset under `data/raw/<dataset>/`, then:

```bash
# preprocess raw -> canonical (224x224x3 PNG + one column per canonical disease)
uv run python src/script/run_preprocess.py --dataset nih-cxr14 \
    --raw-root data/raw/nih-cxr14 --out-root data/preprocessed

# train (saves last / best-AUROC / best-loss / best-F1 checkpoints)
uv run python src/script/run_train.py --config configs/flat_densenet_baseline.yaml

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
