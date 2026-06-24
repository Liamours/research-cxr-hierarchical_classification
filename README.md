# research-cxr

Flat multi-label chest X-ray classification over the official Indonesian
clinical-authority disease set (51 diseases), comparing a convolutional backbone
(DenseNet121) and a transformer backbone (DeiT-Base), with Monte Carlo Dropout
uncertainty and Grad-CAM explainability.

Status: code-complete and tested on synthetic data; not yet run on a real
dataset. See `report.md` for the project report and
`document/ibiomed_experiment_protocol-2606.md` for the frozen experiment plan.

## Requirements

- Python 3.11+ and [uv](https://docs.astral.sh/uv/). This project uses uv only
  (no conda).
- A CUDA GPU is optional but recommended (tested on RTX 4050 6 GB and 5070 12 GB).

torch/torchvision are picked per device with an extra (not installed by a bare
`uv sync`). See `document/device_setup-2606.md` for CPU / RunPod / Mac details.

```bash
uv sync --extra cu128   # RTX 4050, RTX 5070, recent NVIDIA (Ada + Blackwell)
uv sync --extra cu126   # older CUDA driver
uv sync --extra cpu     # no GPU / CI
# cloud zero-config: uv pip install torch torchvision --torch-backend=auto

uv run pytest           # fast test suite
uv run pytest -m slow   # training + loader tests
```

## Layout

```
configs/                experiment YAMLs, label_equivalence.json
src/
  config/               ExperimentConfig (one YAML drives a run)
  data/                 label space, equivalence, transforms, dataset,
                        loader (lazy/ram/gpu), preprocess/ (per-dataset adapters)
  model/                backbones (DenseNet121, DeiT), head (flat multi-label),
                        classifier, mc_dropout
  train/                losses (masked BCE), trainer
  evaluate/             metrics, calibration, selective, evaluator
  inference/            predictor
  xai/                  Grad-CAM, localization metrics
  eda/                  dataset statistics
  script/               command-line entry points
  util/                 seed, logging, registry, device
tests/                  pytest suite
```

## Configuration

One YAML configures a run; nothing is hardcoded. Sections: `experiment`, `data`,
`aug`, `model`, `seg`, `label`, `uq`, `training`, `eval`, `xai`, `paths`, `logging`.
Cross-dataset label equivalence (`configs/label_equivalence.json`) is an external
data file, so new datasets need no code change. Unknown keys are rejected.

Baselines: `configs/flat_deit_baseline.yaml`, `configs/flat_densenet_baseline.yaml`.
The 16-condition grid lives in `configs/grid/`.

## Workflow

Data is not included. Place a raw dataset under `data/raw/<dataset>/`, then:

```bash
# 1. inspect cross-dataset label mapping for a raw label file
uv run python src/script/run_label_audit.py --raw-csv <raw_labels.csv> --dataset nih-cxr14

# 2. preprocess raw -> canonical (224x224x3 PNG + one column per canonical disease)
uv run python src/script/run_preprocess.py --dataset nih-cxr14 \
    --raw-root data/raw/nih-cxr14 --out-root data/preprocessed

# 3. exploratory analysis (class distribution, leakage, image integrity)
uv run python src/script/run_eda.py --config configs/flat_densenet_baseline.yaml

# 4. train (saves last / best-AUROC / best-loss / best-F1 checkpoints)
uv run python src/script/run_train.py --config configs/flat_densenet_baseline.yaml

# 5. evaluate any split; writes metrics + per-image predictions
uv run python src/script/run_evaluate.py --config <cfg> --split test

# 6. single-image prediction (point estimate, or MC-Dropout if uq.method=mc_dropout)
uv run python src/script/run_inference.py --config <cfg> --checkpoint <ckpt> --image <img>

# 7. Grad-CAM overlay
uv run python src/script/run_xai.py --config <cfg> --checkpoint <ckpt> --image <img>

# 8. generate / run the 16-condition grid
uv run python src/script/run_grid.py --base-config configs/flat_deit_baseline.yaml --generate
uv run python src/script/run_grid.py --base-config configs/flat_deit_baseline.yaml
```

Each run writes to `result/<date>_<name>/`: `config.yaml` snapshot, `run.log`,
`events.jsonl`, `train_log.csv`, `metrics.json`, `checkpoints/`, `predictions/`,
`xai/`.

## Method

Flat multi-label classification over the 51 official Indonesian clinical-authority
diseases (see context/research-cxr_canonical-diseases-2606.md). A shared backbone
feeds one linear head with 51 sigmoid outputs; masked BCE supervises only the
labels a dataset provides (unannotated diseases are masked, never trained). Monte
Carlo Dropout (evaluation/inference only) gives a predictive variance per disease
as an uncertainty signal, and Grad-CAM provides localization. The comparison of
interest is DenseNet121 versus DeiT-Base under one identical pipeline.

## Hardware and loading

`data.loading` selects `lazy` (default), `ram`, or `gpu` caching; it
auto-downgrades to a feasible mode when a cache would not fit the device, so the
same config runs on different GPUs. `bf16` plus gradient accumulation keep memory
within a 6 GB budget.
```
