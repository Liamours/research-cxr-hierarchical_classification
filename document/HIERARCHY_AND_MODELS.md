# Hierarchy & Models — reference for new experiments

Everything an experiment needs to (a) reuse the label hierarchy and (b) understand
the current model + loss options. All paths are repo-relative to `C:/Rifqi/research-cxr`.

---

## 1. Label hierarchy

The model output is **flat** — a 51-way sigmoid head, one logit per canonical label.
The hierarchy is **not** an architectural tree; it is a set of parent→child IS-A edges
used only to (a) define a training-time consistency penalty (HBCE loss) and (b) drive an
optional inference-time fallback. Editing the edges changes both.

**Source of truth:** [`src/data/hierarchy.py`](../src/data/hierarchy.py) → `PARENT_CHILD_EDGES`
(13 edges). The docstring is explicit that these are clinically-motivated modelling edges,
**not** an official PDPI/PNPK/KKI taxonomy.

```
Pneumonia
├── COVID19_Pneumonia
├── Aspiration_Pneumonia
└── Other_Viral_Pneumonia

ILD (Interstitial Lung Disease)
├── IPF
├── COP
├── Hypersensitivity_Pneumonitis
├── Silicosis
├── Asbestosis
├── Other_Pneumoconiosis
└── Sarcoidosis

Tuberculosis
└── Post_TB_Obstructive_Syndrome

Pleural_Effusion
└── Pleural_Empyema

Pulmonary_Hypertension
└── Cor_Pulmonale
```

5 parents, 13 children, 18 labels touched (of 51). The other 33 labels are flat (no edge).

### How the edges are consumed

- `edge_index_pairs(conditions)` → list of `(parent_col_idx, child_col_idx)` for the edges
  whose **both** endpoints are in the active label set. Edges with a missing endpoint are
  silently skipped.
- `validate_edges(conditions)` → warnings for edges whose parent/child is absent.

### To add / change a hierarchy edge

Edit the `PARENT_CHILD_EDGES` list in `src/data/hierarchy.py`. Both labels must be members of
`CANONICAL_LABELS` (`src/data/label_space.py`). No other file needs changing — the loss and
the fallback both read this list. Note: an edge only has training effect if **the child label
has data** (positives) in the corpus; edges whose child is always-masked never fire.

### Two distinct places the hierarchy shows up

| Mechanism | File | When active | Effect |
|---|---|---|---|
| **HBCE training penalty** | `src/train/losses.py` `HBCELoss` | `label.label_structure: hierarchical`, `lam > 0` | Adds `lam · mean_edges[relu(child−0.5)·relu(0.5−parent)]` to the loss. Zero unless the model predicts child≥0.5 while parent<0.5. |
| **Inference fallback** | `src/inference/hierarchical_fallback.py` | `uq.method: mc_dropout` AND `label_structure: hierarchical` | Suppresses a child prediction when its MC-Dropout variance > `uq.gate_threshold`. |

`lam` (the sweep variable) governs **only** the training penalty. The fallback is a separate
inference gate (`uq.gate_threshold`, default 0.003). On current data the fallback never fires
(MC variance ≈ 1e-4 ≪ threshold) — kept for completeness, not a load-bearing component.

---

## 2. Loss options (all mask-aware, multi-label)

Factory: `build_loss(label_structure, conditions, lam=0.5, *, bafl_weights=None, ...)`
in [`src/train/losses.py`](../src/train/losses.py). Selected by `label.label_structure` in the config.

| `label_structure` | Class | Formula / idea | Extra config |
|---|---|---|---|
| `flat` | `MaskedBCELoss` | Masked BCE. Not-applicable labels (`mask==0`) excluded from num + denom. | — |
| `hierarchical` | `HBCELoss` | `BCE + lam · penalty` (Asadi 2025 CIHMLC). `lam=0` ≡ flat. | `label.lam` |
| `bafl` | `BAFLLoss` | Balanced Adaptive Focal Loss (HP-ViT, Khan 2026 Sect. 3.3): per-class effective-number weight (Cui 2019) × focal `(1−p_t)^γ`, γ ramps `gamma_init→gamma_final` over `t_warmup` epochs. | `label.bafl_beta/bafl_gamma_init/bafl_gamma_final/bafl_t_warmup` |

Masking convention (all losses): a `(sample, label)` entry contributes only where that label is
annotated for the sample's dataset (`mask==1`). Non-applicable labels carry `label=0, mask=0`
and never produce gradient.

BAFL class weights come from `train_class_weights(label_csv, conditions, beta)` — computed from
the **train split only**, normalized to mean 1. The trainer calls `set_epoch(epoch)` each epoch
to advance γ.

---

## 3. Current model

| Item | Value |
|---|---|
| Backbone | DenseNet121, TorchXRayVision `densenet121-res224-all` (CXR-pretrained on 6 datasets) |
| Head | `nn.Sequential(nn.Dropout(0.2), nn.Linear(1024, 51))` — flat 51-way, sigmoid |
| Total params | 7,018,309 (backbone 6,966,034 + head 52,275) |
| Compute | 2.755 GMACs @ 224×224, ~28 MB fp32 |
| Input | (B, 1, 224, 224) grayscale, XRV-normalized to [−1024, 1024] |
| Output | (B, 51) logits → per-label sigmoid (labels are NOT mutually exclusive) |

Backbone builder: `src/model/backbones.py`; assembly + param groups: `src/model/classifier.py`
(`build_model_from_cfg`, `model_profile`). Only this backbone is wired.

### Fine-tuning protocol (`src/train/trainer.py`, config `training:`)

- Transfer learning: pretrained backbone + randomly-initialized 51-way head.
- Differential LR: backbone `1e-5`, head `1e-4` (param groups).
- Backbone **frozen for epoch 1** (`freeze_backbone_epochs: 1`), then unfrozen.
- OneCycleLR cosine, AdamW, weight decay 0.01, bf16 autocast, grad clip 1.0.
- Early stop patience 3 on `val_loss`. Seed 42. `epochs: 15`, `batch_size: 32`.
- Checkpoints saved: `last`, `best_val_loss`, `best_val_auroc_macro`, `best_val_f1_macro`,
  `best_val_aurc_macro` (+ `*_meta.json`).

### Uncertainty (optional, not in the core ablation)

MC-Dropout: keep dropout active at inference, `uq.mc_passes` stochastic forward passes → per-label
(mean, variance). `mc_passes=0` during training-time validation (deterministic). It can be layered
on any trained model at inference without retraining.

---

## 4. How to run an experiment / ablation

The current ablation axis is a **λ sweep** (hierarchy strength): `λ ∈ {0, 0.1, 0.5, 1.0}`, where
`λ=0` is the flat baseline. Configs live in `configs/lambda/densenet121_xrv__lam{0p0,0p1,0p5,1p0}.yaml`.

```
uv run python src/script/run_train.py --config configs/lambda/densenet121_xrv__lam0p5.yaml
uv run python src/script/run_evaluate.py --config <same> --split test --run-dir result/<run_dir>
```

To try a **new loss** (e.g. BAFL): set `label.label_structure: bafl` in a config and tune
`label.bafl_*`. To try **new hierarchy edges**: edit `src/data/hierarchy.py`. To try a **new
backbone**: add it in `src/model/backbones.py` and reference it via `model.backbone`. The dataset,
splits, and label space stay fixed across these, so results are comparable.

See [`DATASET_ACCESS.md`](DATASET_ACCESS.md) for the corpus, splits, and label space.
