# Dataset Access — for training

How to load the combined CXR corpus, the correct splits, and the label space.
Repo root: `C:/Rifqi/research-cxr`. Last updated after the PadChest merge + 27-label
mapping + stratified re-split.

---

## 1. Paths

### The one file training reads
```
dataset/combined/combined.csv
```
527,745 rows, one per image. This is what every config points at
(`data.label_csv`). It is regenerated from the per-dataset CSVs by
`src/script/run_combine_datasets.py` (absolutizes image paths, drops missing files,
drops cross-dataset duplicate image_ids).

### Per-dataset sources (7 ingested)
```
dataset/<name>/preprocessed/labels/<name>.csv     # canonical 55-col label CSV
dataset/<name>/preprocessed/images/<name>/*.png   # 224x224x3 uint8
```
`<name>` ∈ chexpert, covidx-cxr4, nih-cxr14, padchest, tbx11k, vindr-cxr, vindr-pcxr.
Registry (roots, status, provenance): `configs/dataset_registry.json`.

### Backups (revert points, do not train on)
```
dataset/combined/backups/combined_pre-padchest.csv     # before PadChest
dataset/combined/backups/combined_pre-resplit.csv      # before fair re-split
dataset/combined/backups/combined_pre-stratified.csv   # before stratified split
dataset/combined/backups/combined_24label.csv          # before the 3 extra labels
dataset/<name>/preprocessed/labels/<name>.csv.bak      # each dataset's ORIGINAL official split
```

---

## 2. CSV schema (55 columns)

```
image_id, image_path, dataset, split, <51 canonical label columns>
```
- `image_path` — absolute path to the 224x224 PNG.
- `dataset` — source dataset name.
- `split` — `train` | `val` | `test`.
- label columns — value ∈ {`1.0` present, `0.0` absent, `NaN` not-annotated by this dataset}.

**Masking rule (critical):** a `NaN` means the sample's dataset does not annotate that label.
It is NOT a negative. The loader turns non-NaN into `mask=1` and NaN into `mask=0`; losses and
metrics ignore `mask=0` entries entirely. So each sample only supervises the labels its dataset
actually annotated.

---

## 3. How to use

Training/eval go through a YAML config whose `data.label_csv` is `dataset/combined/combined.csv`
(already set in `configs/lambda/*.yaml` and `configs/grid/*.yaml`).

```
uv run python src/script/run_train.py --config configs/lambda/densenet121_xrv__lam0p5.yaml
```

Under the hood: `src/data/loader.py::build_loaders(cfg)` builds train/val/test `DataLoader`s from
`CxrClsDataset` (`src/data/dataset.py`), filtering by the `split` column. Transforms: grayscale →
resize/crop 224 → (train-only augment) → XRV normalize [−1024, 1024]. `data.num_workers: 12`,
`data.batch_size: 32` are the tuned defaults.

To use a subset of labels, set `label.conditions: [...]` in the config (empty = all 51 canonical).

---

## 4. Splits (correct usage)

| Split | Images | Use |
|---|--:|---|
| train | 440,080 | fit |
| val | 43,969 | checkpoint selection + early stop (model IS tuned on it — optimistic) |
| test | 43,696 | held-out; report final numbers here only |

**How the split was made (reproducible):** multi-label **iterative stratification**
(Sechidis 2011), `src/data/stratify.py`, applied per dataset by
`src/script/run_resplit_stratified.py`, 80/10/10, **seed 42**. It runs at **group level** where a
patient/study key is recoverable (nih & chexpert from the image_id prefix, covidx from the
`train_<N>` group, padchest via join to the raw `PADCHEST.csv` PatientID), so no patient's images
are split across train/val/test (**leakage-free — verified 0 spanning groups**). tbx11k / vindr-cxr
/ vindr-pcxr are ~1 image per unit → per-row.

To regenerate the splits (deterministic — same result): rerun `run_resplit_stratified.py` then
`run_combine_datasets.py`. Each dataset's `*.csv.bak` preserves its original official split.

### Images per dataset × split
| Dataset | Train | Val | Test |
|---|--:|--:|--:|
| chexpert | 161,920 | 12,890 | 12,864 |
| covidx-cxr4 | 71,690 | 6,644 | 6,483 |
| nih-cxr14 | 92,105 | 10,017 | 9,998 |
| padchest | 88,349 | 11,166 | 11,099 |
| tbx11k | 6,720 | 840 | 840 |
| vindr-cxr | 11,996 | 1,499 | 1,500 |
| vindr-pcxr | 7,300 | 913 | 912 |

---

## 5. Labels

- **51 canonical labels** total (`src/data/label_space.py::CANONICAL_LABELS`), Indonesian
  clinical-authority disease set (PDPI/PNPK/KKI).
- **27 currently have signal** (≥1 positive in the corpus). The other 24 have no annotating
  dataset (`datasets: {}` in the mapping) and are always NaN.
- Cross-dataset raw-finding → canonical mapping: `configs/label_equivalence.json` (consumed via
  `src/data/label_map.py::load_equivalence().dataset_to_canonical(<dataset>)`).

### Positives (=1) per label × split

| Label | Train | Val | Test |
|---|--:|--:|--:|
| Atelectasis | 39,479 | 4,014 | 4,072 |
| Cardiomegaly | 33,728 | 3,250 | 3,296 |
| COVID19_Pneumonia | 56,155 | 4,849 | 4,676 |
| Pleural_Effusion | 84,765 | 7,368 | 7,263 |
| Pulmonary_Edema | 45,752 | 3,745 | 3,794 |
| Pneumothorax | 17,909 | 1,795 | 1,961 |
| Solitary_Pulmonary_Nodule | 16,776 | 2,074 | 2,083 |
| COPD | 13,596 | 1,863 | 1,814 |
| Pneumonia | 9,145 | 1,123 | 1,122 |
| ILD | 7,069 | 904 | 840 |
| Chest_Trauma | 5,773 | 635 | 612 |
| Diaphragmatic_Hernia | 1,454 | 189 | 196 |
| Bronchiectasis | 1,247 | 150 | 152 |
| Acute_Bronchitis | 813 | 102 | 101 |
| Tuberculosis | 785 | 96 | 106 |
| IPF | 565 | 83 | 67 |
| Post_TB_Obstructive_Syndrome | 512 | 61 | 66 |
| Bronchiolitis | 469 | 59 | 59 |
| Pulmonary_Metastases | 182 | 20 | 21 |
| Subcutaneous_Emphysema | 146 | 15 | 15 |
| Pulmonary_Hypertension | 88 | 11 | 11 |
| Airway_Foreign_Body | 43 | 5 | 4 |
| Asbestosis | 33 | 4 | 5 |
| Hydropneumothorax | 22 | 3 | 3 |
| Mediastinal_Tumor | 7 | 1 | 1 |
| Lung_Cancer | 4 | 1 | **0** |
| Pleural_Empyema | 3 | **0** | **0** |

**25 of 27 labels are evaluable** (val + test both have positives). **2 are not**, purely from
data scarcity, not a split defect: **Lung_Cancer** (5 positives total) and **Pleural_Empyema**
(3 total) fall below the ~10-positive floor needed to fill a 10% split. They still train (masked)
but cannot be scored reliably. Asbestosis (42) and Airway_Foreign_Body (52) are evaluable but
small — treat their test metrics as high-variance.

### Which dataset supplies each label (quick reference)
NIH: Atelectasis, Cardiomegaly, Pulmonary_Edema, Pleural_Effusion, Pneumothorax, Pneumonia,
COPD, ILD, Solitary_Pulmonary_Nodule, Diaphragmatic_Hernia · CheXpert: Atelectasis, Cardiomegaly,
Pulmonary_Edema, Pleural_Effusion, Pneumothorax, Pneumonia, Solitary_Pulmonary_Nodule,
Chest_Trauma · COVIDx: COVID19_Pneumonia · TBX11K: Tuberculosis · VinDr-CXR: Atelectasis,
Cardiomegaly, Pleural_Effusion, Pneumothorax, Solitary_Pulmonary_Nodule, ILD · VinDr-PCXR:
Pneumonia, Tuberculosis, Bronchiolitis, Acute_Bronchitis, Diaphragmatic_Hernia, Mediastinal_Tumor,
Lung_Cancer · PadChest: 21 labels incl. the rare ones (Bronchiectasis, Post_TB_Obstructive_Syndrome,
Hydropneumothorax, Pulmonary_Metastases, Pulmonary_Hypertension, Subcutaneous_Emphysema,
Pleural_Empyema, IPF, Asbestosis, Airway_Foreign_Body).

See [`HIERARCHY_AND_MODELS.md`](HIERARCHY_AND_MODELS.md) for the label hierarchy, loss options,
and model spec.
