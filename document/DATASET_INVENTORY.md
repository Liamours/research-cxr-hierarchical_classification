# Dataset Inventory — canonical table

Canonical, cross-chat reference for what's in `dataset/` locally vs. what each dataset
officially contains. **This file is the single source for this table** — when asked to
add a column/dataset, edit this file in place rather than rebuilding it, and re-verify any
new figures against the web (official paper / homepage / PhysioNet / Kaggle) before adding
them. Registry detail (per-dataset ingestion notes, label mappings): `configs/dataset_registry.json`.

Last verified against the web: 2026-07-16.

| Dataset | Status | Official Total (images) | Official Patients | Official Splits | Our Ingested (rows) | Local Path | Primary Source |
|---|---|--:|--:|---|--:|---|---|
| VinDr-CXR | ingested | 18,000 | — | 15,000 train / 3,000 test | 15,000 | `dataset/vindr-cxr` | [PhysioNet](https://physionet.org/content/vindr-cxr/1.0.0/) · [paper](https://arxiv.org/abs/2012.15029) |
| NIH ChestX-ray14 | ingested | 112,120 | 30,805 | no official split | 112,120 | `dataset/nih-cxr14` | [NIH Box](https://nihcc.app.box.com/v/ChestXray-NIHCC) · [paper](https://arxiv.org/abs/1705.02315) |
| CheXpert(+) | ingested | 224,316 (190,498 frontal + 32,294 lateral)¹ | 65,240 | no official split | 187,674 | `dataset/chexpert` | [Stanford AIMI](https://stanfordmlgroup.github.io/competitions/chexpert/) · [paper](https://arxiv.org/abs/1901.07031) |
| TBX11K | ingested | 11,200 | — | 6,600 train / 1,800 val / 2,800 test | 8,400² | `dataset/tbx11k` | [CVPR 2020 paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Liu_Rethinking_Computer-Aided_Tuberculosis_Diagnosis_CVPR_2020_paper.html) |
| VinDr-PCXR | ingested | 9,125 | — | 7,728 train / 1,397 test | 9,125 | `dataset/vindr-pcxr` | [PhysioNet](https://physionet.org/content/vindr-pcxr/1.0.0/) · [paper](https://www.nature.com/articles/s41597-022-01498-w) |
| COVIDx CXR-4 | ingested | 84,818 (65,681 pos / 19,137 neg) | 45,342 | no official split | 84,817³ | `dataset/covidx-cxr4` | [Kaggle](https://www.kaggle.com/datasets/andyczhao/covidx-cxr2) · [paper](https://arxiv.org/abs/2311.17677) |
| PadChest | ingested | ~160,000⁴ | 67,000 | no official split | 110,614⁵ | `dataset/padchest` | [paper](https://arxiv.org/abs/1901.07441) · [BIMCV homepage](https://bimcv.cipf.es/bimcv-projects/padchest/) |

**Combined corpus:** `dataset/combined/combined.csv` — 527,745 rows (sum of ingested rows above,
minus 5 cross-dataset duplicate `image_id`s dropped by `run_combine_datasets.py`).

Footnotes:
1. Frontal/lateral split varies slightly by CheXpert release version (191,027/32,387 seen
   elsewhere). We ingest **CheXpert+ (2024 re-annotated)**, not the original 2019 release —
   see `configs/dataset_registry.json` note.
2. TBX11K test set (2,800–3,302 depending on release) is unlabeled; excluded from ingest.
3. One image file missing from our downloaded archive vs. official 84,818.
4. Paper abstract rounds to "more than 160,000"; exact figures vary by source (some cite
   168,861 images / 109,931 studies). Our `official_total: 160,861` in the registry is the
   raw `PADCHEST.csv` manifest row count, not a paper-stated figure — treat as approximate
   pending a manifest-level recheck.
5. Frontal-only (PA/AP/AP_horizontal) ingested from the 160,861-row manifest.
