"""Canonical label space — the official Indonesian clinical-authority disease set.

CANONICAL_LABELS is the 51-disease canonical set defined by Indonesian clinical
authority (PDPI PUPK 2021 / Kewenangan Klinis, Kemenkes PNPK, KKI SKDI). This is
a FLAT multi-label set — there is no grouping or hierarchy. Full per-disease
provenance: context/design_notes/canonical_diseases-260618.md (top-level project
directory, one level above repo/research-cxr-hierarchical_classification/)

All downstream dimensions derive from len(CANONICAL_LABELS); no other file
hardcodes the label count.

NIH_LABEL_MAP / VINDR_LABEL_MAP (and CHEXPERT_LABEL_MAP in preprocess/common.py)
map a dataset's raw radiographic-finding column to the closest canonical disease.
Public datasets annotate radiographic FINDINGS while the canonical set is clinical
DIAGNOSES, so the mapping is deliberately lossy: only defensible finding->disease
correspondences are kept; findings with no clear disease home are left unmapped,
and most canonical diseases have no signal in the public datasets (written NaN /
masked, never 0). The operational map is configs/label_equivalence.json; these
dicts are the fallback / drift reference.
"""

from __future__ import annotations

CANONICAL_LABELS: list[str] = [
    "Pneumonia", "Tuberculosis", "NTM_Infection", "Lung_Abscess",
    "Pleural_Empyema", "Pulmonary_Mycosis", "Melioidosis", "Bronchiolitis",
    "Mediastinitis", "Acute_Bronchitis", "COVID19_Pneumonia",
    "Aspiration_Pneumonia", "Other_Viral_Pneumonia",
    "COPD", "Asthma", "Bronchiectasis", "Post_TB_Obstructive_Syndrome", "ACOS",
    "Pleural_Effusion", "Pneumothorax", "Hemothorax", "Hydropneumothorax",
    "Pneumomediastinum",
    "Lung_Cancer", "Benign_Lung_Tumor", "Pulmonary_Metastases",
    "Mediastinal_Tumor", "Mesothelioma", "Pleural_Malignancy",
    "Chest_Wall_Tumor", "Solitary_Pulmonary_Nodule",
    "ILD", "IPF", "Sarcoidosis", "COP", "Hypersensitivity_Pneumonitis",
    "Silicosis", "Asbestosis", "Other_Pneumoconiosis",
    "Pulmonary_Edema", "ARDS", "Pulmonary_Embolism", "Pulmonary_Hypertension",
    "Cor_Pulmonale", "Cardiomegaly",
    "Chest_Trauma", "Subcutaneous_Emphysema", "Diaphragmatic_Hernia",
    "Chest_Wall_Abnormality", "Airway_Foreign_Body", "Atelectasis",
]

# Dataset raw finding -> canonical disease (lossy; see module docstring). Only
# defensible finding->disease correspondences; unmapped raw findings are not
# annotated for any canonical and become NaN. Mirrors configs/label_equivalence.json.
NIH_LABEL_MAP: dict[str, str] = {
    "Atelectasis": "Atelectasis",
    "Cardiomegaly": "Cardiomegaly",
    "Edema": "Pulmonary_Edema",
    "Effusion": "Pleural_Effusion",
    "Pneumonia": "Pneumonia",
    "Pneumothorax": "Pneumothorax",
    "Emphysema": "COPD",               # emphysema = radiographic hallmark of COPD
    "Fibrosis": "ILD",                 # pulmonary fibrosis -> ILD umbrella
    "Mass": "Solitary_Pulmonary_Nodule",
    "Nodule": "Solitary_Pulmonary_Nodule",
    "Hernia": "Diaphragmatic_Hernia",
}

VINDR_LABEL_MAP: dict[str, str] = {
    "Atelectasis": "Atelectasis",
    "Cardiomegaly": "Cardiomegaly",
    "ILD": "ILD",
    "Pulmonary fibrosis": "ILD",
    "Nodule/Mass": "Solitary_Pulmonary_Nodule",
    "Pleural effusion": "Pleural_Effusion",
    "Pneumothorax": "Pneumothorax",
}

# VinDr-PCXR pediatric labels -> canonical. Multiple raw labels can map to the
# same canonical (OR logic: any source=1 → canonical=1). Unmapped pediatric-only
# conditions (CPAM, Hyaline membrane disease, Situs inversus, Congenital emphysema,
# Other disease, No finding) are left as NaN — not applicable.
VINDR_PCXR_LABEL_MAP: dict[str, str] = {
    "Bronchitis": "Acute_Bronchitis",
    "Brocho-pneumonia": "Pneumonia",
    "Bronchiolitis": "Bronchiolitis",
    "Pneumonia": "Pneumonia",
    "Pleuro-pneumonia": "Pneumonia",
    "Diagphramatic hernia": "Diaphragmatic_Hernia",
    "Tuberculosis": "Tuberculosis",
    "Mediastinal tumor": "Mediastinal_Tumor",
    "Lung tumor": "Lung_Cancer",
}
