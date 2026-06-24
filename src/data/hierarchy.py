"""Clinical hierarchy edges for HBCE training.

Defines parent→child edges as strict IS-A subtype relationships grounded in
standard medical taxonomy. These are NOT a published official PDPI/PNPK/KKI
hierarchy — they are clinically-motivated modelling edges for the HBCE loss.

The 51-label canonical set remains flat (all labels predicted independently);
these edges only add a training-time penalty when child_prob > 0.5 while
parent_prob < 0.5 (hierarchy violation in the loss function).

All labels must be members of CANONICAL_LABELS (src/data/label_space.py).
"""

from __future__ import annotations

# (parent, child): "child IS-A subtype of parent"
# HBCE penalty fires when child confidently positive but parent negative.
PARENT_CHILD_EDGES: list[tuple[str, str]] = [
    # Pneumonia subtypes
    ("Pneumonia", "COVID19_Pneumonia"),
    ("Pneumonia", "Aspiration_Pneumonia"),
    ("Pneumonia", "Other_Viral_Pneumonia"),
    # ILD subtypes (ILD = umbrella for all interstitial lung disease patterns)
    ("ILD", "IPF"),
    ("ILD", "COP"),
    ("ILD", "Hypersensitivity_Pneumonitis"),
    ("ILD", "Silicosis"),
    ("ILD", "Asbestosis"),
    ("ILD", "Other_Pneumoconiosis"),
    ("ILD", "Sarcoidosis"),
    # TB sequela (SOPT always follows prior TB)
    ("Tuberculosis", "Post_TB_Obstructive_Syndrome"),
    # Pleural space (empyema = infected pleural effusion)
    ("Pleural_Effusion", "Pleural_Empyema"),
    # Vascular (cor pulmonale = RV failure secondary to pulmonary hypertension)
    ("Pulmonary_Hypertension", "Cor_Pulmonale"),
]


def edge_index_pairs(conditions: list[str]) -> list[tuple[int, int]]:
    """Return (parent_idx, child_idx) index pairs for edges present in conditions.

    Edges whose parent or child is absent from conditions are silently skipped
    (e.g. when running a subset of labels in ablation or single-dataset mode).
    """
    idx = {label: i for i, label in enumerate(conditions)}
    return [
        (idx[p], idx[c])
        for p, c in PARENT_CHILD_EDGES
        if p in idx and c in idx
    ]


def validate_edges(conditions: list[str]) -> list[str]:
    """Return list of warning strings for edges with missing labels."""
    label_set = set(conditions)
    warnings = []
    for p, c in PARENT_CHILD_EDGES:
        if p not in label_set:
            warnings.append(f"parent '{p}' not in conditions — edge ({p}→{c}) skipped")
        if c not in label_set:
            warnings.append(f"child '{c}' not in conditions — edge ({p}→{c}) skipped")
    return warnings
