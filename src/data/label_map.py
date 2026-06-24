"""Cross-dataset label equivalence (resolves dataset-specific names + synonyms
to the canonical labels).

Loads configs/label_equivalence.json. This is a separate definition file: the
original dataset files and label CSVs are never modified. If the JSON is absent,
a default table is built from the existing label_space / preprocess maps so
behavior stays consistent. Used by EDA (coverage report) and by preprocessing to
map raw dataset columns to canonical.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.data.label_space import CANONICAL_LABELS


def _norm(s: str) -> str:
    s = str(s).lower().strip().replace("_", " ")
    return " ".join("".join(ch for ch in s if ch.isalnum() or ch == " ").split())


class LabelEquivalence:
    def __init__(self, table: dict):
        self.table = table  # canonical -> {"synonyms": [...], "datasets": {ds: name | [name, ...]}}
        self._by_norm: dict[str, str] = {}
        for canon, e in table.items():
            self._by_norm[_norm(canon)] = canon
            for syn in e.get("synonyms", []):
                self._by_norm.setdefault(_norm(syn), canon)
            for names in e.get("datasets", {}).values():
                for name in (names if isinstance(names, list) else [names]):
                    self._by_norm.setdefault(_norm(name), canon)

    def to_canonical(self, name: str, dataset: str | None = None) -> str | None:
        if dataset:
            for canon, e in self.table.items():
                names = e.get("datasets", {}).get(dataset)
                if names == name or (isinstance(names, list) and name in names):
                    return canon
        return self._by_norm.get(_norm(name))

    def dataset_name(self, canonical: str, dataset: str) -> str | list[str] | None:
        return self.table.get(canonical, {}).get("datasets", {}).get(dataset)

    def dataset_to_canonical(self, dataset: str) -> dict[str, str]:
        """raw column name -> canonical, for one dataset. This is the map a
        preprocessing adapter consumes, sourced from the equivalence file so the
        mapping is no longer hardcoded per adapter. A canonical with multiple raw
        names for this dataset (e.g. Solitary_Pulmonary_Nodule <- Mass + Nodule) expands to one
        entry per raw name."""
        out: dict[str, str] = {}
        for canon, e in self.table.items():
            names = e.get("datasets", {}).get(dataset)
            if names is None:
                continue
            for name in (names if isinstance(names, list) else [names]):
                out[name] = canon
        return out

    def synonyms(self, canonical: str) -> list[str]:
        return self.table.get(canonical, {}).get("synonyms", [])

    def coverage(self, raw_names, dataset: str | None = None) -> dict:
        """For a dataset's raw label column names: which map to canonical, which
        do not, and which canonical labels remain uncovered."""
        mapped, unmapped = {}, []
        for n in raw_names:
            c = self.to_canonical(n, dataset)
            if c:
                mapped[n] = c
            else:
                unmapped.append(n)
        covered = set(mapped.values())
        return {
            "dataset": dataset,
            "mapped": mapped,
            "unmapped": unmapped,
            "canonical_covered": sorted(covered),
            "canonical_missing": [c for c in CANONICAL_LABELS if c not in covered],
        }

    @classmethod
    def load(cls, path) -> "LabelEquivalence":
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["canonical"])

    @classmethod
    def default(cls) -> "LabelEquivalence":
        from src.data.label_space import NIH_LABEL_MAP, VINDR_LABEL_MAP
        from src.data.preprocess.common import CHEXPERT_LABEL_MAP
        table = {c: {"synonyms": [], "datasets": {}} for c in CANONICAL_LABELS}

        def add(canon: str, dataset: str, raw_name: str) -> None:
            slot = table[canon]["datasets"]
            if dataset not in slot:
                slot[dataset] = raw_name
            elif isinstance(slot[dataset], list):
                slot[dataset].append(raw_name)
            else:
                slot[dataset] = [slot[dataset], raw_name]

        for col, canon in CHEXPERT_LABEL_MAP.items():
            add(canon, "mimic-cxr", col)
            add(canon, "chexpert", col)
        for col, canon in NIH_LABEL_MAP.items():
            add(canon, "nih-cxr14", col)
        for col, canon in VINDR_LABEL_MAP.items():
            add(canon, "vindr-cxr", col)
        return cls(table)


def load_equivalence(path: str = "configs/label_equivalence.json") -> LabelEquivalence:
    p = Path(path)
    return LabelEquivalence.load(p) if p.exists() else LabelEquivalence.default()
