"""Experiment configuration for the CXR classification + UQ ablation.

One YAML file = one experiment = one point in the 3-factor grid
(segmentation x label structure x uncertainty quantification).
Backbone is fixed: densenet121_xrv (TorchXRayVision, CXR-pretrained).
All experiments are launched through this config; nothing is hardcoded.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from datetime import date
from pathlib import Path
from typing import Any


@dataclass
class ExperimentMeta:
    name: str = "densenet121_xrv_baseline"
    seed: int = 42
    notes: str = ""


@dataclass
class DataCfg:
    dataset: str = "combined"  # combined | chexpert | nih-cxr14 | vindr-cxr | ...
    label_csv: Path = Path("dataset/combined/combined.csv")
    image_root: Path = Path("data/preprocessed/images")
    image_size: int = 224
    batch_size: int = 16
    num_workers: int = 0
    skip_missing_check: bool = False  # skip per-row Path.exists(); safe for verified CSVs

    def __post_init__(self):
        self.label_csv = Path(self.label_csv)
        self.image_root = Path(self.image_root)


@dataclass
class AugCfg:
    """Train-time augmentation (applied only to the train split).

    hflip defaults OFF: a horizontal flip mirrors the heart/aorta (rare
    situs-inversus anatomy). It is common in the literature but clinically
    contested; enable explicitly if desired. Geometric jitter (small rotation +
    translate + scale) and mild brightness/contrast are the safe defaults.
    """
    hflip: bool = False
    rotation_deg: float = 10.0
    translate: float = 0.05
    scale_jitter: float = 0.05  # scale ~ U(1-s, 1+s)
    brightness: float = 0.1
    contrast: float = 0.1


@dataclass
class ModelCfg:
    backbone: str = "densenet121_xrv"
    pretrained: bool = True
    dropout: float = 0.2  # consumed by MC Dropout when uq.method == "mc_dropout"


@dataclass
class LabelCfg:
    """Factor 3 — label structure."""
    label_structure: str = "flat"  # flat | hierarchical | hierarchical_soft | bafl
    conditions: list = field(default_factory=list)  # empty = all CANONICAL_LABELS; non-empty = subset
    lam: float = 0.5  # HBCE/soft-HBCE penalty weight (hierarchical* only; 0 == flat)
    # BAFL only (Balanced Adaptive Focal Loss, HP-ViT paper Sect. 3.3). Defaults
    # are the paper's own values -- untuned for our label imbalance, see
    # context/hpvit-bafl-ablation-proposal.md "Open questions".
    bafl_beta: float = 0.999        # effective-number-of-samples decay (Cui et al. 2019)
    bafl_gamma_init: float = 0.5    # focal exponent at epoch 0
    bafl_gamma_final: float = 2.5   # focal exponent at epoch >= bafl_t_warmup
    bafl_t_warmup: int = 30         # epochs to ramp gamma_init -> gamma_final


@dataclass
class SegCfg:
    """Factor 2 — organ segmentation conditioning (CheXmask-U)."""
    enabled: bool = False
    method: str = "concat_channel"  # concat_channel (LOCKED canonical) | crop
    mask_source: str = "chexmask_u"  # chexmask_u | synthetic (smoke only)
    mask_root: str = "data/preprocessed/masks/chexmask_u"


@dataclass
class UQCfg:
    """Factor 4 — uncertainty quantification."""
    method: str = "none"  # none | mc_dropout
    mc_passes: int = 30
    gate_threshold: float = 0.5


@dataclass
class TrainCfg:
    epochs: int = 15
    grad_accum_steps: int = 2
    head_lr: float = 1e-4
    backbone_lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    bf16: bool = True
    early_stop_patience: int = 3
    freeze_backbone_epochs: int = 1


@dataclass
class EvalCfg:
    metrics: list[str] = field(default_factory=lambda: ["auroc", "f1", "ece"])
    reliability_bins: int = 15
    threshold: float = 0.5


@dataclass
class XaiCfg:
    target_layer: str = "auto"
    overlay_alpha: float = 0.5


@dataclass
class PathsCfg:
    result_root: Path = Path("result")

    def __post_init__(self):
        self.result_root = Path(self.result_root)


@dataclass
class LoggingCfg:
    level: str = "info"
    val_display_rows: int = 5


_SECTIONS: dict[str, type] = {
    "experiment": ExperimentMeta,
    "data": DataCfg,
    "aug": AugCfg,
    "model": ModelCfg,
    "seg": SegCfg,
    "label": LabelCfg,
    "uq": UQCfg,
    "training": TrainCfg,
    "eval": EvalCfg,
    "xai": XaiCfg,
    "paths": PathsCfg,
    "logging": LoggingCfg,
}


def _build_section(name: str, cls: type, data: dict[str, Any]):
    valid = {f.name for f in fields(cls)}
    unknown = set(data) - valid
    if unknown:
        raise ValueError(
            f"Unknown keys in config section '{name}': {sorted(unknown)}. "
            f"Valid keys: {sorted(valid)}"
        )
    return cls(**data)


@dataclass
class ExperimentConfig:
    experiment: ExperimentMeta = field(default_factory=ExperimentMeta)
    data: DataCfg = field(default_factory=DataCfg)
    aug: AugCfg = field(default_factory=AugCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    seg: SegCfg = field(default_factory=SegCfg)
    label: LabelCfg = field(default_factory=LabelCfg)
    uq: UQCfg = field(default_factory=UQCfg)
    training: TrainCfg = field(default_factory=TrainCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    xai: XaiCfg = field(default_factory=XaiCfg)
    paths: PathsCfg = field(default_factory=PathsCfg)
    logging: LoggingCfg = field(default_factory=LoggingCfg)
    _run_date: str | None = field(default=None, init=False, repr=False, compare=False)

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: dict[str, Any] | None = None) -> "ExperimentConfig":
        import yaml

        with open(path, encoding="utf-8") as f:
            raw: dict = yaml.safe_load(f) or {}

        unknown_sections = set(raw) - set(_SECTIONS)
        if unknown_sections:
            raise ValueError(
                f"Unknown config sections: {sorted(unknown_sections)}. "
                f"Valid sections: {sorted(_SECTIONS)}"
            )

        kwargs = {}
        for name, sec_cls in _SECTIONS.items():
            kwargs[name] = _build_section(name, sec_cls, raw.get(name) or {})

        cfg = cls(**kwargs)
        if overrides:
            cfg.apply_overrides(overrides)
        return cfg

    def apply_overrides(self, overrides: dict[str, Any]) -> None:
        """Apply dotted-key overrides, e.g. {'training.epochs': 2}."""
        for dotted, value in overrides.items():
            if value is None:
                continue
            section, _, key = dotted.partition(".")
            if section not in _SECTIONS or not key:
                raise ValueError(f"Bad override key: {dotted!r} (expected 'section.field')")
            sec = getattr(self, section)
            if not hasattr(sec, key):
                raise ValueError(f"Unknown override field: {dotted!r}")
            setattr(sec, key, value)
        self.data.__post_init__()
        self.paths.__post_init__()

    def to_yaml(self, path: str | Path) -> None:
        import yaml

        nested: dict[str, dict] = {}
        for name in _SECTIONS:
            sec = getattr(self, name)
            section_dict = {}
            for f in fields(sec):
                v = getattr(sec, f.name)
                if isinstance(v, Path):
                    section_dict[f.name] = str(v)
                elif isinstance(v, list) and v and hasattr(v[0], "__dataclass_fields__"):
                    from dataclasses import asdict as _asdict
                    section_dict[f.name] = [_asdict(item) for item in v]
                else:
                    section_dict[f.name] = v
            nested[name] = section_dict

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(nested, fh, sort_keys=False, allow_unicode=True)

    def resolved_conditions(self) -> list[str]:
        """Active label set: cfg.label.conditions if non-empty, else all 51 CANONICAL_LABELS."""
        from src.data.label_space import CANONICAL_LABELS
        return list(self.label.conditions) if self.label.conditions else list(CANONICAL_LABELS)

    def run_dir(self) -> Path:
        """Date is stamped once per instance so a run spanning midnight keeps
        writing to the directory it started in; the name stays live."""
        if self._run_date is None:
            self._run_date = f"{date.today():%Y%m%d}"
        return self.paths.result_root / f"{self._run_date}_{self.experiment.name}"

    def summary(self) -> str:
        data_desc = self.data.dataset
        conds = self.resolved_conditions()
        cond_desc = f"{len(conds)} ({'subset' if self.label.conditions else 'all'})"
        return "\n".join([
            f"experiment={self.experiment.name}  seed={self.experiment.seed}",
            f"backbone={self.model.backbone}  pretrained={self.model.pretrained}",
            f"factors: seg={self.seg.enabled}({self.seg.method})  label={self.label.label_structure}  uq={self.uq.method}",
            f"conditions={cond_desc}  metrics={self.eval.metrics}",
            f"data={data_desc}  img={self.data.image_size}  bs={self.data.batch_size}",
            f"train: epochs={self.training.epochs}  head_lr={self.training.head_lr}  "
            f"backbone_lr={self.training.backbone_lr}  bf16={self.training.bf16}",
            f"run_dir={self.run_dir()}",
        ])
