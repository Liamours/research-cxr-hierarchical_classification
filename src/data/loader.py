"""DataLoader construction from an ExperimentConfig."""

from __future__ import annotations

from dataclasses import asdict

import torch
from torch.utils.data import DataLoader

from src.data.dataset import CxrClsDataset
from src.data.transforms import build_transform, norm_kind_for_backbone
from src.util.seed import seed_worker


def build_loaders(cfg) -> dict[str, DataLoader | None]:
    norm_kind = norm_kind_for_backbone(cfg.model.backbone)
    image_size = cfg.data.image_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator()
    generator.manual_seed(cfg.experiment.seed)

    seg_provider = None
    if cfg.seg.enabled:
        from src.data.segmentation import build_mask_provider
        try:
            seg_provider = build_mask_provider(cfg.seg, image_size)
        except FileNotFoundError:
            return {"train": None, "val": None, "test": None}

    aug_params = asdict(cfg.aug)
    loaders: dict[str, DataLoader | None] = {}
    for split, augment in (("train", True), ("val", False), ("test", False)):
        transform = build_transform(image_size, augment=augment, norm_kind=norm_kind,
                                    aug_params=aug_params)
        try:
            ds = CxrClsDataset(
                cfg.data.label_csv, cfg.data.image_root, split, transform,
                seg_provider=seg_provider, seg_method=cfg.seg.method,
                skip_missing_check=cfg.data.skip_missing_check,
            )
        except FileNotFoundError:
            loaders[split] = None
            continue
        if len(ds) == 0:
            loaders[split] = None
            continue

        nw = cfg.data.num_workers
        pin = device.type == "cuda"
        loaders[split] = DataLoader(
            ds,
            batch_size=cfg.data.batch_size,
            shuffle=(split == "train"),
            num_workers=nw,
            pin_memory=pin,
            drop_last=(split == "train"),
            worker_init_fn=seed_worker if nw > 0 else None,
            generator=generator if split == "train" else None,
            persistent_workers=nw > 0,
        )
    return loaders
