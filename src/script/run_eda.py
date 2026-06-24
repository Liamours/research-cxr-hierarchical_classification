"""EDA entry point. Runs through a config; prints tables to the console.

    uv run python src/script/run_eda.py --config configs/<experiment>.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.experiment_config import ExperimentConfig
from src.eda.dataset_stats import run_eda
from src.util.logging import RunLogger


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--max-image-check", type=int, default=5000)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    print("Config:\n" + cfg.summary())
    report = run_eda(cfg, max_image_check=args.max_image_check)
    logger = RunLogger(cfg.run_dir(), level=cfg.logging.level)
    logger.event("eda", dataset=cfg.data.dataset, leakage=report["leakage"],
                 integrity=report["integrity"], label_sanity_ok=report["label_sanity"]["ok"])


if __name__ == "__main__":
    main()
