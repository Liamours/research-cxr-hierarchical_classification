"""Structured run logging — console, run.log, events.jsonl, train_log.csv, metrics.json.

One RunLogger per experiment run directory. Captures enough to debug a run
after the fact: a human-readable log, a machine-readable event stream, a
per-epoch metrics table, and a final metrics snapshot.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


_LEVELS = {"debug": 10, "info": 20, "warning": 30, "error": 40, "critical": 50}


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunLogger:
    def __init__(self, run_dir: str | Path, level: str = "info"):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.level = level
        self._threshold = _LEVELS.get(str(level).lower(), 20)
        self.run_log = self.run_dir / "run.log"
        self.events = self.run_dir / "events.jsonl"
        self.train_log = self.run_dir / "train_log.csv"

    def log(self, msg: str, level: str = "info") -> None:
        if _LEVELS.get(level.lower(), 20) < self._threshold:
            return
        line = f"[{_ts()}] {level.upper()} {msg}"
        print(line)
        with open(self.run_log, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def event(self, kind: str, **fields: Any) -> None:
        rec = {"ts": _ts(), "kind": kind, **fields}
        with open(self.events, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def snapshot_config(self, cfg) -> None:
        out = self.run_dir / "config.yaml"
        cfg.to_yaml(out)
        self.event("config_snapshot", path=str(out))

    def log_epoch(self, row: dict[str, Any]) -> None:
        write_header = not self.train_log.exists()
        with open(self.train_log, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        self.event("epoch", **row)

    def gpu_mem(self) -> dict[str, float]:
        try:
            if torch.cuda.is_available():
                return {
                    "alloc_mb": round(torch.cuda.memory_allocated() / 1e6, 1),
                    "reserved_mb": round(torch.cuda.memory_reserved() / 1e6, 1),
                }
        except Exception:
            pass
        return {"alloc_mb": 0.0, "reserved_mb": 0.0}

    def final_metrics(self, metrics: dict[str, Any]) -> None:
        with open(self.run_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=str)
        self.event("final_metrics", **metrics)
