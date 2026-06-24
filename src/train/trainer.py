"""MultiLabelTrainer — training loop for flat multi-label CXR classification.

Features:
  - differential learning rates (backbone 1e-5 / head 1e-4) via param groups
  - bf16 autocast on CUDA (no GradScaler: bf16 has sufficient range)
  - gradient accumulation + grad clipping
  - backbone freeze for the first freeze_backbone_epochs
  - OneCycleLR (cosine) stepped per optimizer update
  - per-epoch validation loss + AUROC-macro + F1-macro (single deterministic
    pass via evaluate.evaluator.gather_predictions, mc_passes=0 -- MC Dropout
    stays eval/inference-only) + 5-sample prediction-vs-ground-truth display
  - all metrics logged through RunLogger (train_log.csv + events.jsonl + gpu mem)
  - checkpoints: run_dir/checkpoints/{last,best_val_loss,best_val_auroc_macro,
    best_val_f1_macro}.pt + matching *_meta.json; last overwritten every
    epoch, best_* only on strict improvement
  - early stopping on val loss
"""

from __future__ import annotations

import json
import math
import time

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from src.evaluate.evaluator import gather_predictions
from src.evaluate.metrics import compute_classification_metrics
from src.train.losses import build_loss
from src.util.device import bf16_supported


class MultiLabelTrainer:
    def __init__(self, model, loaders, cfg, device, logger, conditions=None):
        self.model = model.to(device)
        self.train_loader = loaders["train"]
        self.val_loader = loaders.get("val")
        self.cfg = cfg
        self.device = device
        self.logger = logger
        self.conditions = conditions if conditions is not None else cfg.resolved_conditions()
        self.criterion = build_loss(
            cfg.label.label_structure,
            self.conditions,
        )
        self.use_amp = cfg.training.bf16 and bf16_supported(device)
        self._best_ckpt = {
            "val_loss": float("nan"),
            "val_auroc_macro": float("nan"),
            "val_f1_macro": float("nan"),
        }
        self._build_optim()

    def _loss(self, out, y, m):
        return self.criterion(out, y, m)

    def _build_optim(self):
        t = self.cfg.training
        self.optimizer = AdamW(
            self.model.param_groups(t.backbone_lr, t.head_lr),
            weight_decay=t.weight_decay,
        )
        steps_per_epoch = max(len(self.train_loader) // t.grad_accum_steps, 1)
        total = max(steps_per_epoch * t.epochs, 1)
        pct = min(t.warmup_steps / total, 0.3)
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=[t.backbone_lr, t.head_lr],
            total_steps=total,
            pct_start=pct,
            anneal_strategy="cos",
        )

    def _set_backbone_frozen(self, frozen: bool):
        for p in self.model.backbone.parameters():
            p.requires_grad = not frozen

    def _train_epoch(self, epoch: int) -> dict:
        t = self.cfg.training
        self.model.train()
        self._set_backbone_frozen(epoch <= t.freeze_backbone_epochs)
        total_loss = bce_sum = penalty_sum = 0.0
        grad_norm_sum = grad_norm_max = 0.0
        nb = n_updates = 0
        self.optimizer.zero_grad()
        pbar = tqdm(self.train_loader, desc=f"epoch {epoch} train", dynamic_ncols=True)
        for step, batch in enumerate(pbar):
            x = batch["pixel_values"].to(self.device)
            y = batch["labels"].to(self.device)
            m = batch["label_mask"].to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                out = self.model(x)
                unscaled = self._loss(out, y, m)
                loss = unscaled / t.grad_accum_steps
            loss.backward()
            if (step + 1) % t.grad_accum_steps == 0:
                raw_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), t.max_grad_norm))
                grad_norm_sum += raw_norm
                grad_norm_max = max(grad_norm_max, raw_norm)
                n_updates += 1
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            step_loss = float(unscaled.detach())
            total_loss += step_loss
            bce_sum    += getattr(self.criterion, "_last_bce", step_loss)
            penalty_sum += getattr(self.criterion, "_last_penalty", 0.0)
            nb += 1
            pbar.set_postfix(loss=f"{step_loss:.4f}")
        n = max(nb, 1)
        nu = max(n_updates, 1)
        return {
            "loss":           total_loss / n,
            "bce":            bce_sum / n,
            "penalty":        penalty_sum / n,
            "grad_norm_mean": grad_norm_sum / nu,
            "grad_norm_max":  grad_norm_max,
        }

    @torch.no_grad()
    def _eval_epoch(self) -> dict[str, float]:
        if self.val_loader is None:
            return {"val_loss": float("nan"), "val_auroc_macro": float("nan"),
                    "val_f1_macro": float("nan")}
        self.model.eval()
        total, nb = 0.0, 0
        for batch in tqdm(self.val_loader, desc="val", dynamic_ncols=True):
            x = batch["pixel_values"].to(self.device)
            y = batch["labels"].to(self.device)
            m = batch["label_mask"].to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                out = self.model(x)
                total += self._loss(out, y, m).item()
            nb += 1
        val_loss = total / max(nb, 1)

        # Single deterministic pass (mc_passes=0) for AUROC/F1 -- MC Dropout
        # stays eval/inference-only, see module docstring.
        _, probs, targets, mask, *_ = gather_predictions(
            self.model, self.val_loader, self.device, self.use_amp, mc_passes=0
        )
        cm = compute_classification_metrics(probs, targets, mask, self.conditions)
        return {
            "val_loss": val_loss,
            "val_auroc_macro": cm["auroc"]["macro"],
            "val_f1_macro": cm["f1"]["macro"],
        }

    @torch.no_grad()
    def _show_val_samples(self, n: int):
        if self.val_loader is None or n <= 0:
            return
        self.model.eval()
        shown = 0
        for batch in self.val_loader:
            x = batch["pixel_values"].to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                out = self.model(x)
                probs = torch.sigmoid(out).float().cpu()
            for i in range(probs.size(0)):
                if shown >= n:
                    break
                y, mask = batch["labels"][i], batch["label_mask"][i]
                gt = [self.conditions[j] for j in range(len(self.conditions))
                      if mask[j] > 0 and y[j] == 1]
                # top-3 from applicable labels only (mask[j]>0)
                applicable_idx = [j for j in range(len(self.conditions)) if mask[j] > 0]
                if applicable_idx:
                    order = sorted(applicable_idx, key=lambda j: probs[i][j], reverse=True)[:3]
                else:
                    order = torch.argsort(probs[i], descending=True)[:3].tolist()
                top = [(self.conditions[j], round(probs[i][j].item(), 3)) for j in order]
                print(f"  [val {shown + 1}] GT+={gt or '[none]'} | top3={top}")
                shown += 1
            if shown >= n:
                break
        self.model.train()

    def _save_checkpoint(self, name: str, epoch: int, val: dict[str, float]):
        ckpt_dir = self.cfg.run_dir() / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), ckpt_dir / f"{name}.pt")
        meta = {"epoch": epoch, **val}
        (ckpt_dir / f"{name}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _is_better(self, metric: str, value: float) -> bool:
        if math.isnan(value):
            return False
        current = self._best_ckpt[metric]
        if math.isnan(current):
            return True
        return value < current if metric == "val_loss" else value > current

    def _save_checkpoints(self, epoch: int, val: dict[str, float]):
        self._save_checkpoint("last", epoch, val)
        for metric in ("val_loss", "val_auroc_macro", "val_f1_macro"):
            if self._is_better(metric, val[metric]):
                self._best_ckpt[metric] = val[metric]
                self._save_checkpoint(f"best_{metric}", epoch, val)

    def train(self) -> float:
        t = self.cfg.training
        self.logger.log(
            f"train {self.cfg.experiment.name} backbone={self.cfg.model.backbone} "
            f"epochs={t.epochs} device={self.device}"
        )
        self.logger.event("param_counts", **self.model.param_counts())

        best = float("inf")
        patience = 0
        last_epoch = 0
        for epoch in range(1, t.epochs + 1):
            last_epoch = epoch
            t0 = time.time()
            train_stats = self._train_epoch(epoch)
            train_loss = train_stats["loss"]
            val = self._eval_epoch()
            val_loss = val["val_loss"]
            elapsed = time.time() - t0

            self.logger.log(
                f"epoch {epoch}/{t.epochs}  "
                f"loss={train_loss:.4f}  bce={train_stats['bce']:.4f}  "
                f"penalty={train_stats['penalty']:.4f}  "
                f"gnorm={train_stats['grad_norm_mean']:.3f}(max={train_stats['grad_norm_max']:.3f})  "
                f"val_loss={val_loss:.4f}  "
                f"auroc={val['val_auroc_macro']:.4f}  f1={val['val_f1_macro']:.4f}  "
                f"{elapsed:.0f}s"
            )

            print("  [val samples]")
            self._show_val_samples(self.cfg.logging.val_display_rows)
            self.logger.log_epoch({
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_bce": round(train_stats["bce"], 4),
                "train_hbce_penalty": round(train_stats["penalty"], 4),
                "grad_norm_mean": round(train_stats["grad_norm_mean"], 4),
                "grad_norm_max": round(train_stats["grad_norm_max"], 4),
                "val_loss": round(val_loss, 4),
                "val_auroc_macro": round(val["val_auroc_macro"], 4),
                "val_f1_macro": round(val["val_f1_macro"], 4),
                "lr_backbone": self.optimizer.param_groups[0]["lr"],
                "lr_head": self.optimizer.param_groups[1]["lr"],
                "elapsed_s": round(elapsed, 1),
                **self.logger.gpu_mem(),
            })

            self._save_checkpoints(epoch, val)
            if not math.isnan(val_loss) and val_loss < best:
                best = val_loss
                patience = 0
            else:
                patience += 1
                if patience >= t.early_stop_patience:
                    self.logger.log(f"early stop at epoch {epoch}")
                    break

        self.logger.final_metrics({"best_val_loss": round(best, 4), "epochs_ran": last_epoch})
        self.logger.log(f"done. best_val_loss={best:.4f}")
        return best
