# -*- coding: utf-8 -*-
"""
Hook 系统：把训练过程的“副作用”解耦（保存ckpt、评估、日志等）
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


class Hook:
    def on_train_start(self, trainer): ...
    def on_epoch_start(self, trainer): ...
    def on_step_end(self, trainer, logs: Dict[str, Any]): ...
    def on_epoch_end(self, trainer, logs: Dict[str, Any]): ...
    def on_train_end(self, trainer): ...


@dataclass
class CheckpointHook(Hook):
    ckpt_dir: Path
    save_every_epochs: int = 1
    save_best: bool = True
    monitor: str = "val/acc"   # 你后续可以改成 val/loss 或其他
    mode: str = "max"          # "max" or "min"

    best_value: Optional[float] = None

    def _is_better(self, v: float) -> bool:
        if self.best_value is None:
            return True
        return v > self.best_value if self.mode == "max" else v < self.best_value

    def on_epoch_end(self, trainer, logs: Dict[str, Any]):
        from src.utils.checkpoint import pack_checkpoint, save_checkpoint

        epoch = trainer.epoch
        if self.save_every_epochs > 0 and (epoch + 1) % self.save_every_epochs == 0:
            state = pack_checkpoint(
                model=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                epoch=epoch,
                global_step=trainer.global_step,
                best_metric=self.best_value,
            )
            save_checkpoint(self.ckpt_dir / f"epoch_{epoch+1}.pt", state)

        if self.save_best and self.monitor in logs:
            try:
                v = float(logs[self.monitor])
            except Exception:
                return
            if self._is_better(v):
                self.best_value = v
                state = pack_checkpoint(
                    model=trainer.model,
                    optimizer=trainer.optimizer,
                    scheduler=trainer.scheduler,
                    epoch=epoch,
                    global_step=trainer.global_step,
                    best_metric=self.best_value,
                )
                save_checkpoint(self.ckpt_dir / "best.pt", state)


class EvalHook(Hook):
    """每隔若干 epoch 做一次评估（依赖 trainer.evaluator）"""
    def __init__(self, eval_every_epochs: int = 1):
        self.eval_every_epochs = eval_every_epochs

    def on_epoch_end(self, trainer, logs: Dict[str, Any]):
        if trainer.evaluator is None:
            return
        epoch = trainer.epoch
        if self.eval_every_epochs > 0 and (epoch + 1) % self.eval_every_epochs == 0:
            metrics = trainer.evaluator.evaluate(trainer.model, trainer.device)
            # 写回 logs，统一前缀 val/
            for k, v in metrics.items():
                logs[f"val/{k}"] = v
