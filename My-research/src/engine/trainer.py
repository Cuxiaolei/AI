# -*- coding: utf-8 -*-
"""
通用 Trainer（先跑通框架，不绑定具体算法）
后续 DSFSFD 这种“两阶段 episodic 优化”可以写专用 trainer 并复用 Hook/日志/ckpt。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch.cuda.amp import autocast, GradScaler


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        train_loader,
        device: torch.device,
        scheduler=None,
        evaluator=None,
        hooks: Optional[List[Any]] = None,
        amp: bool = False,
        log_interval: int = 50,
        max_grad_norm: Optional[float] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.evaluator = evaluator
        self.device = device

        self.hooks = hooks or []
        self.amp = amp
        self.scaler = GradScaler(enabled=amp)
        self.log_interval = log_interval
        self.max_grad_norm = max_grad_norm

        self.epoch = 0
        self.global_step = 0

    def fit(self, epochs: int, logger) -> None:
        self.model.to(self.device)

        for h in self.hooks:
            h.on_train_start(self)

        for epoch in range(epochs):
            self.epoch = epoch
            for h in self.hooks:
                h.on_epoch_start(self)

            logs = self.train_one_epoch(logger)

            # epoch end hooks（评估、保存等）
            for h in self.hooks:
                h.on_epoch_end(self, logs)

            # scheduler step（默认按 epoch）
            if self.scheduler is not None:
                try:
                    self.scheduler.step()
                except TypeError:
                    pass

            # 打印 epoch 汇总
            if logger:
                msg = " | ".join([f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in logs.items()])
                logger.info(f"[Epoch {epoch+1}/{epochs}] {msg}")

        for h in self.hooks:
            h.on_train_end(self)

    def train_one_epoch(self, logger) -> Dict[str, Any]:
        self.model.train()

        running = {}
        count = 0

        for step, batch in enumerate(self.train_loader):
            batch = self._to_device(batch, self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.amp):
                outputs = self.model(batch)
                loss_dict = self.model.compute_loss(batch, outputs)
                loss = loss_dict["loss"]

            self.scaler.scale(loss).backward()

            if self.max_grad_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.global_step += 1

            # 累积日志
            count += 1
            for k, v in loss_dict.items():
                if torch.is_tensor(v):
                    v = float(v.detach().cpu().item())
                running[k] = running.get(k, 0.0) + float(v)

            # step hooks
            step_logs = {k: running[k] / count for k in running}
            for h in self.hooks:
                h.on_step_end(self, step_logs)

            if logger and self.log_interval > 0 and (step + 1) % self.log_interval == 0:
                msg = " | ".join([f"{k}={v:.4f}" for k, v in step_logs.items()])
                logger.info(f"[Epoch {self.epoch+1}] step={step+1} global_step={self.global_step} | {msg}")

        return {k: running[k] / max(count, 1) for k in running}

    def _to_device(self, batch: Any, device: torch.device) -> Any:
        if isinstance(batch, dict):
            return {k: self._to_device(v, device) for k, v in batch.items()}
        if torch.is_tensor(batch):
            return batch.to(device)
        if isinstance(batch, (list, tuple)):
            t = [self._to_device(x, device) for x in batch]
            return type(batch)(t)
        return batch
