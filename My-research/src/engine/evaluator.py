# -*- coding: utf-8 -*-
"""
评估器：默认做分类评估的“壳”，具体指标你后续可替换/增强
"""

from __future__ import annotations

from typing import Any, Dict

import torch


class Evaluator:
    def __init__(self, dataloader):
        self.dataloader = dataloader

    @torch.no_grad()
    def evaluate(self, model, device: torch.device) -> Dict[str, float]:
        model.eval()
        total = 0
        correct = 0

        for batch in self.dataloader:
            batch = self._to_device(batch, device)
            out = model.predict(batch)

            # 默认约定 out 里有 logits；你接入 DSFSFD 时再按其输出结构改写即可
            logits = out["logits"] if isinstance(out, dict) and "logits" in out else out
            y = batch.get("y", None)
            if y is None:
                continue

            pred = torch.argmax(logits, dim=-1)
            total += int(y.numel())
            correct += int((pred == y).sum().item())

        acc = (correct / total) if total > 0 else 0.0
        return {"acc": float(acc)}

    def _to_device(self, batch: Any, device: torch.device) -> Any:
        if isinstance(batch, dict):
            return {k: self._to_device(v, device) for k, v in batch.items()}
        if torch.is_tensor(batch):
            return batch.to(device)
        if isinstance(batch, (list, tuple)):
            t = [self._to_device(x, device) for x in batch]
            return type(batch)(t)
        return batch
