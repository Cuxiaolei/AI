# -*- coding: utf-8 -*-
"""
Checkpoint 工具：
- 保存/加载模型、优化器、调度器、epoch、best 指标
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class CheckpointState:
    epoch: int
    global_step: int
    best_metric: Optional[float]
    payload: Dict[str, Any]


def save_checkpoint(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import torch  # type: ignore
    torch.save(state, str(path))


def load_checkpoint(path: Path, map_location: str = "cpu") -> Dict[str, Any]:
    import torch  # type: ignore
    return torch.load(str(path), map_location=map_location)


def pack_checkpoint(
    model,
    optimizer=None,
    scheduler=None,
    epoch: int = 0,
    global_step: int = 0,
    best_metric: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "model": model.state_dict(),
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if extra:
        state["extra"] = extra
    return state


def unpack_checkpoint(state: Dict[str, Any], model, optimizer=None, scheduler=None, strict: bool = True) -> Dict[str, Any]:
    model.load_state_dict(state["model"], strict=strict)
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    return state
