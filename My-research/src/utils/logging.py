# -*- coding: utf-8 -*-
"""
日志与运行目录管理：
- outputs/runs/<exp_name>/<run_id>/{checkpoints,logs,metrics,figures}
- 同时输出到控制台与文件（UTF-8）
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class RunDirs:
    run_dir: Path
    checkpoints: Path
    logs: Path
    metrics: Path
    figures: Path


def make_run_id() -> str:
    """生成一个简洁的 run_id：时间戳 + pid"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_pid{os.getpid()}"


def create_run_dirs(outputs_root: str, exp_name: str, run_id: Optional[str] = None) -> RunDirs:
    outputs_root = str(outputs_root)
    exp_name = exp_name or "default_exp"
    run_id = run_id or make_run_id()

    run_dir = Path(outputs_root) / "runs" / exp_name / run_id
    checkpoints = run_dir / "checkpoints"
    logs = run_dir / "logs"
    metrics = run_dir / "metrics"
    figures = run_dir / "figures"

    for p in [checkpoints, logs, metrics, figures]:
        p.mkdir(parents=True, exist_ok=True)

    return RunDirs(run_dir=run_dir, checkpoints=checkpoints, logs=logs, metrics=metrics, figures=figures)


def setup_logger(log_file: Path, level: int = logging.INFO, name: str = "myresearch") -> logging.Logger:
    """配置 logger：控制台 + 文件"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # 避免重复添加 handler（例如多次运行 main）
    if logger.handlers:
        return logger

    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # 控制台
    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(fmt)

    # 文件（UTF-8）
    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)

    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger
