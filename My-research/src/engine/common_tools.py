# src/engine/common_tools.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm


class TqdmLoggingHandler(logging.Handler):
    """让 logger 输出不破坏 tqdm 进度条"""
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


def setup_logger(log_file: Path, name: str = "trainer") -> logging.Logger:
    """
    console: tqdm.write (不破坏进度条)
    file: 正常落盘
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"{name}_{log_file.parent.parent.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(message)s")

    sh = TqdmLoggingHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.propagate = False
    return logger


class EMA:
    def __init__(self, momentum: float = 0.95):
        self.m = float(momentum)
        self.v: Optional[float] = None

    def update(self, x: float) -> float:
        x = float(x)
        if self.v is None:
            self.v = x
        else:
            self.v = self.m * self.v + (1.0 - self.m) * x
        return self.v


def make_unique_exp_dir(base_dir: Path, exp_name: str, max_tries: int = 10000) -> Path:
    """
    base_dir/exp_name 若存在，则返回 base_dir/exp_name_1, _2, ...
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    cand = base_dir / exp_name
    if not cand.exists():
        return cand
    for i in range(1, max_tries + 1):
        cand_i = base_dir / f"{exp_name}_{i}"
        if not cand_i.exists():
            return cand_i
    raise RuntimeError(f"Too many existing experiment folders for name={exp_name}")


def append_csv_row(csv_path: Path, header: List[str], row: List[Any]):
    """按行追加（如不存在先写 header）"""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(header)
        w.writerow(row)


def append_csv_dict_row(csv_path: Path, fieldnames: List[str], row_dict: Dict[str, Any]):
    """
    追加一行 dict（按 fieldnames 顺序写入）。
    若文件不存在先写 header。
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            w.writeheader()
        # 只写 fieldnames 内的键
        w.writerow({k: row_dict.get(k, "") for k in fieldnames})


def save_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def set_seed(seed: int):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_device(device: str) -> torch.device:
    """
    兼容你现有 DSFSFD：内部存在 .cuda() 写死，当前实现要求 GPU。
    """
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device)
    raise RuntimeError(
        "当前 DSFSFD 实现要求 GPU（内部存在 .cuda() 写死）。如果你想支持 CPU，我可以给你统一 .to(device) 的补丁。"
    )


def ensure_device_any(device: str) -> torch.device:
    """
    通用版：PCDG 等模型可以 CPU/GPU 自适应。
    """
    device = str(device)
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device)
    return torch.device("cpu")


def resolve_path(project_root: Path, p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (project_root / pp).resolve()


def stats_line(pbar: tqdm, metrics: Dict[str, str]) -> str:
    """
    输出你想要的这一行格式：
    [00:09<00:55,  1.53it/s, ps_loss=..., ...]
    """
    fd = getattr(pbar, "format_dict", {})
    elapsed = tqdm.format_interval(fd.get("elapsed", 0.0))

    remaining = fd.get("remaining", None)
    remaining_str = tqdm.format_interval(remaining) if remaining is not None else "??:??"

    rate = fd.get("rate", None)
    rate_str = f"{rate:5.2f}it/s" if rate else "  ?.?it/s"

    m = ", ".join([f"{k}={v}" for k, v in metrics.items()])
    return f"[{elapsed}<{remaining_str}, {rate_str}, {m}]"


def safe_name(s: str) -> str:
    """把 run_name 变成适合文件名的字符串"""
    bad = ['\\', '/', ':', '*', '?', '"', '<', '>', '|', ' ']
    out = s
    for b in bad:
        out = out.replace(b, "_")
    # 避免过长
    return out[:180]
