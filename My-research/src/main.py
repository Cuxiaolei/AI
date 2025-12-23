# src/main.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any

import yaml


def deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并：upd 覆盖 base（对 dict 做深合并，其它直接覆盖）。"""
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    return data


def main(argv: List[str] | None = None):
    parser = argparse.ArgumentParser("My-research training entry")
    # 支持多个配置叠加：python run.py --config configs/base.yaml configs/experiments/pu_dsfsfd_lodo.yaml
    parser.add_argument(
        "--config",
        nargs="+",
        required=True,
        help="One or more yaml configs. Later ones override earlier ones.",
    )
    args = parser.parse_args(argv)

    # project root: <root>/src/main.py -> parents[1] is <root>
    project_root = Path(__file__).resolve().parents[1]

    # 确保 src 可 import（run.py 已经做了，这里再保险一次）
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # 1) 先加载用户给的 configs（按顺序叠加覆盖）
    cfg: Dict[str, Any] = {}
    for c in args.config:
        p = (project_root / c).resolve() if not Path(c).is_absolute() else Path(c).resolve()
        cfg = deep_update(cfg, load_yaml(p))

    # 2) 自动加载 model cfg：configs/model/<name>.yaml，再与上面的 cfg 合并
    model_name = cfg.get("model", {}).get("name", None)
    if not model_name:
        raise ValueError("你的 experiments yaml 里必须包含：model: { name: dsfsfd }（或其他模型名）")

    model_cfg_path = (project_root / "configs" / "model" / f"{model_name}.yaml").resolve()
    if not model_cfg_path.exists():
        raise FileNotFoundError(f"找不到模型配置文件：{model_cfg_path}")

    model_cfg = load_yaml(model_cfg_path)

    # 合并顺序：先 model_cfg，再让用户传入的 cfg 覆盖（用户配置优先）
    merged = deep_update(model_cfg, cfg)

    # 3) dispatch：按模型名进入对应 trainer
    if merged["model"]["name"] == "dsfsfd":
        from .engine.dsfsfd_trainer import run as run_dsfsfd
        run_dsfsfd(merged, project_root)
    else:
        raise NotImplementedError(f"Unknown model.name={merged['model']['name']} (目前只接入了 dsfsfd)。")
