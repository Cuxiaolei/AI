# -*- coding: utf-8 -*-
"""
配置加载器（支持 YAML/JSON），并提供深度合并（deep merge）。
用法建议：
- experiments/*.yaml 里写最完整的实验配置
- base.yaml 提供公共默认值
加载顺序：base -> data -> model -> train -> experiment（后者覆盖前者）
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def _try_import_yaml():
    try:
        import yaml  # type: ignore
        return yaml
    except Exception:
        return None


def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并字典：b 覆盖 a（同 key 且都是 dict 则继续递归）。"""
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config_file(path: Union[str, Path]) -> Dict[str, Any]:
    """加载单个 YAML/JSON 配置文件。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")

    suffix = p.suffix.lower()
    text = p.read_text(encoding="utf-8")

    if suffix in [".yaml", ".yml"]:
        yaml = _try_import_yaml()
        if yaml is None:
            raise RuntimeError("未安装 PyYAML，无法读取 .yaml/.yml 配置文件。请在 requirements 里加入 pyyaml。")
        data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            raise ValueError(f"YAML 顶层必须是 dict：{p}")
        return data

    if suffix == ".json":
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError(f"JSON 顶层必须是 dict：{p}")
        return data

    raise ValueError(f"不支持的配置后缀：{suffix}（仅支持 .yaml/.yml/.json）")


def load_configs(paths: List[Union[str, Path]]) -> Dict[str, Any]:
    """按顺序加载多个配置并深度合并（后者覆盖前者）。"""
    cfg: Dict[str, Any] = {}
    for p in paths:
        part = load_config_file(p)
        cfg = deep_merge(cfg, part)
    return cfg


def dump_config(cfg: Dict[str, Any], path: Union[str, Path]) -> None:
    """将最终配置写入 YAML（若无 PyYAML 则写 JSON）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    yaml = _try_import_yaml()
    if yaml is not None and p.suffix.lower() in [".yaml", ".yml"]:
        p.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    else:
        p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


@dataclass
class ConfigPaths:
    """可选：规范化“多配置组合”"""
    base: Optional[str] = None
    data: Optional[str] = None
    model: Optional[str] = None
    train: Optional[str] = None
    experiment: Optional[str] = None

    def to_list(self) -> List[str]:
        return [p for p in [self.base, self.data, self.model, self.train, self.experiment] if p]
