# -*- coding: utf-8 -*-
"""YAML config loading and recursive merging utilities."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge_dict(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f'Config at {path} must be a YAML mapping.')
    return data


def load_config(config_path: str | Path) -> Dict[str, Any]:
    config_path = Path(config_path)
    cfg = load_yaml(config_path)
    base_ref = cfg.pop('BASE', None)
    if base_ref is None:
        return cfg
    base_path = (config_path.parent / base_ref).resolve() if not Path(base_ref).is_absolute() else Path(base_ref)
    base_cfg = load_config(base_path)
    return _deep_merge_dict(base_cfg, cfg)


def dump_yaml(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


__all__ = ['load_config', 'dump_yaml']
