# -*- coding: utf-8 -*-
"""
模型注册器：用于把 backbones 统一纳入框架管理
- 通过 @register_model("name") 装饰器注册
- create_model(cfg["model"]) 实例化
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

_MODEL_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register_model(name: str):
    def deco(fn: Callable[..., Any]):
        if name in _MODEL_REGISTRY:
            raise KeyError(f"Model already registered: {name}")
        _MODEL_REGISTRY[name] = fn
        return fn
    return deco


def list_models() -> Dict[str, Callable[..., Any]]:
    return dict(_MODEL_REGISTRY)


def get_model_builder(name: str) -> Callable[..., Any]:
    if name not in _MODEL_REGISTRY:
        raise KeyError(f"Model not registered: {name}. Available: {sorted(_MODEL_REGISTRY.keys())}")
    return _MODEL_REGISTRY[name]


def create_model(model_cfg: Dict[str, Any]) -> Any:
    """
    model_cfg 约定：
    {
      "name": "xxx",
      "kwargs": {...}
    }
    """
    name = model_cfg.get("name")
    if not name:
        raise ValueError("model_cfg 缺少字段：name")
    kwargs = model_cfg.get("kwargs", {}) or {}
    builder = get_model_builder(name)
    return builder(**kwargs)
