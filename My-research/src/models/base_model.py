# -*- coding: utf-8 -*-
"""
BaseModel：给你的框架一个统一接口（不限定具体算法）
后续具体模型（比如 DSFSFD）可以直接继承并实现必要方法。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch: Dict[str, Any]) -> Any:
        """
        默认约定 batch 是 dict。具体模型自己决定字段：
        - batch["x"] / batch["y"] / batch["domain"] / batch["meta"] ...
        """
        raise NotImplementedError

    def compute_loss(self, batch: Dict[str, Any], outputs: Any) -> Dict[str, torch.Tensor]:
        """
        返回一个 dict：
        - 必须包含 key="loss" 的标量 Tensor
        - 其他 key 作为日志输出（例如 cls_loss、reg_loss）
        """
        raise NotImplementedError

    @torch.no_grad()
    def predict(self, batch: Dict[str, Any]) -> Any:
        """推理接口（默认直接 forward）"""
        return self.forward(batch)

    def get_param_groups(self) -> Optional[list]:
        """
        可选：返回 optimizer 参数组（用于不同 lr/weight_decay）。
        返回 None 表示直接用 model.parameters()。
        """
        return None
