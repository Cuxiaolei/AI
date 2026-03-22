# -*- coding: utf-8 -*-
"""CWRU dataset wrapper built on top of UnifiedH5Dataset."""

from __future__ import annotations
from typing import List, Optional
from .base_h5_dataset import UnifiedH5Dataset


class CWRUH5Dataset(UnifiedH5Dataset):
    DATASET_NAME = "CWRU"
    DEFAULT_CLASS_NAMES: List[str] = [
        "Normal", "IR007", "B007", "OR007", "IR014",
        "B014", "OR014", "IR021", "B021", "OR021",
    ]
    NORMAL_CLASS_NAME = "Normal"

    def get_dataset_name(self) -> str:
        return self.DATASET_NAME

    def get_normal_class_name(self) -> str:
        return self.NORMAL_CLASS_NAME

    def get_default_class_names(self) -> List[str]:
        return list(self.DEFAULT_CLASS_NAMES)

    def get_default_class_name(self, label_id: int) -> Optional[str]:
        label_id = int(label_id)
        if 0 <= label_id < len(self.DEFAULT_CLASS_NAMES):
            return self.DEFAULT_CLASS_NAMES[label_id]
        return None
