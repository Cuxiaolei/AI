# -*- coding: utf-8 -*-
"""PU dataset wrapper built on top of UnifiedH5Dataset."""

from __future__ import annotations
from typing import List, Optional
from .base_h5_dataset import UnifiedH5Dataset


class PUH5Dataset(UnifiedH5Dataset):
    DATASET_NAME = "PU"
    DEFAULT_CLASS_NAMES: List[str] = [
        "K004", "KI21", "KI18", "KI16",
        "KA04", "KA16", "KB27", "KB23", "KB24",
    ]
    NORMAL_CLASS_NAME = "K004"

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
