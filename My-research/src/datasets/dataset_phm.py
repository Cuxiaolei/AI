# -*- coding: utf-8 -*-
"""PHM 2009 dataset wrapper built on top of UnifiedH5Dataset."""

from __future__ import annotations
from typing import List, Optional
from .base_h5_dataset import UnifiedH5Dataset


class PHM2009H5Dataset(UnifiedH5Dataset):
    DATASET_NAME = "PHM 2009"
    DEFAULT_CLASS_NAMES: List[str] = [
        "spur 1", "spur 2", "spur 3", "spur 4",
        "spur 5", "spur 6", "spur 7", "spur 8",
    ]
    NORMAL_CLASS_NAME = "spur 1"

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
