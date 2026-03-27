# -*- coding: utf-8 -*-
"""Factory helpers for creating dataset / dataloader instances."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from torch.utils.data import DataLoader

from .base_h5_dataset import UnifiedH5Dataset
from .dataset_phm import PHM2009H5Dataset
from .dataset_pu import PUH5Dataset


DATASET_REGISTRY = {
    "phm": PHM2009H5Dataset,
    "phm2009": PHM2009H5Dataset,
    "pu": PUH5Dataset,
    "generic": UnifiedH5Dataset,
}



def build_dataset(
    h5_path: str | Path,
    dataset_name: str = "generic",
    **dataset_kwargs: Any,
) -> UnifiedH5Dataset:
    key = dataset_name.lower()
    dataset_cls = DATASET_REGISTRY.get(key, UnifiedH5Dataset)
    return dataset_cls(h5_path=h5_path, **dataset_kwargs)



def build_dataloader(
    h5_path: str | Path,
    dataset_name: str = "generic",
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    persistent_workers: bool = False,
    dataset_kwargs: Optional[Dict[str, Any]] = None,
) -> DataLoader:
    dataset_kwargs = dataset_kwargs or {}
    dataset = build_dataset(h5_path=h5_path, dataset_name=dataset_name, **dataset_kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=(persistent_workers and num_workers > 0),
    )
