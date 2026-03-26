from typing import Optional, Dict, Any
from torch.utils.data import DataLoader

from .base_h5_dataset import UnifiedH5Dataset
from .phm2009_dataset import PHM2009H5Dataset
from .pu_dataset import PUH5Dataset

# 数据集注册表（已删除CWRU）
DATASET_REGISTRY = {
    "phm": PHM2009H5Dataset,
    "phm2009": PHM2009H5Dataset,
    "pu": PUH5Dataset,
    "generic": UnifiedH5Dataset,
}


def build_dataset(
        h5_path: str,
        dataset_name: str = "generic",
        to_tensor: bool = True,
        return_index: bool = False,
        **kwargs
):
    """构建数据集（仅freq模态）"""
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"支持的数据集: {list(DATASET_REGISTRY.keys())}")

    dataset_cls = DATASET_REGISTRY[dataset_name]
    return dataset_cls(
        h5_path=h5_path,
        to_tensor=to_tensor,
        return_index=return_index,
        **kwargs
    )


def build_dataloader(
        h5_path: str,
        dataset_name: str = "generic",
        batch_size: int = 32,
        shuffle: bool = False,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        return_index: bool = False,
        dataset_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
) -> DataLoader:
    """构建数据加载器（仅freq）"""
    dataset_kwargs = dataset_kwargs or {}
    dataset = build_dataset(
        h5_path=h5_path,
        dataset_name=dataset_name,
        return_index=return_index,
        **dataset_kwargs
    )

    if persistent_workers and num_workers == 0:
        persistent_workers = False

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        **kwargs
    )