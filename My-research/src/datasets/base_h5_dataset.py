import h5py
import numpy as np
import torch
from typing import Optional, Dict, Any

from .utils_h5 import validate_h5_structure, parse_h5_attr


class H5FormatError(Exception):
    """HDF5文件格式异常"""
    pass


class UnifiedH5Dataset:
    """统一的HDF5数据集基类（仅支持频域freq模态）"""

    def __init__(
            self,
            h5_path: str,
            to_tensor: bool = True,
            dtype_freq: torch.dtype = torch.float32,
            return_index: bool = False
    ):
        self.h5_path = h5_path
        self.to_tensor = to_tensor
        self.dtype_freq = dtype_freq
        self.return_index = return_index

        # HDF5文件句柄（懒加载）
        self._h5_file: Optional[h5py.File] = None
        self._length: Optional[int] = None

        # 元数据
        self._id_to_label: Dict[int, str] = {}
        self._id_to_domain: Dict[int, str] = {}

        # 初始化
        self._init_dataset()

    def _init_dataset(self):
        """初始化数据集：校验结构 + 读取元数据"""
        with h5py.File(self.h5_path, "r") as f:
            # 校验结构（仅校验x_freq, y, domain）
            validate_h5_structure(f, require_tf=False)

            # 读取长度
            self._length = len(f["x_freq"])

            # 读取标签映射
            if "label_map" in f.attrs:
                self._id_to_label = parse_h5_attr(f.attrs["label_map"])
            if "domain_map" in f.attrs:
                self._id_to_domain = parse_h5_attr(f.attrs["domain_map"])

    def _ensure_open(self):
        """懒打开HDF5文件，避免多进程冲突"""
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r")

    def close(self):
        """关闭文件句柄"""
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None

    def __del__(self):
        self.close()

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取单条数据（仅频域）"""
        self._ensure_open()

        # 读取数据
        x_freq = self._h5_file["x_freq"][idx]
        y = self._h5_file["y"][idx]
        domain = self._h5_file["domain"][idx]

        # 转为Tensor
        if self.to_tensor:
            x_freq = torch.tensor(x_freq, dtype=self.dtype_freq)
            y = torch.tensor(y, dtype=torch.long)
            domain = torch.tensor(domain, dtype=torch.long)

        # 构造输出
        output = {
            "x_freq": x_freq,
            "y": y,
            "domain": domain
        }
        if self.return_index:
            output["index"] = idx
        return output

    # 元数据接口
    def get_num_classes(self) -> int:
        return len(self._id_to_label)

    def get_num_domains(self) -> int:
        return len(self._id_to_domain)

    def get_class_name(self, label_id: int) -> Optional[str]:
        return self._id_to_label.get(label_id)

    def get_domain_name(self, domain_id: int) -> Optional[str]:
        return self._id_to_domain.get(domain_id)

    def get_all_labels(self) -> np.ndarray:
        self._ensure_open()
        return self._h5_file["y"][:]

    def get_all_domains(self) -> np.ndarray:
        self._ensure_open()
        return self._h5_file["domain"][:]