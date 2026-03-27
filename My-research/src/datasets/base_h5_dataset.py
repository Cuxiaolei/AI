# -*- coding: utf-8 -*-
"""Base PyTorch Dataset for unified HDF5 bearing datasets (freq-only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .utils_h5 import (
    load_h5_attrs,
    normalize_mapping_keys_to_int,
    validate_h5_structure,
)


class UnifiedH5Dataset(Dataset):
    """
    Generic dataset for unified train.h5 / test.h5 files (freq-only).

    Expected required datasets inside one HDF5 file:
        - x_freq: [N, 1, F], float32
        - y:      [N], int64
        - domain: [N], int64

    Optional dataset:
        - x_tf:   [N, 1, 128, 128], ignored if present

    Notes
    -----
    This dataset is intentionally simplified to frequency-domain input only.
    To keep old configs from crashing immediately, `feature_mode` and `dtype_tf`
    are still accepted in __init__, but they are ignored.
    """

    def __init__(
        self,
        h5_path: str | Path,
        feature_mode: str = "freq",            # backward-compatible, ignored
        to_tensor: bool = True,
        dtype_tf: Optional[torch.dtype] = None,  # backward-compatible, ignored
        dtype_freq: Optional[torch.dtype] = torch.float32,
        return_index: bool = False,
    ) -> None:
        super().__init__()

        self.h5_path = str(h5_path)
        self.feature_mode = "freq"
        self.to_tensor = to_tensor
        self.dtype_freq = dtype_freq
        self.return_index = return_index

        self._h5_file: Optional[h5py.File] = None
        self._summary: Dict[str, Any] = {}
        self._attrs: Dict[str, Any] = {}
        self._num_samples = 0

        self._inspect_file_once()

    def _inspect_file_once(self) -> None:
        with h5py.File(self.h5_path, "r") as f:
            self._summary = validate_h5_structure(f)
            self._attrs = load_h5_attrs(f)
            self._num_samples = int(self._summary["num_samples"])

        self._label_map = self._attrs.get("label_map", {})
        self._domain_map = self._attrs.get("domain_map", {})
        self._split_name = self._attrs.get("split_name", None)

        self._id_to_label = normalize_mapping_keys_to_int(
            self._label_map if isinstance(self._label_map, dict) else {}
        )
        if not self._id_to_label and isinstance(self._label_map, dict):
            try:
                self._id_to_label = {int(v): k for k, v in self._label_map.items()}
            except Exception:
                self._id_to_label = {}

        self._id_to_domain = normalize_mapping_keys_to_int(
            self._domain_map if isinstance(self._domain_map, dict) else {}
        )
        if not self._id_to_domain and isinstance(self._domain_map, dict):
            try:
                self._id_to_domain = {int(v): k for k, v in self._domain_map.items()}
            except Exception:
                self._id_to_domain = {}

    def _ensure_open(self) -> h5py.File:
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r")
        return self._h5_file

    def close(self) -> None:
        if self._h5_file is not None:
            try:
                self._h5_file.close()
            except Exception:
                pass
            finally:
                self._h5_file = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __len__(self) -> int:
        return self._num_samples

    def _read_freq(self, idx: int) -> np.ndarray:
        h5_file = self._ensure_open()
        x = h5_file["x_freq"][idx]
        return np.asarray(x, dtype=np.float32)

    def _read_label(self, idx: int) -> int:
        h5_file = self._ensure_open()
        return int(h5_file["y"][idx])

    def _read_domain(self, idx: int) -> int:
        h5_file = self._ensure_open()
        return int(h5_file["domain"][idx])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        x_freq = self._read_freq(idx)
        y = self._read_label(idx)
        d = self._read_domain(idx)

        if self.to_tensor:
            x_freq = torch.from_numpy(x_freq)
            if self.dtype_freq is not None:
                x_freq = x_freq.to(self.dtype_freq)

            sample: Dict[str, Any] = {
                "x_freq": x_freq,
                "y": torch.tensor(y, dtype=torch.long),
                "domain": torch.tensor(d, dtype=torch.long),
            }
        else:
            sample = {
                "x_freq": x_freq,
                "y": y,
                "domain": d,
            }

        if self.return_index:
            sample["index"] = idx

        return sample

    # ---------- metadata / utility interfaces ----------
    def get_num_classes(self) -> int:
        if self._id_to_label:
            return len(self._id_to_label)
        with h5py.File(self.h5_path, "r") as f:
            return int(np.unique(f["y"][:]).size)

    def get_num_domains(self) -> int:
        if self._id_to_domain:
            return len(self._id_to_domain)
        with h5py.File(self.h5_path, "r") as f:
            return int(np.unique(f["domain"][:]).size)

    def get_label_map(self) -> Dict[Any, Any]:
        return self._label_map if isinstance(self._label_map, dict) else {}

    def get_domain_map(self) -> Dict[Any, Any]:
        return self._domain_map if isinstance(self._domain_map, dict) else {}

    def get_split_name(self) -> Optional[str]:
        return self._split_name

    def get_summary(self) -> Dict[str, Any]:
        return dict(self._summary)

    def get_attrs(self) -> Dict[str, Any]:
        return dict(self._attrs)

    def get_class_name(self, label_id: int) -> Optional[str]:
        return self._id_to_label.get(int(label_id))

    def get_domain_name(self, domain_id: int) -> Optional[str]:
        return self._id_to_domain.get(int(domain_id))

    def get_all_labels(self) -> np.ndarray:
        with h5py.File(self.h5_path, "r") as f:
            return np.asarray(f["y"][:], dtype=np.int64)

    def get_all_domains(self) -> np.ndarray:
        with h5py.File(self.h5_path, "r") as f:
            return np.asarray(f["domain"][:], dtype=np.int64)

    def extra_repr(self) -> str:
        return (
            f"h5_path={self.h5_path}, feature_mode=freq, "
            f"num_samples={self._num_samples}, split_name={self._split_name}"
        )