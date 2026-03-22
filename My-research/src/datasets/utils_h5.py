# -*- coding: utf-8 -*-
"""Common utilities for reading unified HDF5 bearing datasets."""

from __future__ import annotations

import ast
import json
from typing import Any, Dict, Optional

import h5py
import numpy as np


REQUIRED_DATASETS = ("x_freq", "x_tf", "y", "domain")


class H5FormatError(RuntimeError):
    """Raised when the HDF5 file does not match the expected format."""



def _to_python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value



def decode_if_bytes(value: Any) -> Any:
    """Decode bytes-like objects to str when possible."""
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.dtype.kind in {"S", "O"}:
        return [decode_if_bytes(v) for v in value.tolist()]
    return value



def parse_h5_attr(value: Any, default: Optional[Any] = None) -> Any:
    """
    Parse an HDF5 attribute into a clean Python object.

    Supported common cases:
    - JSON string: '{"a": 1}'
    - Python literal string: "{'a': 1}"
    - bytes / byte arrays
    - numpy scalars / arrays
    """
    if value is None:
        return default

    value = decode_if_bytes(value)
    value = _to_python_scalar(value)

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(text)
            except Exception:
                pass
        return text

    if isinstance(value, np.ndarray):
        return value.tolist()

    return value



def load_h5_attrs(h5_file: h5py.File) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {}
    for key in h5_file.attrs.keys():
        attrs[key] = parse_h5_attr(h5_file.attrs.get(key))
    return attrs



def validate_h5_structure(h5_file: h5py.File) -> Dict[str, Any]:
    """Validate the file structure and return a summary dictionary."""
    missing = [name for name in REQUIRED_DATASETS if name not in h5_file]
    if missing:
        raise H5FormatError(f"Missing required datasets: {missing}")

    x_freq = h5_file["x_freq"]
    x_tf = h5_file["x_tf"]
    y = h5_file["y"]
    domain = h5_file["domain"]

    n = len(y)
    if len(x_freq) != n or len(x_tf) != n or len(domain) != n:
        raise H5FormatError(
            "Dataset length mismatch: "
            f"len(x_freq)={len(x_freq)}, len(x_tf)={len(x_tf)}, len(y)={len(y)}, len(domain)={len(domain)}"
        )

    summary = {
        "num_samples": n,
        "x_freq_shape": tuple(x_freq.shape),
        "x_tf_shape": tuple(x_tf.shape),
        "y_shape": tuple(y.shape),
        "domain_shape": tuple(domain.shape),
        "x_freq_dtype": str(x_freq.dtype),
        "x_tf_dtype": str(x_tf.dtype),
        "y_dtype": str(y.dtype),
        "domain_dtype": str(domain.dtype),
    }
    return summary



def invert_mapping(mapping: Optional[Dict[Any, Any]]) -> Dict[Any, Any]:
    if not isinstance(mapping, dict):
        return {}
    return {v: k for k, v in mapping.items()}



def normalize_mapping_keys_to_int(mapping: Optional[Dict[Any, Any]]) -> Dict[int, Any]:
    """Convert mapping keys like '0'/'1' to int when possible."""
    if not isinstance(mapping, dict):
        return {}
    normalized: Dict[int, Any] = {}
    for key, value in mapping.items():
        try:
            normalized[int(key)] = value
        except Exception:
            # fallback: skip non-numeric keys here
            pass
    return normalized
