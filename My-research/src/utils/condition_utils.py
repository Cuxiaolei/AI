# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from typing import Any, Dict, Tuple, Optional

import torch


# 统一条件向量维度：
# [speed_rpm_norm, torque_nm_norm, radial_force_n_norm]
COND_DIM = 3


def _safe_int(x: Any) -> int:
    return int(x)


def _reverse_domain_map(domain_map: Dict[Any, Any]) -> Dict[int, str]:
    """
    h5 attrs里的 domain_map 通常是:
        {'30hz_High': 0, '35hz_High': 1, ...}
    这里转成:
        {0: '30hz_High', 1: '35hz_High', ...}
    """
    out: Dict[int, str] = {}
    for cond_name, domain_id in domain_map.items():
        out[_safe_int(domain_id)] = str(cond_name)
    return out


def _get_dataset_attrs(dataset) -> Dict[str, Any]:
    """
    兼容你当前的数据集接口。
    优先使用 get_attrs()，没有则返回空字典。
    """
    if dataset is None:
        return {}
    if hasattr(dataset, "get_attrs"):
        attrs = dataset.get_attrs()
        if isinstance(attrs, dict):
            return attrs
    return {}


def parse_phm_condition(cond_name: str) -> torch.Tensor:
    """
    PHM:
        例如 '30hz_High'
    只提取速度信息，忽略 High/Low 等载荷描述。
    输出:
        [speed_rpm_norm, 0.0, 0.0]
    """
    m = re.search(r"(\d+)\s*hz", cond_name, flags=re.IGNORECASE)
    if m is None:
        raise ValueError(f"Invalid PHM condition name: {cond_name}")

    speed_hz = float(m.group(1))
    speed_rpm = speed_hz * 60.0

    # 归一化：PHM常见转速量级可按 3000 rpm 归一
    speed_rpm_norm = speed_rpm / 3000.0

    return torch.tensor(
        [speed_rpm_norm, 0.0, 0.0],
        dtype=torch.float32,
    )


def parse_pu_condition(cond_name: str) -> torch.Tensor:
    """
    PU:
        例如 'N15_M07_F10'
    常见解释：
        N15 -> 1500 rpm
        M07 -> 0.7 Nm
        F10 -> 1000 N

    输出:
        [speed_rpm_norm, torque_nm_norm, radial_force_n_norm]
    """
    m = re.fullmatch(r"N(\d+)_M(\d+)_F(\d+)", cond_name.strip(), flags=re.IGNORECASE)
    if m is None:
        raise ValueError(f"Invalid PU condition name: {cond_name}")

    speed_rpm = float(m.group(1)) * 100.0
    torque_nm = float(m.group(2)) / 10.0
    radial_force_n = float(m.group(3)) * 100.0

    # 归一化
    speed_rpm_norm = speed_rpm / 3000.0
    torque_nm_norm = torque_nm / 10.0
    radial_force_n_norm = radial_force_n / 1000.0

    return torch.tensor(
        [speed_rpm_norm, torque_nm_norm, radial_force_n_norm],
        dtype=torch.float32,
    )


def parse_condition(dataset_name: str, cond_name: str) -> torch.Tensor:
    dataset_name = str(dataset_name).lower()
    if dataset_name == "phm":
        return parse_phm_condition(cond_name)
    if dataset_name == "pu":
        return parse_pu_condition(cond_name)
    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


def _collect_domain_map_from_dataset(dataset) -> Dict[int, str]:
    """
    从单个 dataset 中收集 domain_id -> condition_name
    """
    attrs = _get_dataset_attrs(dataset)
    domain_map = attrs.get("domain_map", None)
    if domain_map is None:
        return {}
    if not isinstance(domain_map, dict):
        raise TypeError("dataset attrs['domain_map'] must be a dict")
    return _reverse_domain_map(domain_map)


def _merge_domain_maps(
    train_dataset,
    test_dataset,
) -> Dict[int, str]:
    """
    合并 train/test 的 domain_map
    同一 domain_id 若条件名冲突则报错
    """
    merged: Dict[int, str] = {}
    for ds in [train_dataset, test_dataset]:
        cur = _collect_domain_map_from_dataset(ds)
        for domain_id, cond_name in cur.items():
            if domain_id in merged and merged[domain_id] != cond_name:
                raise ValueError(
                    f"Conflicting condition name for domain_id={domain_id}: "
                    f"{merged[domain_id]} vs {cond_name}"
                )
            merged[domain_id] = cond_name

    if len(merged) == 0:
        raise ValueError("No valid domain_map found in train/test datasets.")
    return merged


def build_condition_table(
    domain_id_to_name: Dict[int, str],
    dataset_name: str,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    根据 domain_id -> condition_name 构造:
        condition_table: [max_domain_id + 1, COND_DIM]
        meta: 记录辅助信息
    """
    if len(domain_id_to_name) == 0:
        raise ValueError("domain_id_to_name is empty")

    max_domain_id = max(domain_id_to_name.keys())
    table = torch.zeros(max_domain_id + 1, COND_DIM, dtype=torch.float32)

    parsed: Dict[int, Dict[str, Any]] = {}



    for domain_id, cond_name in sorted(domain_id_to_name.items(), key=lambda x: x[0]):
        cond_vec = parse_condition(dataset_name, cond_name)
        if cond_vec.numel() != COND_DIM:
            raise ValueError(
                f"Condition vector dim mismatch for domain_id={domain_id}, cond_name={cond_name}"
            )
        table[domain_id] = cond_vec
        parsed[domain_id] = {
            "condition_name": cond_name,
            "condition_vector": cond_vec.tolist(),
        }

    meta = {
        "dataset_name": str(dataset_name).lower(),
        "cond_dim": COND_DIM,
        "domain_id_to_name": {int(k): str(v) for k, v in domain_id_to_name.items()},
        "parsed_conditions": parsed,
    }
    return table, meta


def build_condition_table_from_datasets(
    train_dataset,
    test_dataset,
    dataset_name: str,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    domain_id_to_name = _merge_domain_maps(train_dataset, test_dataset)
    return build_condition_table(
        domain_id_to_name=domain_id_to_name,
        dataset_name=dataset_name,
    )