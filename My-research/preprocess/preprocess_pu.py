#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# 文件功能概述
# -----------------------------------------------------------------------------
# 本脚本用于将 PU（Paderborn University）轴承数据集预处理为“不平衡小样本域泛化”实验
# 可直接使用的 HDF5 数据文件。
#
# 最终每个 split 会生成：
#   1) .h5 文件：包含 x_freq, x_tf, y, domain 四个核心数据集
#   2) _trace.json：记录每个类、每个工况、每个原始 mat 文件的抽样轨迹，便于检查和复现
#
# 核心流程
# -----------------------------------------------------------------------------
# 1. 读取 YAML 配置文件
# 2. 扫描 PU 数据集目录，按“类别文件夹 / 工况 / mat 文件”组织数据
# 3. 从每个 .mat 文件中提取第 7 个通道 vibration_1 的一维振动信号
# 4. 对每个 split（train / val / test）逐个工况、逐个类别进行样本抽取
# 5. 对每个原始振动信号按 win_len 和 stride 进行滑窗切片
# 6. 若常规滑窗样本不足，则仍只在当前 class-condition 对应文件内部做错位重叠补样
# 7. 每个窗样本生成：
#      - 一维频域特征 x_freq：rFFT -> power/magnitude -> log1p
#      - 二维时频图特征 x_tf：STFT power -> log1p -> resize
# 8. 保存为 HDF5，并同步保存详细 trace.json
#
# 重要设计原则
# -----------------------------------------------------------------------------
# 1. 样本只允许来自当前“类别-工况”对应的原始文件集合，不允许从其他工况/类别/域借数据
# 2. 抽样优先在当前文件集合内尽量均匀分配
# 3. 常规滑窗不足时，仅在同一文件内部进行“错位重叠采样”
# 4. 错位重叠采样生成的新窗口起点必须与原滑窗不同，避免完全重合
# 5. 终端会打印简要检查信息，trace.json 会保存详细抽样轨迹
#
# 主要函数及相互关系
# -----------------------------------------------------------------------------
# set_seed / load_yaml / ensure_dir
#   基础工具函数：设置随机种子、读取配置、创建目录
#
# parse_condition_from_filename / infer_available_conditions
#   从文件名中解析工况，并扫描数据集中有哪些可用工况
#
# load_pu_vibration_signal
#   从 .mat 文件中提取一维 vibration_1 振动信号（优先取 Y 的第 7 个通道）
#
# sliding_start_positions / build_candidate_pools_for_signal
#   为单个原始信号生成常规滑窗起点和错位重叠滑窗起点
#
# sample_window_starts_per_file_strict_same_group
#   在当前 class-condition 对应的所有文件内部进行严格抽样：
#   先用常规滑窗，不够再用同组文件内部的错位重叠采样
#
# compute_freq_feature / compute_tf_feature
#   将一个时域窗样本转换为频域特征和时频图特征
#
# collect_split
#   按 train/val/test 的设定采集整个 split 的所有样本，并记录 trace 信息
#
# save_h5 / save_trace_json
#   保存 HDF5 和 trace.json
#
# build_split_cfgs / main
#   构造 train/val/test 任务划分，并组织整个预处理入口
#
# 重要配置参数（最关键）
# -----------------------------------------------------------------------------
# win_len / stride
#   控制滑窗长度与常规步长
#
# source_conditions / target_condition / use_val / val_condition
#   控制源域、目标域、验证域的任务划分
#
# normal_per_domain / fault_per_class_per_domain
#   控制每个域里正常类和故障类的抽样数
#
# overlap_sampling.enabled / overlap_sampling.offsets
#   控制是否启用错位重叠补样及其偏移量
#
# x_tf_dtype / tf_image_size / stft
#   控制时频图的存储类型、尺寸和 STFT 参数
# =============================================================================

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import h5py
import numpy as np
import yaml
from scipy import signal
from scipy.io import loadmat
from scipy.ndimage import zoom

from preprocess.config import load_config


@dataclass
class SplitConfig:
    name: str
    conditions: List[str]
    normal_per_domain: int
    fault_per_class_per_domain: int


# ========================= 基础工具函数 =========================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ========================= 文件名 / 工况解析 =========================

def normalize_condition_name(cond: str) -> str:
    return cond.strip()


def parse_condition_from_filename(file_path: Path) -> str:
    """
    例如：
      N09_M07_F10_K001_1.mat -> N09_M07_F10
      N15_M07_F04_KA04_20.mat -> N15_M07_F04
    """
    stem = file_path.stem.strip()
    parts = [p.strip() for p in stem.split("_") if p.strip()]
    if len(parts) < 5:
        raise ValueError(f"文件名无法解析工况: {file_path.name}")
    return normalize_condition_name("_".join(parts[:3]))


def parse_repeat_from_filename(file_path: Path) -> str:
    stem = file_path.stem.strip()
    parts = [p.strip() for p in stem.split("_") if p.strip()]
    if len(parts) < 1:
        raise ValueError(f"文件名无法解析重复编号: {file_path.name}")
    return parts[-1]


def infer_available_conditions(root_dir: Path, class_folders: List[str]) -> List[str]:
    conds = set()
    for cls in class_folders:
        class_dir = root_dir / cls
        if not class_dir.exists():
            continue
        for fp in class_dir.rglob("*"):
            if not fp.is_file() or fp.suffix.lower() != ".mat":
                continue
            try:
                conds.add(parse_condition_from_filename(fp))
            except Exception:
                pass
    return sorted(conds)


def list_class_files_for_condition(class_dir: Path, condition: str) -> List[Path]:
    target_cond = normalize_condition_name(condition)
    files = []
    for fp in sorted(class_dir.rglob("*")):
        if not fp.is_file() or fp.suffix.lower() != ".mat":
            continue
        try:
            cond = parse_condition_from_filename(fp)
        except Exception:
            continue
        if cond == target_cond:
            files.append(fp)
    return files


def build_class_folder_map(root_dir: Path, class_folders: List[str]) -> Dict[str, Path]:
    mapping = {}
    for cls in class_folders:
        p = root_dir / cls
        if not p.exists():
            raise FileNotFoundError(f"类别文件夹不存在: {p}")
        mapping[cls] = p
    return mapping


def make_label_map(class_folders: List[str]) -> Dict[str, int]:
    return {cls: i for i, cls in enumerate(class_folders)}


# ========================= MAT 结构解析与振动信号提取 =========================

def is_numpy_numeric_array(x: Any) -> bool:
    return isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.number)


def is_python_numeric_scalar(x: Any) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating))


def object_has_field(obj: Any, field_name: str) -> bool:
    return hasattr(obj, field_name)


def get_object_fields(obj: Any) -> List[str]:
    if hasattr(obj, "_fieldnames") and obj._fieldnames is not None:
        return list(obj._fieldnames)
    if isinstance(obj, dict):
        return list(obj.keys())
    return []


def get_object_field(obj: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(field_name, default)
    return getattr(obj, field_name, default)


def flatten_to_object_list(x: Any) -> List[Any]:
    """
    将 MATLAB 读入后的对象、list、tuple、object ndarray 展平为 Python list
    """
    if x is None:
        return []

    if isinstance(x, (list, tuple)):
        out = []
        for item in x:
            out.extend(flatten_to_object_list(item))
        return out

    if isinstance(x, np.ndarray):
        if np.issubdtype(x.dtype, np.number):
            return [x]
        out = []
        for item in x.flat:
            out.extend(flatten_to_object_list(item))
        return out

    return [x]


def choose_top_struct(mat_dict: Dict[str, Any]) -> Any:
    """
    在 loadmat 结果中找到真正的顶层结构体对象：
    优先选择带 Y 字段的对象。
    """
    keys = [k for k in mat_dict.keys() if not k.startswith("__")]
    candidates = [mat_dict[k] for k in keys]

    # 优先找有 Y 字段的对象
    for obj in candidates:
        if object_has_field(obj, "Y"):
            return obj

    # 如果没有，返回第一个非空对象
    for obj in candidates:
        if obj is not None:
            return obj

    raise RuntimeError("未在 .mat 中找到可用顶层结构体")


def find_best_numeric_vector(x: Any, min_len: int = 1024) -> Optional[np.ndarray]:
    """
    递归寻找最合适的一维数值向量。
    优先返回长度较长的向量。
    """
    best = None

    def _update(arr: np.ndarray) -> None:
        nonlocal best
        arr = np.asarray(arr).astype(np.float32).reshape(-1)
        if arr.size < min_len:
            return
        if best is None or arr.size > best.size:
            best = arr

    def _recurse(obj: Any) -> None:
        if obj is None:
            return

        if is_python_numeric_scalar(obj):
            return

        if is_numpy_numeric_array(obj):
            if obj.ndim == 0:
                return
            _update(obj)
            return

        if isinstance(obj, np.ndarray):
            for item in obj.flat:
                _recurse(item)
            return

        if isinstance(obj, (list, tuple)):
            for item in obj:
                _recurse(item)
            return

        if isinstance(obj, dict):
            preferred_keys = [
                "vibration_1", "vibration", "data", "Data", "signal", "Signal",
                "values", "Value", "x_values", "y_values"
            ]
            used = set()
            for k in preferred_keys:
                if k in obj:
                    used.add(k)
                    _recurse(obj[k])
            for k, v in obj.items():
                if k not in used:
                    _recurse(v)
            return

        fields = get_object_fields(obj)
        if len(fields) > 0:
            preferred_fields = [
                "vibration_1", "vibration", "data", "Data", "signal", "Signal",
                "values", "Value", "x_values", "y_values"
            ]
            used = set()
            for f in preferred_fields:
                if f in fields:
                    used.add(f)
                    _recurse(get_object_field(obj, f))
            for f in fields:
                if f not in used:
                    _recurse(get_object_field(obj, f))
            return

    _recurse(x)
    return best


def choose_vibration_channel_from_Y(y_obj: Any) -> Any:
    """
    用户给出的结构是：
      顶层 struct -> Y (1x7 struct array)
      第 7 个通道为 vibration_1
    因此优先取 Y 的第 7 个元素；
    若解析失败，再退化为从 Y 中搜索最合理的振动向量。
    """
    items = flatten_to_object_list(y_obj)

    # 优先按第7通道取（索引6）
    if len(items) >= 7:
        return items[6]

    # 不足 7 个时，返回整个 y_obj 让后续递归搜索
    return y_obj


def load_pu_vibration_signal(file_path: Path) -> np.ndarray:
    """
    从 PU 的 .mat 文件中提取 vibration_1 一维振动信号。
    """
    mat = loadmat(file_path, squeeze_me=True, struct_as_record=False)
    top = choose_top_struct(mat)
    y_obj = get_object_field(top, "Y", None)
    if y_obj is None:
        raise RuntimeError(f"未在 MAT 顶层结构中找到 Y 字段: {file_path}")

    channel_obj = choose_vibration_channel_from_Y(y_obj)
    sig = find_best_numeric_vector(channel_obj, min_len=1024)

    # 若第7通道没成功，再在整个 Y 中兜底搜索
    if sig is None:
        sig = find_best_numeric_vector(y_obj, min_len=1024)

    if sig is None:
        raise RuntimeError(f"未能从 MAT 中提取 vibration_1 振动信号: {file_path}")

    sig = np.asarray(sig, dtype=np.float32).reshape(-1)
    sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
    return sig


# ========================= 信号处理与特征构造 =========================

def zscore_1d(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mu = float(np.mean(x))
    std = float(np.std(x))
    if std < eps:
        return x - mu
    return (x - mu) / (std + eps)


def sliding_start_positions(sig_len: int, win_len: int, stride: int) -> np.ndarray:
    if sig_len < win_len:
        return np.empty((0,), dtype=np.int64)
    return np.arange(0, sig_len - win_len + 1, stride, dtype=np.int64)


def sliding_start_positions_with_offset(
    sig_len: int,
    win_len: int,
    stride: int,
    offset: int,
) -> np.ndarray:
    if sig_len < win_len:
        return np.empty((0,), dtype=np.int64)
    max_start = sig_len - win_len
    if offset > max_start:
        return np.empty((0,), dtype=np.int64)
    return np.arange(offset, max_start + 1, stride, dtype=np.int64)


def make_overlap_offsets(stride: int) -> List[int]:
    cands = []
    for x in [stride // 2, stride // 4, (3 * stride) // 4, stride // 3, (2 * stride) // 3]:
        if 0 < x < stride:
            cands.append(int(x))
    out = []
    seen = set()
    for x in cands:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def build_candidate_pools_for_signal(
    sig_len: int,
    win_len: int,
    stride: int,
    extra_offsets: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    base_starts = sliding_start_positions(sig_len, win_len, stride)

    overlap_list = []
    for off in extra_offsets:
        starts = sliding_start_positions_with_offset(sig_len, win_len, stride, off)
        if len(starts) > 0:
            overlap_list.append(starts)

    if len(overlap_list) == 0:
        overlap_starts = np.empty((0,), dtype=np.int64)
    else:
        overlap_starts = np.unique(np.concatenate(overlap_list, axis=0))
        overlap_starts = np.setdiff1d(overlap_starts, base_starts, assume_unique=False)

    return base_starts, overlap_starts


def compute_freq_feature(window: np.ndarray, fft_mode: str = "power") -> np.ndarray:
    fft = np.fft.rfft(window)
    mag = np.abs(fft)
    if fft_mode == "power":
        feat = mag ** 2
    elif fft_mode == "magnitude":
        feat = mag
    else:
        raise ValueError(f"未知 fft_mode: {fft_mode}")
    feat = np.log1p(feat).astype(np.float32)
    return feat[None, :]


def resize_2d(img: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    h, w = img.shape
    out_h, out_w = out_hw
    zoom_h = out_h / max(h, 1)
    zoom_w = out_w / max(w, 1)
    return zoom(img, zoom=(zoom_h, zoom_w), order=1)


def compute_tf_feature(
    window: np.ndarray,
    nperseg: int,
    noverlap: int,
    nfft: int,
    out_hw: Tuple[int, int],
) -> np.ndarray:
    _, _, zxx = signal.stft(
        window,
        fs=1.0,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        boundary=None,
        padded=False,
    )
    power = np.abs(zxx) ** 2
    power = np.log1p(power.astype(np.float32))
    power = resize_2d(power, out_hw)
    return power[None, :, :]


# ========================= 抽样逻辑 =========================

def allocate_counts_evenly(total_count: int, n_buckets: int) -> List[int]:
    base = total_count // n_buckets
    rem = total_count % n_buckets
    return [base + (1 if i < rem else 0) for i in range(n_buckets)]


def _sample_from_pool_without_reuse(
    pool: np.ndarray,
    n_take: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if n_take <= 0 or len(pool) == 0:
        return np.empty((0,), dtype=np.int64), pool
    n_take = min(n_take, len(pool))
    idx = rng.choice(len(pool), size=n_take, replace=False)
    chosen = np.sort(pool[idx])
    remain = np.setdiff1d(pool, chosen, assume_unique=False)
    return chosen, remain


def fill_deficits_round_robin(
    remain_pools: List[np.ndarray],
    selected_pairs: List[List[Tuple[int, str]]],
    deficits: int,
    mode: str,
    rng: np.random.Generator,
) -> int:
    n_files = len(remain_pools)

    while deficits > 0:
        progressed = False
        for i in range(n_files):
            if deficits <= 0:
                break
            pool = remain_pools[i]
            if len(pool) == 0:
                continue

            idx = int(rng.integers(0, len(pool)))
            start = int(pool[idx])

            selected_pairs[i].append((start, mode))
            remain_pools[i] = np.delete(pool, idx)
            deficits -= 1
            progressed = True

        if not progressed:
            break

    return deficits


def sample_window_starts_per_file_strict_same_group(
    signals: List[np.ndarray],
    total_count: int,
    win_len: int,
    stride: int,
    rng: np.random.Generator,
    extra_offsets: List[int],
) -> Tuple[List[List[Tuple[int, str]]], List[Dict[str, Any]]]:
    """
    仅允许在当前 class-condition 对应的这组文件内部抽样：
    1) 先按 quota 从每个文件的 base 池抽样
    2) 若不足，在这组文件的剩余 base 池中轮转补样
    3) 若仍不足，在这组文件的 overlap 池中轮转补样
    4) 仍不足则报错
    """
    n_files = len(signals)
    quotas = allocate_counts_evenly(total_count, n_files)

    selected_pairs: List[List[Tuple[int, str]]] = [[] for _ in range(n_files)]
    remain_base_pools: List[np.ndarray] = []
    remain_overlap_pools: List[np.ndarray] = []
    file_debug_infos: List[Dict[str, Any]] = []

    deficits = 0

    for i, (sig, q) in enumerate(zip(signals, quotas)):
        base_pool, overlap_pool = build_candidate_pools_for_signal(
            sig_len=len(sig),
            win_len=win_len,
            stride=stride,
            extra_offsets=extra_offsets,
        )

        chosen_base, remain_base = _sample_from_pool_without_reuse(
            pool=base_pool,
            n_take=q,
            rng=rng,
        )

        for s in chosen_base:
            selected_pairs[i].append((int(s), "base"))

        remain_base_pools.append(remain_base)
        remain_overlap_pools.append(overlap_pool.copy())

        deficits += max(0, q - len(chosen_base))

        file_debug_infos.append(
            {
                "signal_length": int(len(sig)),
                "base_candidate_count": int(len(base_pool)),
                "overlap_candidate_count": int(len(overlap_pool)),
                "total_candidate_count": int(len(base_pool) + len(overlap_pool)),
            }
        )

    if deficits > 0:
        deficits = fill_deficits_round_robin(
            remain_pools=remain_base_pools,
            selected_pairs=selected_pairs,
            deficits=deficits,
            mode="base",
            rng=rng,
        )

    if deficits > 0:
        deficits = fill_deficits_round_robin(
            remain_pools=remain_overlap_pools,
            selected_pairs=selected_pairs,
            deficits=deficits,
            mode="overlap",
            rng=rng,
        )

    if deficits > 0:
        raise RuntimeError(
            f"当前 class-condition 对应文件内部样本不足："
            f"请求 {total_count}，但常规滑窗 + 错位重叠滑窗仍不足，仍缺少 {deficits} 个。"
        )

    for i in range(n_files):
        selected_pairs[i] = sorted(selected_pairs[i], key=lambda x: x[0])

    return selected_pairs, file_debug_infos


# ========================= split 构造 =========================

def validate_conditions(
    available_conditions: List[str],
    source_conditions: List[str],
    target_condition: str,
    use_val: bool,
    val_condition: Optional[str],
) -> Tuple[List[str], List[str], List[str]]:
    avail = set(available_conditions)

    for cond in source_conditions + [target_condition]:
        if cond not in avail:
            raise ValueError(
                f"工况不存在于数据中: {cond}\n"
                f"当前扫描到的工况有: {sorted(avail)}"
            )

    if target_condition in source_conditions:
        raise ValueError("target_condition 不能同时出现在 source_conditions 中")

    if use_val:
        if val_condition is None:
            raise ValueError("use_val=True 时必须提供 val_condition")
        if val_condition not in source_conditions:
            raise ValueError("val_condition 必须来自 source_conditions")
        train_conditions = [c for c in source_conditions if c != val_condition]
        if len(train_conditions) == 0:
            raise ValueError("划分验证集后，训练域数量不能为空")
        val_conditions = [val_condition]
    else:
        train_conditions = list(source_conditions)
        val_conditions = []

    test_conditions = [target_condition]
    return train_conditions, val_conditions, test_conditions


def choose_split_sampling(cfg: Dict[str, Any], split_name: str) -> Tuple[int, int]:
    sampling = cfg["sampling"]
    if split_name == "train":
        key = "source"
    elif split_name == "test":
        key = "target"
    elif split_name == "val":
        key = "val"
    else:
        raise ValueError(f"未知 split_name: {split_name}")

    normal = int(sampling[key]["normal_per_domain"])
    fault = int(sampling[key]["fault_per_class_per_domain"])
    return normal, fault


def build_split_cfgs(cfg: Dict[str, Any]) -> Tuple[List[SplitConfig], Dict[str, Any]]:
    ds_cfg = cfg["dataset"]
    sp_cfg = cfg["split"]

    available_conditions = infer_available_conditions(
        root_dir=Path(ds_cfg["root_dir"]),
        class_folders=ds_cfg["class_folders"],
    )

    train_conditions, val_conditions, test_conditions = validate_conditions(
        available_conditions=available_conditions,
        source_conditions=sp_cfg["source_conditions"],
        target_condition=sp_cfg["target_condition"],
        use_val=bool(sp_cfg.get("use_val", False)),
        val_condition=sp_cfg.get("val_condition", None),
    )

    train_normal, train_fault = choose_split_sampling(cfg, "train")
    test_normal, test_fault = choose_split_sampling(cfg, "test")

    split_cfgs = [
        SplitConfig(
            name="train",
            conditions=train_conditions,
            normal_per_domain=train_normal,
            fault_per_class_per_domain=train_fault,
        )
    ]

    if len(val_conditions) > 0:
        val_normal, val_fault = choose_split_sampling(cfg, "val")
        split_cfgs.append(
            SplitConfig(
                name="val",
                conditions=val_conditions,
                normal_per_domain=val_normal,
                fault_per_class_per_domain=val_fault,
            )
        )

    split_cfgs.append(
        SplitConfig(
            name="test",
            conditions=test_conditions,
            normal_per_domain=test_normal,
            fault_per_class_per_domain=test_fault,
        )
    )

    info = {
        "available_conditions": available_conditions,
        "train_conditions": train_conditions,
        "val_conditions": val_conditions,
        "test_conditions": test_conditions,
    }
    return split_cfgs, info


# ========================= split 数据收集 =========================

def collect_split(
    root_dir: Path,
    class_folders: List[str],
    normal_class_folder: str,
    label_map: Dict[str, int],
    domain_map: Dict[str, int],
    split_cfg: SplitConfig,
    feat_cfg: Dict[str, Any],
    seed: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    class_dir_map = build_class_folder_map(root_dir, class_folders)

    x_freq_list: List[np.ndarray] = []
    x_tf_list: List[np.ndarray] = []
    y_list: List[int] = []
    d_list: List[int] = []

    trace_entries: List[Dict[str, Any]] = []

    win_len = int(feat_cfg["win_len"])
    stride = int(feat_cfg["stride"])
    do_zscore = bool(feat_cfg.get("zscore_per_window", True))
    fft_mode = str(feat_cfg.get("fft_mode", "power"))
    nperseg = int(feat_cfg["stft"]["nperseg"])
    noverlap = int(feat_cfg["stft"]["noverlap"])
    nfft = int(feat_cfg["stft"]["nfft"])
    img_hw = (
        int(feat_cfg["tf_image_size"][0]),
        int(feat_cfg["tf_image_size"][1]),
    )
    x_tf_dtype = np.float16 if str(feat_cfg.get("x_tf_dtype", "float16")) == "float16" else np.float32

    overlap_cfg = feat_cfg.get("overlap_sampling", {})
    if overlap_cfg.get("enabled", True):
        extra_offsets = overlap_cfg.get("offsets", None)
        if extra_offsets is None or len(extra_offsets) == 0:
            extra_offsets = make_overlap_offsets(stride)
        else:
            extra_offsets = [int(x) for x in extra_offsets if 0 < int(x) < stride]
    else:
        extra_offsets = []

    for condition in split_cfg.conditions:
        domain_id = domain_map[condition]

        for class_name in class_folders:
            is_normal = class_name == normal_class_folder
            need_n = split_cfg.normal_per_domain if is_normal else split_cfg.fault_per_class_per_domain
            class_label = label_map[class_name]

            files = list_class_files_for_condition(class_dir_map[class_name], condition)
            if len(files) == 0:
                raise FileNotFoundError(
                    f"类别 {class_name} 在工况 {condition} 下未找到 mat 文件"
                )

            signals = [load_pu_vibration_signal(fp) for fp in files]

            selected_pairs_per_file, file_debug_infos = sample_window_starts_per_file_strict_same_group(
                signals=signals,
                total_count=need_n,
                win_len=win_len,
                stride=stride,
                rng=rng,
                extra_offsets=extra_offsets,
            )

            class_condition_trace = {
                "split_name": split_cfg.name,
                "condition": condition,
                "domain_id": int(domain_id),
                "class_name": class_name,
                "class_label": int(class_label),
                "is_normal": bool(is_normal),
                "requested_sample_count": int(need_n),
                "files": [],
            }

            actual_count = 0

            for fp, sig, selected_pairs, dbg in zip(files, signals, selected_pairs_per_file, file_debug_infos):
                selected_starts = [int(x[0]) for x in selected_pairs]
                selected_modes = [str(x[1]) for x in selected_pairs]

                base_count = sum(1 for _, m in selected_pairs if m == "base")
                overlap_count = sum(1 for _, m in selected_pairs if m == "overlap")

                for s, _mode in selected_pairs:
                    window = sig[s: s + win_len].astype(np.float32, copy=False)
                    if do_zscore:
                        window = zscore_1d(window)

                    freq_feat = compute_freq_feature(window, fft_mode=fft_mode)
                    tf_feat = compute_tf_feature(
                        window,
                        nperseg=nperseg,
                        noverlap=noverlap,
                        nfft=nfft,
                        out_hw=img_hw,
                    ).astype(x_tf_dtype)

                    x_freq_list.append(freq_feat)
                    x_tf_list.append(tf_feat)
                    y_list.append(class_label)
                    d_list.append(domain_id)

                actual_count += len(selected_pairs)

                class_condition_trace["files"].append(
                    {
                        "file_path": str(fp),
                        "signal_length": int(dbg["signal_length"]),
                        "base_candidate_count": int(dbg["base_candidate_count"]),
                        "overlap_candidate_count": int(dbg["overlap_candidate_count"]),
                        "total_candidate_count": int(dbg["total_candidate_count"]),
                        "selected_count": int(len(selected_pairs)),
                        "selected_base_count": int(base_count),
                        "selected_overlap_count": int(overlap_count),
                        "selected_starts": selected_starts,
                        "selected_modes": selected_modes,
                    }
                )

            class_condition_trace["actual_sample_count"] = int(actual_count)
            trace_entries.append(class_condition_trace)

            # 预处理时直接做简要输出检查
            short_parts = []
            for item in class_condition_trace["files"]:
                short_parts.append(
                    f"{Path(item['file_path']).name}: "
                    f"baseCand={item['base_candidate_count']}, "
                    f"overlapCand={item['overlap_candidate_count']}, "
                    f"sel={item['selected_count']} "
                    f"(base={item['selected_base_count']}, overlap={item['selected_overlap_count']})"
                )

            preview = " | ".join(short_parts[:4])
            if len(short_parts) > 4:
                preview += f" | ... total_files={len(short_parts)}"

            print(
                f"[{split_cfg.name}] {condition} | {class_name} | "
                f"requested={need_n}, actual={actual_count}\n"
                f"  {preview}"
            )

    if len(x_freq_list) == 0:
        raise RuntimeError(f"split={split_cfg.name} 未采样到任何样本")

    x_freq = np.stack(x_freq_list, axis=0).astype(np.float32)
    x_tf = np.stack(x_tf_list, axis=0).astype(x_tf_dtype)
    y = np.asarray(y_list, dtype=np.int64)
    domain = np.asarray(d_list, dtype=np.int64)

    trace_info = {
        "split_name": split_cfg.name,
        "entries": trace_entries,
    }

    data = {
        "x_freq": x_freq,
        "x_tf": x_tf,
        "y": y,
        "domain": domain,
    }
    return data, trace_info


# ========================= 保存与汇总 =========================

def save_h5(out_path: Path, data: Dict[str, np.ndarray], attrs: Dict[str, Any]) -> None:
    ensure_dir(out_path.parent)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("x_freq", data=data["x_freq"], compression="gzip")
        f.create_dataset("x_tf", data=data["x_tf"], compression="gzip")
        f.create_dataset("y", data=data["y"], compression="gzip")
        f.create_dataset("domain", data=data["domain"], compression="gzip")
        for k, v in attrs.items():
            if isinstance(v, (dict, list)):
                f.attrs[k] = json.dumps(v, ensure_ascii=False)
            else:
                f.attrs[k] = v


def save_trace_json(out_path: Path, trace_info: Dict[str, Any]) -> None:
    ensure_dir(out_path.parent)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(trace_info, f, ensure_ascii=False, indent=2)


def summarize_split(name: str, data: Dict[str, np.ndarray]) -> str:
    y = data["y"]
    d = data["domain"]
    uniq_y, cnt_y = np.unique(y, return_counts=True)
    uniq_d, cnt_d = np.unique(d, return_counts=True)
    return (
        f"[{name}] N={len(y)} | x_freq={tuple(data['x_freq'].shape)} | x_tf={tuple(data['x_tf'].shape)}\n"
        f"  label_count={dict(zip(uniq_y.tolist(), cnt_y.tolist()))}\n"
        f"  domain_count={dict(zip(uniq_d.tolist(), cnt_d.tolist()))}"
    )


# ========================= 主函数 =========================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=str, required=True, help="YAML 配置文件路径")
    args = parser.parse_args()

    cfg = load_config(args.configs)
    seed = int(cfg.get("seed", 2026))
    set_seed(seed)

    root_dir = Path(cfg["dataset"]["root_dir"])
    output_dir = Path(cfg["output"]["output_dir"])
    task_name = str(cfg["output"]["task_name"])
    class_folders = list(cfg["dataset"]["class_folders"])
    normal_class_folder = str(cfg["dataset"]["normal_class_folder"])

    # 新增：拼接根目录 + 子文件夹路径
    task_dir = output_dir / task_name
    ensure_dir(task_dir)  # 自动创建子文件夹（依赖你文件中已有的 ensure_dir 函数）

    if normal_class_folder not in class_folders:
        raise ValueError("normal_class_folder 必须在 class_folders 中")

    split_cfgs, split_info = build_split_cfgs(cfg)

    all_conditions = (
        split_info["train_conditions"]
        + split_info["val_conditions"]
        + split_info["test_conditions"]
    )
    domain_map = {cond: i for i, cond in enumerate(all_conditions)}
    label_map = make_label_map(class_folders)

    print("=" * 88)
    print("PU IFDG preprocessing")
    print(f"root_dir      : {root_dir}")
    print(f"task_name     : {task_name}")
    print(f"class_folders : {class_folders}")
    print(f"normal_folder : {normal_class_folder}")
    print(f"domain_map    : {domain_map}")
    print(f"label_map     : {label_map}")
    print(f"available     : {split_info['available_conditions']}")
    print(f"train         : {split_info['train_conditions']}")
    print(f"val           : {split_info['val_conditions']}")
    print(f"test          : {split_info['test_conditions']}")
    print("=" * 88)

    split_seed_offsets = {"train": 11, "val": 23, "test": 37}
    for split_cfg in split_cfgs:
        split_seed = seed + split_seed_offsets[split_cfg.name]

        data, trace_info = collect_split(
            root_dir=root_dir,
            class_folders=class_folders,
            normal_class_folder=normal_class_folder,
            label_map=label_map,
            domain_map=domain_map,
            split_cfg=split_cfg,
            feat_cfg=cfg["feature"],
            seed=split_seed,
        )

        print(summarize_split(split_cfg.name, data))

        out_path = task_dir / f"{split_cfg.name}.h5"
        save_h5(
            out_path,
            data,
            attrs={
                "task_name": task_name,
                "split_name": split_cfg.name,
                "conditions": split_cfg.conditions,
                "label_map": label_map,
                "domain_map": domain_map,
                "normal_class_folder": normal_class_folder,
                "feature_cfg": cfg["feature"],
                "sampling_cfg": cfg["sampling"],
                "split_cfg": cfg["split"],
            },
        )
        print(f"saved -> {out_path}")

        trace_path = task_dir / f"{split_cfg.name}_trace.json"
        save_trace_json(trace_path, trace_info)
        print(f"trace saved -> {trace_path}")

    print("\n完成。")


if __name__ == "__main__":
    main()