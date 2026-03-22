#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# 文件功能概述
# -----------------------------------------------------------------------------
# 本脚本用于将 CWRU（Case Western Reserve University）轴承数据集预处理为
# “不平衡小样本域泛化”实验可直接使用的 HDF5 数据文件。
#
# 本脚本面向你当前指定的 CWRU 子集：
#   1) 故障数据来自：
#      "12k Drive End Bearing Fault Data"
#   2) 正常数据来自：
#      "Normal Baseline Data"
#   3) 仅使用驱动端时域信号字段：
#      X***_DE_time
#
# 你当前定义的类别体系是：
#   - 1 个正常类：Normal
#   - 9 个故障类：
#       IR007, B007, OR007,
#       IR014, B014, OR014,
#       IR021, B021, OR021
#   共 10 类
#
# 工况定义：
#   当前按电机负载划分 4 个工况域：
#     0HP, 1HP, 2HP, 3HP
#   你可在配置文件中自由指定：
#     - 哪些工况作为 source_conditions
#     - 哪个工况作为 target_condition
#     - 是否划分 val_condition
#
# 最终每个 split 会生成：
#   1) .h5 文件：包含 x_freq, x_tf, y, domain 四个核心数据集
#   2) _trace.json：记录每个类、每个工况、每个原始 mat 文件的抽样轨迹
#
# 核心流程
# -----------------------------------------------------------------------------
# 1. 读取 YAML 配置文件
# 2. 根据 class_file_map 找到每个“类别-工况”对应的唯一 mat 文件
# 3. 从 mat 中提取 X***_DE_time 一维驱动端时域信号
# 4. 对每个 split（train / val / test）逐工况、逐类别抽样
# 5. 先用常规滑窗生成候选窗口；若不足，则只在同一个原始文件内部做错位重叠补样
# 6. 每个窗样本生成：
#      - x_freq：rFFT -> power/magnitude -> log1p
#      - x_tf  ：STFT power -> log1p -> resize
# 7. 保存为 h5，并同步保存 trace.json
#
# 重要设计原则
# -----------------------------------------------------------------------------
# 1. 一个类别-一个工况只允许使用该配置映射到的那个 mat 文件
# 2. 样本不足时不能从其他文件、其他工况、其他类别借数据
# 3. 只能在原文件内部做“错位重叠采样”
# 4. 错位重叠采样得到的新窗口起点必须与原常规滑窗起点不同
# 5. 预处理结束后终端会打印简要检查；详细轨迹保存到 trace.json
#
# 主要函数及相互关系
# -----------------------------------------------------------------------------
# set_seed / load_yaml / ensure_dir
#   基础工具函数：设置随机种子、读取配置、创建目录
#
# find_cwru_mat_file
#   根据文件编号（如 105、223、97）在指定目录下定位 mat 文件
#
# load_cwru_de_signal
#   从单个 mat 文件中提取 X***_DE_time 驱动端时域信号
#
# sliding_start_positions / build_candidate_pools_for_signal
#   生成常规滑窗起点与错位重叠滑窗起点
#
# sample_window_starts_single_file
#   对单个原始文件进行严格抽样：
#   先用常规滑窗，不够再用同文件内部的 overlap 候选池补样
#
# compute_freq_feature / compute_tf_feature
#   将一个时域窗样本转换为频域特征和时频图特征
#
# build_split_cfgs
#   根据 source / target / val 的配置构造 train / val / test 划分
#
# collect_split
#   收集整个 split 的数据，并记录 trace 信息
#
# save_h5 / save_trace_json
#   保存 HDF5 与 trace.json
#
# 重要参数（最关键）
# -----------------------------------------------------------------------------
# win_len / stride
#   控制滑窗长度与常规步长
#
# source_conditions / target_condition / use_val / val_condition
#   控制域划分方式
#
# normal_per_domain / fault_per_class_per_domain
#   控制每个域中正常类与故障类各抽多少样本
#
# overlap_sampling.enabled / overlap_sampling.offsets
#   控制样本不足时是否启用同文件内错位重叠采样
#
# class_file_map
#   定义每个“类别-工况”对应哪个 mat 文件编号，是本脚本最关键的映射参数
# =============================================================================

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import h5py
import numpy as np
import yaml
from scipy import signal
from scipy.io import loadmat
from scipy.ndimage import zoom
from .config import load_config


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


# ========================= CWRU 文件定位与信号提取 =========================

def normalize_file_id(file_id: str) -> str:
    return str(file_id).strip().replace(".mat", "")


def find_cwru_mat_file(folder: Path, file_id: str) -> Path:
    """
    在指定目录下查找对应编号的 .mat 文件。
    例如：
      file_id = "105" -> 105.mat
      file_id = "97"  -> 97.mat
    """
    file_id = normalize_file_id(file_id)

    direct = folder / f"{file_id}.mat"
    if direct.exists():
        return direct

    # 兜底：有些目录可能名字或扩展大小写不同
    pattern = re.compile(rf"^{re.escape(file_id)}\.mat$", re.IGNORECASE)
    matches = [p for p in folder.iterdir() if p.is_file() and pattern.match(p.name)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(f"目录中存在多个匹配文件: folder={folder}, file_id={file_id}, matches={matches}")

    raise FileNotFoundError(f"未找到对应 mat 文件: folder={folder}, file_id={file_id}")


def find_de_key(mat_dict: Dict[str, Any], file_id: str) -> str:
    """
    优先寻找 X***_DE_time 字段。
    常见形式：
      X105_DE_time
      X097_DE_time
      X3001_DE_time
    """
    file_id = normalize_file_id(file_id)
    int_id = int(file_id)

    candidate_keys = [
        f"X{file_id}_DE_time",
        f"X{int_id}_DE_time",
        f"X{int_id:03d}_DE_time",
    ]
    for k in candidate_keys:
        if k in mat_dict:
            return k

    # 正则兜底
    pat = re.compile(rf"^X0*{int_id}_DE_time$", re.IGNORECASE)
    hits = [k for k in mat_dict.keys() if pat.match(k)]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise RuntimeError(f"找到多个可能的 DE 字段: file_id={file_id}, keys={hits}")

    # 再兜底：找任意 *_DE_time
    de_hits = [k for k in mat_dict.keys() if k.lower().endswith("_de_time")]
    if len(de_hits) == 1:
        return de_hits[0]
    if len(de_hits) > 1:
        raise RuntimeError(f"存在多个 *_DE_time 字段，无法唯一确定: file_id={file_id}, keys={de_hits}")

    raise RuntimeError(f"未找到驱动端时域字段 X***_DE_time: file_id={file_id}")


def load_cwru_de_signal(file_path: Path, file_id: str) -> np.ndarray:
    """
    从单个 CWRU mat 文件中提取驱动端时域信号 X***_DE_time
    """
    mat = loadmat(file_path, squeeze_me=True)
    de_key = find_de_key(mat, file_id)
    sig = np.asarray(mat[de_key], dtype=np.float32).reshape(-1)
    sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)

    if sig.size < 1024:
        raise RuntimeError(f"提取出的 DE 信号长度过短: {file_path}, len={sig.size}")

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


# ========================= 单文件抽样逻辑 =========================

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


def sample_window_starts_single_file(
    sig_len: int,
    total_count: int,
    win_len: int,
    stride: int,
    rng: np.random.Generator,
    extra_offsets: List[int],
) -> Tuple[List[Tuple[int, str]], Dict[str, Any]]:
    """
    单文件严格抽样：
    1) 先从 base 候选池抽样
    2) 若不足，再从同文件的 overlap 候选池补样
    3) 若仍不足，则报错
    """
    base_pool, overlap_pool = build_candidate_pools_for_signal(
        sig_len=sig_len,
        win_len=win_len,
        stride=stride,
        extra_offsets=extra_offsets,
    )

    selected_pairs: List[Tuple[int, str]] = []

    chosen_base, remain_base = _sample_from_pool_without_reuse(
        pool=base_pool,
        n_take=total_count,
        rng=rng,
    )
    for s in chosen_base:
        selected_pairs.append((int(s), "base"))

    deficits = total_count - len(chosen_base)

    if deficits > 0:
        chosen_overlap, remain_overlap = _sample_from_pool_without_reuse(
            pool=overlap_pool,
            n_take=deficits,
            rng=rng,
        )
        for s in chosen_overlap:
            selected_pairs.append((int(s), "overlap"))
        deficits -= len(chosen_overlap)
    else:
        remain_overlap = overlap_pool

    if deficits > 0:
        raise RuntimeError(
            f"单文件样本不足：请求 {total_count}，但常规滑窗 + 错位重叠滑窗仍不足，仍缺少 {deficits} 个。"
        )

    selected_pairs = sorted(selected_pairs, key=lambda x: x[0])

    dbg = {
        "signal_length": int(sig_len),
        "base_candidate_count": int(len(base_pool)),
        "overlap_candidate_count": int(len(overlap_pool)),
        "total_candidate_count": int(len(base_pool) + len(overlap_pool)),
        "remain_base_candidate_count": int(len(remain_base)),
        "remain_overlap_candidate_count": int(len(remain_overlap)),
    }
    return selected_pairs, dbg


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
                f"工况不存在于配置中: {cond}\n"
                f"当前可用工况有: {sorted(avail)}"
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
    elif split_name == "val":
        key = "val"
    elif split_name == "test":
        key = "target"
    else:
        raise ValueError(f"未知 split_name: {split_name}")

    normal = int(sampling[key]["normal_per_domain"])
    fault = int(sampling[key]["fault_per_class_per_domain"])
    return normal, fault


def build_split_cfgs(cfg: Dict[str, Any]) -> Tuple[List[SplitConfig], Dict[str, Any]]:
    sp_cfg = cfg["split"]
    available_conditions = list(cfg["dataset"]["conditions"])

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


# ========================= 数据收集与 trace 记录 =========================

def get_class_source_folder(cfg: Dict[str, Any], class_name: str) -> Path:
    ds_cfg = cfg["dataset"]
    root_dir = Path(ds_cfg["root_dir"])
    class_source_map = ds_cfg["class_source_map"]
    source_tag = class_source_map[class_name]

    if source_tag == "normal":
        return root_dir / ds_cfg["normal_subdir"]
    elif source_tag == "fault":
        return root_dir / ds_cfg["fault_subdir"]
    else:
        raise ValueError(f"未知 class_source_map 标记: class={class_name}, source_tag={source_tag}")


def collect_split(
    cfg: Dict[str, Any],
    label_map: Dict[str, int],
    domain_map: Dict[str, int],
    split_cfg: SplitConfig,
    seed: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    rng = np.random.default_rng(seed)

    ds_cfg = cfg["dataset"]
    feat_cfg = cfg["feature"]
    class_names = list(ds_cfg["class_names"])
    normal_class_name = str(ds_cfg["normal_class_name"])
    class_file_map = ds_cfg["class_file_map"]

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

        for class_name in class_names:
            is_normal = class_name == normal_class_name
            need_n = split_cfg.normal_per_domain if is_normal else split_cfg.fault_per_class_per_domain
            class_label = label_map[class_name]

            if class_name not in class_file_map:
                raise KeyError(f"class_file_map 中缺少类别映射: {class_name}")
            if condition not in class_file_map[class_name]:
                raise KeyError(f"class_file_map 中缺少工况映射: class={class_name}, condition={condition}")

            file_id = str(class_file_map[class_name][condition])
            folder = get_class_source_folder(cfg, class_name)
            file_path = find_cwru_mat_file(folder, file_id)
            sig = load_cwru_de_signal(file_path, file_id=file_id)

            selected_pairs, dbg = sample_window_starts_single_file(
                sig_len=len(sig),
                total_count=need_n,
                win_len=win_len,
                stride=stride,
                rng=rng,
                extra_offsets=extra_offsets,
            )

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

            trace_entry = {
                "split_name": split_cfg.name,
                "condition": condition,
                "domain_id": int(domain_id),
                "class_name": class_name,
                "class_label": int(class_label),
                "is_normal": bool(is_normal),
                "requested_sample_count": int(need_n),
                "actual_sample_count": int(len(selected_pairs)),
                "files": [
                    {
                        "file_id": file_id,
                        "file_path": str(file_path),
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
                ],
            }
            trace_entries.append(trace_entry)

            # 预处理后直接做简要输出检查
            print(
                f"[{split_cfg.name}] {condition} | {class_name} | "
                f"requested={need_n}, actual={len(selected_pairs)}\n"
                f"  {file_path.name}: baseCand={dbg['base_candidate_count']}, "
                f"overlapCand={dbg['overlap_candidate_count']}, "
                f"sel={len(selected_pairs)} (base={base_count}, overlap={overlap_count})"
            )

    if len(x_freq_list) == 0:
        raise RuntimeError(f"split={split_cfg.name} 未采样到任何样本")

    x_freq = np.stack(x_freq_list, axis=0).astype(np.float32)
    x_tf = np.stack(x_tf_list, axis=0).astype(x_tf_dtype)
    y = np.asarray(y_list, dtype=np.int64)
    domain = np.asarray(d_list, dtype=np.int64)

    data = {
        "x_freq": x_freq,
        "x_tf": x_tf,
        "y": y,
        "domain": domain,
    }
    trace_info = {
        "split_name": split_cfg.name,
        "entries": trace_entries,
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

    ds_cfg = cfg["dataset"]

    output_cfg = cfg["output"]
    class_names = list(ds_cfg["class_names"])
    normal_class_name = str(ds_cfg["normal_class_name"])

    if normal_class_name not in class_names:
        raise ValueError("normal_class_name 必须在 class_names 中")

    split_cfgs, split_info = build_split_cfgs(cfg)

    all_conditions = (
        split_info["train_conditions"]
        + split_info["val_conditions"]
        + split_info["test_conditions"]
    )
    domain_map = {cond: i for i, cond in enumerate(all_conditions)}
    label_map = {cls: i for i, cls in enumerate(class_names)}

    print("=" * 88)
    print("CWRU IFDG preprocessing")
    print(f"root_dir        : {ds_cfg['root_dir']}")
    print(f"fault_subdir    : {ds_cfg['fault_subdir']}")
    print(f"normal_subdir   : {ds_cfg['normal_subdir']}")
    print(f"task_name       : {output_cfg['task_name']}")
    print(f"class_names     : {class_names}")
    print(f"normal_class    : {normal_class_name}")
    print(f"domain_map      : {domain_map}")
    print(f"label_map       : {label_map}")
    print(f"available       : {split_info['available_conditions']}")
    print(f"train           : {split_info['train_conditions']}")
    print(f"val             : {split_info['val_conditions']}")
    print(f"test            : {split_info['test_conditions']}")
    print("=" * 88)


    output_dir = Path(output_cfg["output_dir"])
    task_name = str(output_cfg["task_name"])

    # 新增：拼接根目录 + 子文件夹路径
    task_dir = output_dir / task_name
    ensure_dir(task_dir)  # 自动创建子文件夹（依赖你文件中已有的 ensure_dir 函数）

    split_seed_offsets = {"train": 11, "val": 23, "test": 37}
    for split_cfg in split_cfgs:
        split_seed = seed + split_seed_offsets[split_cfg.name]

        data, trace_info = collect_split(
            cfg=cfg,
            label_map=label_map,
            domain_map=domain_map,
            split_cfg=split_cfg,
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
                "normal_class_name": normal_class_name,
                "feature_cfg": cfg["feature"],
                "sampling_cfg": cfg["sampling"],
                "split_cfg": cfg["split"],
                "dataset_cfg": {
                    "root_dir": ds_cfg["root_dir"],
                    "fault_subdir": ds_cfg["fault_subdir"],
                    "normal_subdir": ds_cfg["normal_subdir"],
                    "conditions": ds_cfg["conditions"],
                    "class_names": ds_cfg["class_names"],
                    "normal_class_name": ds_cfg["normal_class_name"],
                    "class_source_map": ds_cfg["class_source_map"],
                    "class_file_map": ds_cfg["class_file_map"],
                },
            },
        )
        print(f"saved -> {out_path}")

        trace_path = task_dir / f"{split_cfg.name}_trace.json"
        save_trace_json(trace_path, trace_info)
        print(f"trace saved -> {trace_path}")

    print("\n完成。")


if __name__ == "__main__":
    main()