#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHM 2009 spur8 -> Imbalanced Few-Shot Domain Generalization preprocessing

功能概述
--------
1. 读取 PHM 2009 spur8 数据目录（spur 1 ~ spur 8）
2. 仅使用每个 txt 的第 2 列（输出侧振动）
3. 按“工况域”进行划分：例如 30hz_High、35hz_High 等
4. 支持：
   - 3 源域 + 1 目标域（train/test）
   - 2 源域 + 1 验证域 + 1 目标域（train/val/test）
5. 支持不平衡小样本抽样：
   - 源域：normal_per_domain、fault_per_class_per_domain
   - 目标域：normal_per_domain、fault_per_class_per_domain
   - 验证域：可单独设置，也可复用 source/target 设置
6. 每个类别在每个工况下通常有 2 个文件，脚本会尽量均匀分配抽样数量
7. 输出 HDF5 文件，包含核心键：x_freq, x_tf, y, domain

输出数据结构
------------
x_freq: [N, 1, F] float32
x_tf  : [N, 1, H, W] float16/float32
y     : [N] int64
domain: [N] int64

依赖
----
pip install numpy scipy h5py pyyaml

使用方式
--------
python preprocess_phm_spur8.py --configs phm_spur8_config.yaml
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any

import h5py
import numpy as np
import yaml
from scipy import signal
from scipy.ndimage import zoom

from data.config import load_config


@dataclass
class SplitConfig:
    name: str
    conditions: List[str]
    normal_per_domain: int
    fault_per_class_per_domain: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_condition_from_filename(file_path: Path) -> str:
    """
    示例：spur 1_30hz_High_1.txt -> 30hz_High
    规则：取去后缀后的最后 3 段中的前 2 段作为工况名。
    """
    stem = file_path.stem
    parts = stem.split("_")
    if len(parts) < 4:
        raise ValueError(f"文件名无法解析工况: {file_path.name}")
    return "_".join(parts[-3:-1])


def parse_repeat_from_filename(file_path: Path) -> str:
    stem = file_path.stem
    parts = stem.split("_")
    if len(parts) < 4:
        raise ValueError(f"文件名无法解析重复编号: {file_path.name}")
    return parts[-1]


def robust_load_txt_second_column(file_path: Path) -> np.ndarray:
    """仅读取第二列（输出侧振动）。尽量兼容空格/制表符/逗号和可能的表头。"""
    loaders = [
        lambda p: np.loadtxt(p, dtype=np.float32),
        lambda p: np.loadtxt(p, dtype=np.float32, delimiter=","),
        lambda p: np.genfromtxt(p, dtype=np.float32),
        lambda p: np.genfromtxt(p, dtype=np.float32, delimiter=","),
        lambda p: np.genfromtxt(p, dtype=np.float32, skip_header=1),
        lambda p: np.genfromtxt(p, dtype=np.float32, delimiter=",", skip_header=1),
    ]
    data = None
    last_err = None
    for fn in loaders:
        try:
            data = fn(file_path)
            if data is not None and np.size(data) > 0:
                break
        except Exception as e:
            last_err = e
    if data is None or np.size(data) == 0:
        raise RuntimeError(f"读取失败: {file_path} | {last_err}")

    data = np.asarray(data)
    if data.ndim == 1:
        if data.shape[0] < 3:
            raise ValueError(f"文件列数不足 3 列: {file_path}")
        data = data.reshape(1, -1)

    if data.shape[1] < 2:
        raise ValueError(f"文件不存在第二列（输出侧振动）: {file_path}")

    sig = data[:, 1].astype(np.float32)
    sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
    return sig


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


def allocate_counts_evenly(total_count: int, n_buckets: int) -> List[int]:
    base = total_count // n_buckets
    rem = total_count % n_buckets
    return [base + (1 if i < rem else 0) for i in range(n_buckets)]

def make_overlap_offsets(stride: int) -> List[int]:
    """
    生成错位重叠采样的偏移量。
    要求：
    - 不为 0
    - 小于 stride
    - 产生的新窗口起点与原始滑窗起点不同
    """
    cands = []
    for x in [stride // 2, stride // 4, (3 * stride) // 4, stride // 3, (2 * stride) // 3]:
        if 0 < x < stride:
            cands.append(int(x))
    # 去重并保持顺序
    out = []
    seen = set()
    for x in cands:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def sliding_start_positions_with_offset(
    sig_len: int,
    win_len: int,
    stride: int,
    offset: int,
) -> np.ndarray:
    """
    错位采样：例如 offset = stride//2
    """
    if sig_len < win_len:
        return np.empty((0,), dtype=np.int64)

    max_start = sig_len - win_len
    if offset > max_start:
        return np.empty((0,), dtype=np.int64)

    return np.arange(offset, max_start + 1, stride, dtype=np.int64)


def build_candidate_pools_for_signal(
    sig_len: int,
    win_len: int,
    stride: int,
    extra_offsets: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回：
    - base_starts: 常规滑窗起点
    - overlap_starts: 错位重叠滑窗起点（与 base 不同）
    """
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


def _sample_from_pool_without_reuse(
    pool: np.ndarray,
    n_take: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 pool 中无放回抽样，返回：
    - chosen
    - remain
    """
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
    """
    在多个文件之间轮转补样本，尽量保持均衡。
    mode:
      - "base"
      - "overlap"
    """
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


def sample_window_starts_per_file_strict_same_files(
    signals: List[np.ndarray],
    total_count: int,
    win_len: int,
    stride: int,
    rng: np.random.Generator,
    extra_offsets: List[int],
) -> Tuple[List[List[Tuple[int, str]]], List[Dict[str, Any]]]:
    """
    严格只在当前 class-condition 对应的这些文件内部采样。

    采样顺序：
    1) 每个文件先按常规滑窗候选池尽量均分采样
    2) 若仍不足，则仍只在这些文件内部，从“剩余常规滑窗”中轮转补样本
    3) 若仍不足，则仍只在这些文件内部，从“错位重叠滑窗”中轮转补样本
    4) 若还不够，直接报错

    返回：
    - selected_pairs:
        每个文件一个 list，元素是 (start, mode)
        mode in {"base", "overlap"}
    - file_debug_infos:
        每个文件的调试信息
    """
    n_files = len(signals)
    quotas = allocate_counts_evenly(total_count, n_files)

    selected_pairs: List[List[Tuple[int, str]]] = [[] for _ in range(n_files)]

    remain_base_pools: List[np.ndarray] = []
    remain_overlap_pools: List[np.ndarray] = []
    file_debug_infos: List[Dict[str, Any]] = []

    deficits = 0

    # Step 1: 每个文件先按 quota 从自己的 base 池抽
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

    # Step 2: 不足时，仍只在这些文件内部，从剩余 base 池继续补
    if deficits > 0:
        deficits = fill_deficits_round_robin(
            remain_pools=remain_base_pools,
            selected_pairs=selected_pairs,
            deficits=deficits,
            mode="base",
            rng=rng,
        )

    # Step 3: 如果 base 全部耗尽，使用错位 overlap 池继续补
    if deficits > 0:
        deficits = fill_deficits_round_robin(
            remain_pools=remain_overlap_pools,
            selected_pairs=selected_pairs,
            deficits=deficits,
            mode="overlap",
            rng=rng,
        )

    # Step 4: 还不够，报错
    if deficits > 0:
        raise RuntimeError(
            f"当前 class-condition 对应文件内部样本不足："
            f"请求 {total_count}，但常规滑窗 + 错位重叠滑窗仍不足，仍缺少 {deficits} 个。"
        )

    # 排序，便于查看
    for i in range(n_files):
        selected_pairs[i] = sorted(selected_pairs[i], key=lambda x: x[0])

    return selected_pairs, file_debug_infos



def sample_window_starts_per_file(
    starts_list: List[np.ndarray],
    total_count: int,
    rng: np.random.Generator,
    allow_reallocate: bool = True,
    allow_replacement: bool = False,
) -> List[np.ndarray]:
    """
    尽量平均到每个文件中抽取窗口起点；若某文件不足，可在 allow_reallocate=True 时从其他文件补足。
    """
    n_files = len(starts_list)
    quotas = allocate_counts_evenly(total_count, n_files)
    picked: List[np.ndarray] = []
    deficits = 0

    for starts, q in zip(starts_list, quotas):
        available = len(starts)
        if q <= available:
            idx = rng.choice(available, size=q, replace=False)
            picked.append(np.sort(starts[idx]))
        else:
            if available == 0:
                picked.append(np.empty((0,), dtype=np.int64))
                deficits += q
                continue
            if allow_replacement:
                idx = rng.choice(available, size=q, replace=True)
                picked.append(np.sort(starts[idx]))
            else:
                idx = np.arange(available)
                picked.append(np.sort(starts[idx]))
                deficits += (q - available)

    if deficits > 0:
        if not allow_reallocate:
            raise RuntimeError(
                f"窗口数量不足，缺少 {deficits} 个；且不允许从其他文件补足。"
            )

        pools: List[Tuple[int, np.ndarray]] = []
        for i, starts in enumerate(starts_list):
            used = picked[i]
            if len(starts) == 0:
                continue
            if len(used) == 0:
                remain = starts
            else:
                remain = np.setdiff1d(starts, used, assume_unique=False)
            if len(remain) > 0:
                pools.append((i, remain))

        total_remain = sum(len(r) for _, r in pools)
        if total_remain < deficits and not allow_replacement:
            raise RuntimeError(
                f"窗口总数不足：仍缺少 {deficits - total_remain} 个，请减小抽样数或启用 replacement。"
            )

        while deficits > 0:
            if pools:
                i, remain = pools[0]
                take = min(deficits, len(remain))
                idx = rng.choice(len(remain), size=take, replace=False)
                extra = np.sort(remain[idx])
                picked[i] = np.sort(np.concatenate([picked[i], extra], axis=0))
                remain = np.setdiff1d(remain, extra, assume_unique=False)
                pools[0] = (i, remain)
                if len(remain) == 0:
                    pools.pop(0)
                deficits -= take
            elif allow_replacement:
                j = int(rng.integers(0, n_files))
                starts = starts_list[j]
                if len(starts) == 0:
                    continue
                extra = starts[rng.choice(len(starts), size=1, replace=True)]
                picked[j] = np.sort(np.concatenate([picked[j], extra], axis=0))
                deficits -= 1
            else:
                raise RuntimeError("无法完成补抽样。")

    return picked


def compute_freq_feature(
    window: np.ndarray,
    fft_mode: str = "power",
) -> np.ndarray:
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
    power = np.log1p(power)
    power = resize_2d(power.astype(np.float32), out_hw)
    return power[None, :, :]


def list_class_files_for_condition(class_dir: Path, condition: str) -> List[Path]:
    files = []
    for fp in sorted(class_dir.glob("*.txt")):
        try:
            cond = parse_condition_from_filename(fp)
        except Exception:
            continue
        if cond == condition:
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


def infer_available_conditions(root_dir: Path, class_folders: List[str]) -> List[str]:
    conds = set()
    for cls in class_folders:
        for fp in (root_dir / cls).glob("*.txt"):
            try:
                conds.add(parse_condition_from_filename(fp))
            except Exception:
                pass
    return sorted(conds)


def validate_conditions(
    available_conditions: List[str],
    source_conditions: List[str],
    target_condition: str,
    use_val: bool,
    val_condition: str | None,
) -> Tuple[List[str], List[str], List[str]]:
    avail = set(available_conditions)
    for cond in source_conditions + [target_condition]:
        if cond not in avail:
            raise ValueError(f"工况不存在于数据中: {cond}")
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
                    f"类别 {class_name} 在工况 {condition} 下未找到 txt 文件"
                )

            signals = [robust_load_txt_second_column(fp) for fp in files]

            # 新逻辑：严格只在当前这些文件内部采样，不向外借
            selected_pairs_per_file, file_debug_infos = sample_window_starts_per_file_strict_same_files(
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

                for s, mode in selected_pairs:
                    window = sig[s : s + win_len].astype(np.float32, copy=False)
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

            # 直接在预处理时输出检查
            short_parts = []
            for item in class_condition_trace["files"]:
                short_parts.append(
                    f"{Path(item['file_path']).name}: "
                    f"baseCand={item['base_candidate_count']}, "
                    f"overlapCand={item['overlap_candidate_count']}, "
                    f"sel={item['selected_count']} "
                    f"(base={item['selected_base_count']}, overlap={item['selected_overlap_count']})"
                )
            print(
                f"[{split_cfg.name}] {condition} | {class_name} | "
                f"requested={need_n}, actual={actual_count}\n"
                f"  " + " | ".join(short_parts)
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

    return {
        "x_freq": x_freq,
        "x_tf": x_tf,
        "y": y,
        "domain": domain,
    }, trace_info

def save_h5(
    out_path: Path,
    data: Dict[str, np.ndarray],
    attrs: Dict[str, Any],
) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=str, required=True, help="YAML 配置文件路径")
    args = parser.parse_args()

    cfg = load_config(args.configs)
    set_seed(int(cfg.get("seed", 2026)))

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

    all_conditions = split_info["train_conditions"] + split_info["val_conditions"] + split_info["test_conditions"]
    domain_map = {cond: i for i, cond in enumerate(all_conditions)}
    label_map = make_label_map(class_folders)

    print("=" * 88)
    print("PHM spur8 IFDG preprocessing")
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
        split_seed = int(cfg.get("seed", 2026)) + split_seed_offsets[split_cfg.name]
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
        print(f"saved -> {out_path}")

    print("\n完成。")


if __name__ == "__main__":
    main()