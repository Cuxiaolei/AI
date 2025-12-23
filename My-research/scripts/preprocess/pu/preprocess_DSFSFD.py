# -*- coding: utf-8 -*-
"""
PU raw .mat  -->  Fusion inputs (time 512 + fft 256 + tf 224x224 RGB)

Output per domain folder (condition):
  out_root/N15_M07_F10/Healthy----1.mat + Healthy----1.jpg ... Healthy----20.*
  out_root/N15_M07_F10/KAxx----1.mat + KAxx----1.jpg ... KAxx----20.*
  ... total 10 classes (1 healthy + 9 faults), each condition has 20 samples per class.

.mat keys:
  - DE_time: (512, 1)
  - FFT_data: (256, 1)

.jpg:
  - RGB (224x224)


"""

from __future__ import annotations

import re
import csv
import json


from PIL import Image
from scipy.io import loadmat, savemat
from scipy.signal import stft
from pathlib import Path
import numpy as np

# ---------------- fixed (match your fusion model expectations) ----------------
FS = 64000
WIN_LEN = 512
FFT_DIM = 256
TF_SIZE = 224

# STFT settings for TF image (works without pywt)
N_PER_SEG = 128
N_OVERLAP = 96
N_FFT = 256

# PU filename: Nxx_Myy_Fzz_Kabc_i.mat
FNAME_RE = re.compile(r"^(N\d+_M\d+_F\d+)_(K[A-Z]*\d+)_(\d+)\.mat$", re.IGNORECASE)
HEALTHY_SET = {f"K{i:03d}" for i in range(1, 7)}


# ---------------- low-level helpers ----------------
def parse_name(p: Path):
    m = FNAME_RE.match(p.name)
    if not m:
        return None
    return m.group(1), m.group(2), int(m.group(3))  # cond, bearing, meas_idx


def to_float_1d(x) -> np.ndarray:
    x = np.asarray(x)
    x = np.squeeze(x)
    if x.ndim != 1:
        x = x.reshape(-1)
    return x.astype(np.float32, copy=False)


def extract_vibration_from_mat_v70(mat_obj) -> np.ndarray:
    """
    Typical PU v7.0 struct: fields Info, X, Y, Description
    Y: struct array with fields like Name/Data.
    Prefer channel whose Name contains vibration/vib/acc.
    Fallback to 7th channel (often vibration_1), then to the longest 1D.
    """
    Y = mat_obj.get("Y", None)
    if Y is None:
        raise ValueError("No field 'Y' found in mat.")

    chans = Y.ravel().tolist() if isinstance(Y, np.ndarray) else [Y]

    def _get(o, a):
        return getattr(o, a, None) if hasattr(o, a) else None

    # 1) by Name keyword
    for ch in chans:
        name = _get(ch, "Name") or _get(ch, "name")
        data = _get(ch, "Data") or _get(ch, "data")
        if data is None:
            continue
        name_str = str(name).lower() if name is not None else ""
        if any(k in name_str for k in ["vibration", "vib", "acc"]):
            return to_float_1d(data)

    # 2) fallback: 7th channel
    try:
        ch = chans[6]
        data = _get(ch, "Data") or _get(ch, "data")
        if data is not None:
            return to_float_1d(data)
    except Exception:
        pass

    # 3) longest 1D
    best = None
    best_len = -1
    for ch in chans:
        data = _get(ch, "Data") or _get(ch, "data")
        if data is None:
            continue
        v = to_float_1d(data)
        if v.shape[0] > best_len:
            best = v
            best_len = v.shape[0]
    if best is None:
        raise ValueError("Cannot locate vibration channel in 'Y'.")
    return best



def load_vibration_signal(mat_path: Path, allow_v73: bool = False) -> np.ndarray:
    mat_path = Path(mat_path)
    m = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)

    # 1) 找顶层 struct（你的 mat 顶层变量名 = 文件名 stem）
    top_struct = None
    for k, v in m.items():
        if k.startswith("__"):
            continue
        if hasattr(v, "_fieldnames") and ("Y" in v._fieldnames):
            top_struct = v
            break
    if top_struct is None:
        keys = [k for k in m.keys() if not k.startswith("__")]
        raise ValueError(f"No top-level struct with field 'Y' found. top-level keys={keys}")

    Y = getattr(top_struct, "Y")

    # 2) Y 通道列表
    chans = Y.ravel().tolist() if isinstance(Y, np.ndarray) else [Y]

    def _get(o, a):
        return getattr(o, a, None) if hasattr(o, a) else None

    def _get_name_data(ch):
        name = _get(ch, "Name")
        if name is None:
            name = _get(ch, "name")

        data = _get(ch, "Data")
        if data is None:
            data = _get(ch, "data")
        return name, data

    # 3) 优先按 Name 关键词找 vibration
    for ch in chans:
        name, data = _get_name_data(ch)
        if data is None:
            continue
        name_str = str(name).lower() if name is not None else ""
        if any(s in name_str for s in ["vibration", "vib", "acc"]):
            return to_float_1d(data)

    # 4) 兜底：第7通道 vibration_1
    if len(chans) >= 7:
        _, data = _get_name_data(chans[6])
        if data is not None:
            return to_float_1d(data)

    # 5) 再兜底：取最长 1D
    best, best_n = None, -1
    for ch in chans:
        _, data = _get_name_data(ch)
        if data is None:
            continue
        v = to_float_1d(data)
        if v.size > best_n:
            best, best_n = v, v.size
    if best is None:
        raise ValueError("Cannot locate any valid signal in Y.")
    return best


def pick_window(sig: np.ndarray, win_len: int, policy: str, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """
    Return x(win_len), start
    policy: 'center' | 'random' | 'head'
    """
    if sig.shape[0] < win_len:
        raise ValueError(f"Signal too short: {sig.shape[0]} < {win_len}")

    max_start = sig.shape[0] - win_len
    if policy == "center":
        start = max_start // 2
    elif policy == "head":
        start = 0
    elif policy == "random":
        start = int(rng.integers(0, max_start + 1))
    else:
        raise ValueError(f"Unknown policy: {policy}")

    x = sig[start:start + win_len].astype(np.float32)
    x = x - float(np.mean(x))  # remove DC
    return x, start


def make_fft256(x512: np.ndarray) -> np.ndarray:
    X = np.abs(np.fft.rfft(x512, n=WIN_LEN)).astype(np.float32)  # 257 bins
    return X[1:1 + FFT_DIM]  # drop DC -> 256


def make_tf_image(x512: np.ndarray) -> Image.Image:
    f, t, Z = stft(
        x512, fs=FS, nperseg=N_PER_SEG, noverlap=N_OVERLAP, nfft=N_FFT,
        boundary=None, padded=False
    )
    S = np.log1p(np.abs(Z).astype(np.float32))
    mn, mx = float(S.min()), float(S.max())
    S = (S - mn) / (mx - mn + 1e-8)
    img = (S * 255.0).astype(np.uint8)
    return Image.fromarray(img, mode="L").resize((TF_SIZE, TF_SIZE), Image.BICUBIC).convert("RGB")


def save_pair(out_dir: Path, class_name: str, meas_idx: int, x512: np.ndarray, fft256: np.ndarray, tf_img: Image.Image):
    stem = f"{class_name}----{meas_idx}"
    savemat(str(out_dir / f"{stem}.mat"), {
        "DE_time": x512.reshape(WIN_LEN, 1),
        "FFT_data": fft256.reshape(FFT_DIM, 1),
    })
    tf_img.save(str(out_dir / f"{stem}.jpg"), quality=95)


# ---------------- main pipeline ----------------
def build_dataset(
    raw_root: Path,
    out_root: Path,
    conditions: list[str],
    healthy_pick: str,
    num_faults: int,
    fault_picks: list[str] | None,
    seed: int,
    window_policy: str = "center",
    allow_v73: bool = False,
):
    out_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # scan all mats
    items = []
    for p in raw_root.rglob("*.mat"):
        info = parse_name(p)
        if info is None:
            continue
        cond, bearing, meas_idx = info
        if cond in conditions:
            items.append((p, cond, bearing, meas_idx))

    if not items:
        raise RuntimeError("No PU mats matched pattern Nxx_Myy_Fzz_Kxxx_i.mat under selected conditions.")

    all_bearings = sorted({b for _, _, b, _ in items})
    if healthy_pick not in all_bearings:
        raise RuntimeError(f"healthy_pick={healthy_pick} not found in dataset bearings.")

    fault_candidates = sorted([b for b in all_bearings if b not in HEALTHY_SET])

    if fault_picks is None:
        if len(fault_candidates) < num_faults:
            raise RuntimeError(f"Need {num_faults} fault bearings but only {len(fault_candidates)} available.")
        fault_picks = rng.choice(fault_candidates, size=num_faults, replace=False).tolist()
    else:
        # sanity: ensure all exist and are not healthy
        for b in fault_picks:
            if b not in all_bearings:
                raise RuntimeError(f"fault_picks contains {b} which is not found in dataset.")
            if b in HEALTHY_SET:
                raise RuntimeError(f"fault_picks contains healthy bearing {b}. Remove it.")

        if len(fault_picks) != num_faults:
            raise RuntimeError(f"fault_picks length ({len(fault_picks)}) must equal num_faults ({num_faults}).")

    selected = [healthy_pick] + fault_picks

    bearing_to_class = {healthy_pick: "Healthy"}
    for b in fault_picks:
        bearing_to_class[b] = b  # keep original id as class name

    # index raw mats by (cond, bearing, meas_idx)
    by_key = {}
    for p, cond, bearing, meas_idx in items:
        if bearing in selected:
            by_key[(cond, bearing, meas_idx)] = p

    # write meta + manifest
    meta_dir = out_root / "meta"
    meta_dir.mkdir(exist_ok=True)
    with open(meta_dir / "selected_classes.json", "w", encoding="utf-8") as f:
        json.dump({
            "seed": seed,
            "conditions": conditions,
            "healthy_pick": healthy_pick,
            "fault_picks": fault_picks,
            "classes": [bearing_to_class[b] for b in selected],
            "bearing_to_class": bearing_to_class,
            "window_policy": window_policy,
            "note": "Each (class, domain) uses original 20 measurement files -> outputs ----1..20",
        }, f, ensure_ascii=False, indent=2)

    manifest_path = meta_dir / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["domain", "bearing", "class_name", "meas_idx", "src_mat", "start", "win_len"])

        # build per condition(domain)
        for cond in conditions:
            out_dir = out_root / cond
            out_dir.mkdir(exist_ok=True)

            for bearing in selected:
                class_name = bearing_to_class[bearing]

                # MUST have 20 measurements: idx 1..20
                for meas_idx in range(1, 21):
                    src = by_key.get((cond, bearing, meas_idx), None)
                    if src is None:
                        raise RuntimeError(f"Missing raw file: cond={cond}, bearing={bearing}, idx={meas_idx}")

                    sig = load_vibration_signal(src, allow_v73=allow_v73)
                    x512, start = pick_window(sig, WIN_LEN, window_policy, rng)
                    fft256 = make_fft256(x512)
                    tf_img = make_tf_image(x512)
                    save_pair(out_dir, class_name, meas_idx, x512, fft256, tf_img)

                    w.writerow([cond, bearing, class_name, meas_idx, src.name, start, WIN_LEN])

    print("DONE.")
    print("Output root:", str(out_root))
    print("Meta:", str(meta_dir / "selected_classes.json"))
    print("Manifest:", str(meta_dir / "manifest.csv"))


def main():
    # ===================== 你只需要改这里（参数区） =====================
    RAW_ROOT = Path(r"D:\user\dataSet\！！工业旋转轴承数据集\德国帕德博恩轴承数据集")             # 改成你的 PU 原始数据根目录（下面有32个轴承文件夹）
    OUT_ROOT = Path(r"D:\user\code\AI\My-research\data\pu\pu_DSFSFD")    # 输出目录（自动创建）

    CONDITIONS = ["N15_M07_F10", "N09_M07_F10", "N15_M01_F10", "N15_M07_F04"]

    SEED = 2025
    HEALTHY_PICK = "K001"     # 选一个健康轴承映射为 "Healthy" 类
    NUM_FAULTS = 9            # 选 9 个故障类

    # 方式1：手动固定 9 个故障轴承（推荐，保证复现实验）
    # FAULT_PICKS = ["KA01","KA03","KA04","KI04","KI14","KB23","KA15","KI16","KA22"]

    # 方式2：留空(None) -> 根据 SEED 从所有故障轴承里随机抽 9 个
    FAULT_PICKS = None

    # 从每个 4s 信号里取 512 点窗口的策略：center / random / head
    WINDOW_POLICY = "center"

    # 如果你的 mat 是 MATLAB v7.3（loadmat 读不了），才改成 True
    ALLOW_V73 = False
    # ===================================================================

    build_dataset(
        raw_root=RAW_ROOT,
        out_root=OUT_ROOT,
        conditions=CONDITIONS,
        healthy_pick=HEALTHY_PICK,
        num_faults=NUM_FAULTS,
        fault_picks=FAULT_PICKS,
        seed=SEED,
        window_policy=WINDOW_POLICY,
        allow_v73=ALLOW_V73,
    )


if __name__ == "__main__":
    main()
