# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from scipy.io import loadmat
from PIL import Image

try:
    from torchvision import transforms
except Exception:
    transforms = None


# -----------------------------
# Utils
# -----------------------------
def _list_mat_basenames(domain_dir: str) -> List[str]:
    out = []
    for fn in os.listdir(domain_dir):
        if fn.lower().endswith(".mat"):
            out.append(os.path.splitext(fn)[0])
    out.sort()
    return out


def _parse_class_from_base(base: str) -> str:
    # 兼容：K001----xxx / K001_xxx / 纯K001
    if "----" in base:
        return base.split("----")[0]
    if "_" in base:
        return base.split("_")[0]
    return base


def _read_mat_features(mat_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    读取 DSFSFD 预处理 mat：
    - DE_time: (512,) time-domain vector
    - FFT_data: (256,) freq-domain vector
    """
    m = loadmat(mat_path)
    if "DE_time" not in m or "FFT_data" not in m:
        keys = [k for k in m.keys() if not k.startswith("__")]
        raise KeyError(f"Missing DE_time/FFT_data in {mat_path}. keys={keys}")

    de = np.asarray(m["DE_time"]).squeeze()
    fft = np.asarray(m["FFT_data"]).squeeze()

    de = de.reshape(-1).astype(np.float32)
    fft = fft.reshape(-1).astype(np.float32)
    return de, fft


@dataclass
class ZScoreScaler:
    mean: np.ndarray
    std: np.ndarray
    eps: float = 1e-6

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (self.std + self.eps)


def fit_concat_zscore(domains: List["DomainIndex"], use_time: bool, use_fft: bool) -> Optional[ZScoreScaler]:
    """
    模仿你原 DSFSFD：把 (DE_time, FFT_data) 拼起来一起做 z-score。
    """
    if not (use_time or use_fft):
        return None

    feats = []
    for d in domains:
        if use_time and use_fft:
            feats.append(np.concatenate([d.de, d.fft], axis=1))
        elif use_time:
            feats.append(d.de)
        else:
            feats.append(d.fft)

    X = np.concatenate(feats, axis=0).astype(np.float64)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return ZScoreScaler(mean=mean.astype(np.float32), std=std.astype(np.float32))


# -----------------------------
# Domain index (cache vectors, lazy-load images)
# -----------------------------
class DomainIndex:
    def __init__(
        self,
        root_dir: str,
        domain: str,
        class_to_id: Dict[str, int],
        use_time: bool,
        use_fft: bool,
        use_tf: bool,
    ):
        self.root_dir = root_dir
        self.domain = domain
        self.domain_dir = os.path.join(root_dir, domain)
        if not os.path.isdir(self.domain_dir):
            raise FileNotFoundError(f"Domain dir not found: {self.domain_dir}")

        self.use_time = use_time
        self.use_fft = use_fft
        self.use_tf = use_tf

        bases = _list_mat_basenames(self.domain_dir)

        basenames: List[str] = []
        labels: List[int] = []
        mat_paths: List[str] = []
        jpg_paths: List[str] = []

        de_list: List[np.ndarray] = []
        fft_list: List[np.ndarray] = []

        for base in bases:
            cls = _parse_class_from_base(base)
            if cls not in class_to_id:
                # 不在你选定的 10 类里就跳过（匹配 selected_classes.json 的情况）
                continue

            mp = os.path.join(self.domain_dir, base + ".mat")
            jp = os.path.join(self.domain_dir, base + ".jpg")

            if self.use_tf and (not os.path.isfile(jp)):
                # tf 模态开启时必须有 jpg
                continue

            # 只缓存向量特征（省显存/内存），图片在 episode 时按需加载
            de, fft = _read_mat_features(mp)

            if use_time and de.shape[0] != 512:
                raise ValueError(f"DE_time length expected 512, got {de.shape} at {mp}")
            if use_fft and fft.shape[0] != 256:
                raise ValueError(f"FFT_data length expected 256, got {fft.shape} at {mp}")

            basenames.append(base)
            labels.append(class_to_id[cls])
            mat_paths.append(mp)
            jpg_paths.append(jp)

            if use_time:
                de_list.append(de)
            if use_fft:
                fft_list.append(fft)

        self.basenames = basenames
        self.labels = np.asarray(labels, dtype=np.int64)
        self.mat_paths = mat_paths
        self.jpg_paths = jpg_paths

        self.de = np.stack(de_list, axis=0) if use_time else None     # (N,512)
        self.fft = np.stack(fft_list, axis=0) if use_fft else None    # (N,256)

        # label -> indices
        self.by_label: Dict[int, np.ndarray] = {}
        for y in np.unique(self.labels):
            self.by_label[int(y)] = np.where(self.labels == y)[0]

    def __len__(self) -> int:
        return len(self.basenames)


# -----------------------------
# Episode Samplers
# -----------------------------
class PCDGTrainSampler:
    """
    训练：每个 episode 从 train_domains 中随机选一个 domain，再采样 K-way(N-shot,Q-query)。
    输出：
      support: dict(modality->tensor)
      query: dict(modality->tensor)
      class_ids: LongTensor(K,) (全局类 id)
    """
    def __init__(self, cfg: dict):
        data_cfg = cfg["data"]
        model_cfg = cfg["model"]
        ep_cfg = cfg["episode"]

        self.root_dir = data_cfg["root"]
        self.train_domains: List[str] = list(data_cfg["train_domains"])
        self.k = int(ep_cfg["k"])
        self.n = int(ep_cfg["n"])
        self.q = int(ep_cfg["q"])

        mods = model_cfg["modalities"]
        self.use_time = bool(mods.get("use_time", False))
        self.use_fft = bool(mods.get("use_fft", True))
        self.use_tf = bool(mods.get("use_tf", False))

        # 类表：建议使用预处理生成的 selected_classes.json 同步；这里直接从 cfg 提供
        class_names: List[str] = list(data_cfg["class_names"])
        self.class_to_id = {c: i for i, c in enumerate(class_names)}
        self.id_to_class = {i: c for c, i in self.class_to_id.items()}

        # build domain indices
        self.domains: Dict[str, DomainIndex] = {}
        for d in self.train_domains:
            self.domains[d] = DomainIndex(
                root_dir=self.root_dir,
                domain=d,
                class_to_id=self.class_to_id,
                use_time=self.use_time,
                use_fft=self.use_fft,
                use_tf=self.use_tf,
            )

        # norm
        norm_cfg = data_cfg.get("normalize", {})
        self.norm_enable = bool(norm_cfg.get("enable", True))
        self.norm_mode = str(norm_cfg.get("mode", "concat_zscore"))

        self.scaler: Optional[ZScoreScaler] = None
        if self.norm_enable and self.norm_mode == "concat_zscore":
            self.scaler = fit_concat_zscore(list(self.domains.values()), self.use_time, self.use_fft)

        # tf transform
        tf_cfg = data_cfg.get("tf", {})
        size = int(tf_cfg.get("size", 224))
        if self.use_tf:
            if transforms is None:
                raise RuntimeError("torchvision not available, but use_tf=True.")
            self.tf_transform = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ])
        else:
            self.tf_transform = None

        self.rng = random.Random(int(cfg.get("seed", 42)))

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        domain = self.rng.choice(self.train_domains)
        return self._sample_episode(self.domains[domain])

    def _sample_episode(self, dset: DomainIndex) -> dict:
        # 选出在该 domain 里样本足够的类
        eligible = []
        for y, idxs in dset.by_label.items():
            if len(idxs) >= (self.n + self.q):
                eligible.append(y)
        if len(eligible) < self.k:
            raise RuntimeError(f"Domain {dset.domain} has only {len(eligible)} eligible classes, need k={self.k}")

        class_ids = self.rng.sample(eligible, self.k)

        # 收集 support/query
        sup: Dict[str, List[torch.Tensor]] = {}
        qry: Dict[str, List[torch.Tensor]] = {}
        if self.use_time:
            sup["time"], qry["time"] = [], []
        if self.use_fft:
            sup["fft"], qry["fft"] = [], []
        if self.use_tf:
            sup["tf"], qry["tf"] = [], []

        for y in class_ids:
            idxs = dset.by_label[int(y)]
            pick = self.rng.sample(list(idxs), self.n + self.q)
            s_idx = pick[: self.n]
            q_idx = pick[self.n :]

            # vectors
            if self.use_time or self.use_fft:
                if self.use_time and self.use_fft:
                    Xs = np.concatenate([dset.de[s_idx], dset.fft[s_idx]], axis=1)
                    Xq = np.concatenate([dset.de[q_idx], dset.fft[q_idx]], axis=1)
                    if self.scaler is not None:
                        Xs = self.scaler.transform(Xs)
                        Xq = self.scaler.transform(Xq)
                    de_s, fft_s = Xs[:, :512], Xs[:, 512:]
                    de_q, fft_q = Xq[:, :512], Xq[:, 512:]
                elif self.use_time:
                    de_s = dset.de[s_idx]
                    de_q = dset.de[q_idx]
                    if self.scaler is not None:
                        de_s = self.scaler.transform(de_s)
                        de_q = self.scaler.transform(de_q)
                    fft_s = fft_q = None
                else:
                    fft_s = dset.fft[s_idx]
                    fft_q = dset.fft[q_idx]
                    if self.scaler is not None:
                        fft_s = self.scaler.transform(fft_s)
                        fft_q = self.scaler.transform(fft_q)
                    de_s = de_q = None

                if self.use_time:
                    sup["time"].append(torch.from_numpy(de_s.astype(np.float32)))
                    qry["time"].append(torch.from_numpy(de_q.astype(np.float32)))
                if self.use_fft:
                    sup["fft"].append(torch.from_numpy(fft_s.astype(np.float32)))
                    qry["fft"].append(torch.from_numpy(fft_q.astype(np.float32)))

            # tf images (lazy)
            if self.use_tf:
                s_imgs = []
                q_imgs = []
                for i in s_idx:
                    img = Image.open(dset.jpg_paths[i]).convert("RGB")
                    s_imgs.append(self.tf_transform(img))
                for i in q_idx:
                    img = Image.open(dset.jpg_paths[i]).convert("RGB")
                    q_imgs.append(self.tf_transform(img))
                sup["tf"].append(torch.stack(s_imgs, dim=0))  # (n,3,H,W)
                qry["tf"].append(torch.stack(q_imgs, dim=0))  # (q,3,H,W)

        # stack to (K,N,...) / (K,Q,...)
        support = {}
        query = {}

        for k in sup.keys():
            support[k] = torch.stack(sup[k], dim=0)
            query[k] = torch.stack(qry[k], dim=0)

        return {
            "support": support,
            "query": query,
            "class_ids": torch.tensor(class_ids, dtype=torch.long),
        }


class PCDGTestSampler:
    """
    测试：固定一个 test_domain，反复采样 episodes 做均值准确率。
    """
    def __init__(self, cfg: dict, test_domain: str):
        data_cfg = cfg["data"]
        model_cfg = cfg["model"]
        ep_cfg = cfg["episode"]

        self.root_dir = data_cfg["root"]
        self.test_domain = test_domain
        self.k = int(ep_cfg["k"])
        self.n = int(ep_cfg["n"])
        self.q = int(ep_cfg["q"])

        mods = model_cfg["modalities"]
        self.use_time = bool(mods.get("use_time", False))
        self.use_fft = bool(mods.get("use_fft", True))
        self.use_tf = bool(mods.get("use_tf", False))

        class_names: List[str] = list(data_cfg["class_names"])
        self.class_to_id = {c: i for i, c in enumerate(class_names)}

        self.domain = DomainIndex(
            root_dir=self.root_dir,
            domain=self.test_domain,
            class_to_id=self.class_to_id,
            use_time=self.use_time,
            use_fft=self.use_fft,
            use_tf=self.use_tf,
        )

        # 复用训练同样的 zscore（用 cfg 里传入的 scaler；这里简单：重新用 train_domains 拟合）
        # 建议：你想严格一致就把 scaler 保存/加载；先按能跑优先。
        norm_cfg = data_cfg.get("normalize", {})
        self.norm_enable = bool(norm_cfg.get("enable", True))
        self.norm_mode = str(norm_cfg.get("mode", "concat_zscore"))
        self.scaler: Optional[ZScoreScaler] = None
        if self.norm_enable and self.norm_mode == "concat_zscore":
            train_domains = list(data_cfg["train_domains"])
            train_indexes = [
                DomainIndex(self.root_dir, d, self.class_to_id, self.use_time, self.use_fft, self.use_tf)
                for d in train_domains
            ]
            self.scaler = fit_concat_zscore(train_indexes, self.use_time, self.use_fft)

        tf_cfg = data_cfg.get("tf", {})
        size = int(tf_cfg.get("size", 224))
        if self.use_tf:
            if transforms is None:
                raise RuntimeError("torchvision not available, but use_tf=True.")
            self.tf_transform = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ])
        else:
            self.tf_transform = None

        self.rng = random.Random(int(cfg.get("seed", 42)) + 999)

    def sample(self) -> dict:
        # 同 TrainSampler 逻辑
        eligible = []
        for y, idxs in self.domain.by_label.items():
            if len(idxs) >= (self.n + self.q):
                eligible.append(y)
        if len(eligible) < self.k:
            raise RuntimeError(f"Domain {self.test_domain} has only {len(eligible)} eligible classes, need k={self.k}")

        class_ids = self.rng.sample(eligible, self.k)

        sup: Dict[str, List[torch.Tensor]] = {}
        qry: Dict[str, List[torch.Tensor]] = {}
        if self.use_time:
            sup["time"], qry["time"] = [], []
        if self.use_fft:
            sup["fft"], qry["fft"] = [], []
        if self.use_tf:
            sup["tf"], qry["tf"] = [], []

        dset = self.domain
        for y in class_ids:
            idxs = dset.by_label[int(y)]
            pick = self.rng.sample(list(idxs), self.n + self.q)
            s_idx = pick[: self.n]
            q_idx = pick[self.n :]

            if self.use_time or self.use_fft:
                if self.use_time and self.use_fft:
                    Xs = np.concatenate([dset.de[s_idx], dset.fft[s_idx]], axis=1)
                    Xq = np.concatenate([dset.de[q_idx], dset.fft[q_idx]], axis=1)
                    if self.scaler is not None:
                        Xs = self.scaler.transform(Xs)
                        Xq = self.scaler.transform(Xq)
                    de_s, fft_s = Xs[:, :512], Xs[:, 512:]
                    de_q, fft_q = Xq[:, :512], Xq[:, 512:]
                elif self.use_time:
                    de_s = dset.de[s_idx]
                    de_q = dset.de[q_idx]
                    if self.scaler is not None:
                        de_s = self.scaler.transform(de_s)
                        de_q = self.scaler.transform(de_q)
                    fft_s = fft_q = None
                else:
                    fft_s = dset.fft[s_idx]
                    fft_q = dset.fft[q_idx]
                    if self.scaler is not None:
                        fft_s = self.scaler.transform(fft_s)
                        fft_q = self.scaler.transform(fft_q)
                    de_s = de_q = None

                if self.use_time:
                    sup["time"].append(torch.from_numpy(de_s.astype(np.float32)))
                    qry["time"].append(torch.from_numpy(de_q.astype(np.float32)))
                if self.use_fft:
                    sup["fft"].append(torch.from_numpy(fft_s.astype(np.float32)))
                    qry["fft"].append(torch.from_numpy(fft_q.astype(np.float32)))

            if self.use_tf:
                s_imgs = []
                q_imgs = []
                for i in s_idx:
                    img = Image.open(dset.jpg_paths[i]).convert("RGB")
                    s_imgs.append(self.tf_transform(img))
                for i in q_idx:
                    img = Image.open(dset.jpg_paths[i]).convert("RGB")
                    q_imgs.append(self.tf_transform(img))
                sup["tf"].append(torch.stack(s_imgs, dim=0))
                qry["tf"].append(torch.stack(q_imgs, dim=0))

        support = {k: torch.stack(v, dim=0) for k, v in sup.items()}
        query = {k: torch.stack(v, dim=0) for k, v in qry.items()}
        return {
            "support": support,
            "query": query,
            "class_ids": torch.tensor(class_ids, dtype=torch.long),
        }
