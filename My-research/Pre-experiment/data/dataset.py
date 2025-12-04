import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
from typing import Dict, List, Tuple
import pandas as pd
from data.preprocessor import SignalPreprocessor


class OttawaBearingDataset(Dataset):
    """渥太华轴承数据集加载器"""

    def __init__(self,
                 data_dir: str,
                 domains: List[str],
                 window_size: int = 2048,
                 overlap: float = 0.5,
                 channels: List[str] = ['Channel_1'],
                 k_shot: int = None,
                 n_query: int = None,
                 mode: str = 'train',
                 preprocessor: SignalPreprocessor = None):
        """
        Args:
            data_dir: 数据集根目录
            domains: 域列表，如 ['H-A', 'I-B']
            window_size: 滑动窗口大小
            overlap: 重叠率
            channels: 使用的通道列表
            k_shot: 小样本k-shot数
            n_query: 查询集数量
            mode: 'train' 或 'test'
            preprocessor: 预处理器实例
        """
        self.data_dir = data_dir
        self.domains = domains
        self.window_size = window_size
        self.overlap = overlap
        self.channels = channels
        self.k_shot = k_shot
        self.n_query = n_query
        self.mode = mode
        self.preprocessor = preprocessor or SignalPreprocessor()

        # 标签映射
        self.label_map = {'H': 0, 'I': 1, 'O': 2}

        # 加载数据
        self.data, self.labels, self.domain_labels = self._load_data()

        # 小样本采样（如果指定）
        if k_shot is not None and mode == 'train':
            self.data, self.labels, self.domain_labels = self._sample_k_shot()

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """加载指定域的所有数据"""
        all_data = []
        all_labels = []
        all_domains = []

        for domain in self.domains:
            health = domain.split('-')[0]  # H, I, O
            condition = domain.split('-')[1]  # A, B, C, D

            # 查找该域的所有文件
            pattern = f"{health}-{condition}-"
            domain_files = [f for f in os.listdir(self.data_dir) if pattern in f]

            for file_name in domain_files:
                file_path = os.path.join(self.data_dir, file_name)
                try:
                    # 加载mat文件
                    mat_data = sio.loadmat(file_path)

                    for channel in self.channels:
                        if channel in mat_data:
                            signal = mat_data[channel].flatten()

                            # 信号预处理
                            signal = self.preprocessor.preprocess(signal,
                                                                  sampling_rate=200000)

                            # 滑动窗口切片
                            samples = self._slice_signal(signal)

                            label = self.label_map[health]

                            all_data.append(samples)
                            all_labels.extend([label] * len(samples))
                            all_domains.extend([domain] * len(samples))

                except Exception as e:
                    print(f"加载文件 {file_name} 失败: {e}")

        if not all_data:
            raise ValueError(f"未找到有效数据，请检查数据路径: {self.data_dir}")

        # 合并所有数据
        all_data = np.vstack(all_data)
        all_labels = np.array(all_labels)
        all_domains = np.array(all_domains)

        return all_data, all_labels, all_domains

    def _slice_signal(self, signal: np.ndarray) -> np.ndarray:
        """将信号切分为重叠窗口"""
        step = int(self.window_size * (1 - self.overlap))
        num_samples = (len(signal) - self.window_size) // step + 1

        samples = np.lib.stride_tricks.sliding_window_view(
            signal, self.window_size
        )[::step]

        return samples[:num_samples]  # 确保长度一致

    def _sample_k_shot(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """小样本采样：每类每域采样k个样本"""

        sampled_data = []
        sampled_labels = []
        sampled_domains = []

        unique_domains = np.unique(self.domain_labels)
        unique_labels = np.unique(self.labels)

        for domain in unique_domains:
            domain_mask = self.domain_labels == domain
            domain_data = self.data[domain_mask]
            domain_labels = self.labels[domain_mask]
            domain_domains = self.domain_labels[domain_mask]

            for label in unique_labels:
                label_mask = domain_labels == label
                label_data = domain_data[label_mask]

                # 如果样本不足k_shot，则重复采样
                if len(label_data) < self.k_shot:
                    indices = np.random.choice(len(label_data), self.k_shot,
                                               replace=True)
                else:
                    indices = np.random.choice(len(label_data), self.k_shot,
                                               replace=False)

                sampled_data.append(label_data[indices])
                sampled_labels.extend([label] * self.k_shot)
                sampled_domains.extend([domain] * self.k_shot)

        return np.vstack(sampled_data), np.array(sampled_labels), \
            np.array(sampled_domains)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        label = self.labels[idx]
        domain = self.domain_labels[idx]

        # 归一化
        sample = (sample - sample.mean()) / (sample.std() + 1e-8)

        # 转换为tensor并增加通道维度
        sample = torch.FloatTensor(sample).unsqueeze(0)  # [1, window_size]

        return {
            'data': sample,
            'label': torch.LongTensor([label])[0],
            'domain': domain
        }


class EpisodeDataLoader:
    """用于小样本学习的Episode数据加载器"""

    def __init__(self, dataset: OttawaBearingDataset,
                 n_way: int, k_shot: int, n_query: int):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query

        self.labels = dataset.labels
        self.domains = dataset.domain_labels

    def generate_episode(self) -> Tuple[torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor]:
        """生成一个episode：支持集+查询集"""

        # 随机选择n_way个类别
        unique_labels = np.unique(self.labels)
        selected_labels = np.random.choice(unique_labels, self.n_way,
                                           replace=False)

        # 每个域中每类采样
        support_data = []
        support_labels = []
        query_data = []
        query_labels = []

        for domain in np.unique(self.domains):
            domain_mask = self.domains == domain
            domain_data = self.dataset.data[domain_mask]
            domain_labels = self.labels[domain_mask]

            for label_idx, label in enumerate(selected_labels):
                # 获取该类在该域中的数据
                label_mask = domain_labels == label
                class_data = domain_data[label_mask]

                if len(class_data) < self.k_shot + self.n_query:
                    continue

                # 随机打乱并分割
                indices = np.random.permutation(len(class_data))
                class_data = class_data[indices]

                # 支持集
                support_data.append(class_data[:self.k_shot])
                support_labels.extend([label_idx] * self.k_shot)

                # 查询集
                query_data.append(class_data[self.k_shot:self.k_shot + self.n_query])
                query_labels.extend([label_idx] * self.n_query)

        if not support_data:
            raise ValueError("无法生成episode，请调整k_shot和n_query参数")

        # 转换为tensor
        support_data = torch.FloatTensor(np.vstack(support_data))
        support_labels = torch.LongTensor(support_labels)
        query_data = torch.FloatTensor(np.vstack(query_data))
        query_labels = torch.LongTensor(query_labels)

        # 增加通道维度
        support_data = support_data.unsqueeze(1)  # [N, 1, window_size]
        query_data = query_data.unsqueeze(1)  # [N, 1, window_size]

        return support_data, support_labels, query_data, query_labels


# 测试代码
if __name__ == "__main__":
    data_dir = "./data/Ottawa_Bearing_Dataset"

    # 创建数据集
    domains = ['H-A', 'I-A', 'O-A']
    dataset = OttawaBearingDataset(data_dir, domains, k_shot=5)

    print(f"数据集大小: {len(dataset)}")
    print(f"数据形状: {dataset.data.shape}")
    print(f"标签分布: {np.unique(dataset.labels, return_counts=True)}")

    # 创建episode loader
    episode_loader = EpisodeDataLoader(dataset, n_way=3, k_shot=5, n_query=15)
    support_data, support_labels, query_data, query_labels = episode_loader.generate_episode()

    print(f"支持集形状: {support_data.shape}")
    print(f"查询集形状: {query_data.shape}")