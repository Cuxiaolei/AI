import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
from typing import Dict, List, Tuple


class OttawaBearingDataset(Dataset):
    """渥太华轴承数据集加载器——每个域只有单一类别"""

    def __init__(self,
                 data_dir: str,
                 domains: List[str],
                 window_size: int = 2048,
                 overlap: float = 0.5,
                 channels: List[str] = ['Channel_1'],
                 mode: str = 'train',
                 preprocessor=None):
        """
        Args:
            data_dir: 数据集根目录
            domains: 域列表（每个域只包含一种健康状况）
            window_size: 滑动窗口大小
            overlap: 重叠率
            channels: 使用的通道列表
            mode: 'train' 或 'test'
            preprocessor: 预处理器实例
        """
        self.data_dir = data_dir
        self.domains = domains
        self.window_size = window_size
        self.overlap = overlap
        self.channels = channels
        self.mode = mode
        self.preprocessor = preprocessor

        # 标签映射（从文件名第一个字母提取）
        self.label_map = {'H': 0, 'I': 1, 'O': 2}

        # 加载数据
        self.data, self.labels, self.domain_labels = self._load_data()

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
                            if self.preprocessor:
                                signal = self.preprocessor.preprocess(signal,
                                                                      sampling_rate=200000)

                            # 滑动窗口切片
                            samples = self._slice_signal(signal)

                            # 标签（从文件名第一个字母提取）
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

        return samples[:num_samples]

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


class MultiDomainEpisodeLoader:
    """跨域Episode数据加载器——核心修改：从不同域采样不同类别"""

    def __init__(self,
                 datasets: Dict[str, OttawaBearingDataset],
                 n_way: int = 3,
                 k_shot: int = 5,
                 n_query: int = 15):
        """
        Args:
            datasets: 域数据集字典，{domain_name: dataset}
            n_way: 每episode的类别数
            k_shot: 支持集样本数
            n_query: 查询集样本数
        """
        self.datasets = datasets
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query

        # 按健康状况分组域（H开头的域，I开头的域，O开头的域）
        self.health_domains = {
            'H': [dom for dom in datasets.keys() if dom.startswith('H-')],
            'I': [dom for dom in datasets.keys() if dom.startswith('I-')],
            'O': [dom for dom in datasets.keys() if dom.startswith('O-')]
        }

        # 健康状态到标签的映射
        self.health_label_map = {'H': 0, 'I': 1, 'O': 2}

        print(f"Episode loader initialized with:")
        print(f"  - H domains: {len(self.health_domains['H'])}")
        print(f"  - I domains: {len(self.health_domains['I'])}")
        print(f"  - O domains: {len(self.health_domains['O'])}")

    def generate_episode(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        生成跨域episode：从不同域采样不同类别
        例如：从H-A域采样健康样本，从I-B域采样内圈故障样本
        """

        # 选择n_way个类别（从H, I, O中选）
        available_healths = list(self.health_domains.keys())
        actual_n_way = min(self.n_way, len(available_healths))

        if actual_n_way < self.n_way:
            print(f"⚠️  Warning: Only {len(available_healths)} health states available. "
                  f"Using {actual_n_way}-way instead of {self.n_way}-way.")

        selected_healths = np.random.choice(available_healths, actual_n_way, replace=False)

        support_data = []
        support_labels = []
        query_data = []
        query_labels = []

        for label_idx, health in enumerate(selected_healths):
            # 为该类别随机选择一个域（如H-A, H-B, H-C, H-D中的一个）
            domain_list = self.health_domains[health]
            if not domain_list:
                print(f"⚠️  Warning: No domains available for health state {health}")
                continue

            selected_domain = np.random.choice(domain_list)
            dataset = self.datasets[selected_domain]

            # 从该域中采样支持集和查询集（所有样本都是同一健康状态）
            data = dataset.data

            if len(data) < self.k_shot + self.n_query:
                # 如果样本不足，重复采样
                indices = np.random.choice(len(data), self.k_shot + self.n_query, replace=True)
            else:
                # 不重复采样
                indices = np.random.choice(len(data), self.k_shot + self.n_query, replace=False)

            selected_samples = data[indices]

            # 支持集（label_idx是episode内的类别索引，不是真实的健康标签）
            support_data.append(selected_samples[:self.k_shot])
            support_labels.extend([label_idx] * self.k_shot)

            # 查询集
            query_data.append(selected_samples[self.k_shot:])
            query_labels.extend([label_idx] * self.n_query)

        # 转换为tensor
        support_data = torch.FloatTensor(np.vstack(support_data))
        support_labels = torch.LongTensor(support_labels)
        query_data = torch.FloatTensor(np.vstack(query_data))
        query_labels = torch.LongTensor(query_labels)

        # 增加通道维度 [N, 1, window_size]
        support_data = support_data.unsqueeze(1)
        query_data = query_data.unsqueeze(1)

        # 打印episode信息（调试用）
        if hasattr(self, 'debug') and self.debug:
            print(f"Episode generated: {len(support_data)} support, {len(query_data)} query, "
                  f"{actual_n_way}-way, domains: {selected_healths}")

        return support_data, support_labels, query_data, query_labels