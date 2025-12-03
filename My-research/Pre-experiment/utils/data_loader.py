import os
import scipy.io as sio
import numpy as np
from sklearn.preprocessing import StandardScaler


class OttawaDataset:
    """渥太华轴承数据集加载器（修复版）"""

    def __init__(self, data_path, config):
        self.data_path = data_path
        self.config = config
        self.window_size = config['DATA']['window_size']
        self.overlap = config['DATA']['overlap']

        # 域映射表
        self.domain_map = self._create_domain_map()

    def _create_domain_map(self):
        """创建域到健康状态和转速条件的映射"""
        health = ['H', 'I', 'O']
        speed = ['A', 'B', 'C', 'D']
        domain_map = {}
        idx = 0
        for h in health:
            for s in speed:
                domain_map[idx] = {
                    'health': h,
                    'speed': s,
                    'files': [f"{h}-{s}-{t}.mat" for t in [1, 2, 3]]
                }
                idx += 1
        return domain_map

    def load_domain(self, domain_idx):
        """加载指定域的所有数据"""
        domain_info = self.domain_map[domain_idx]
        all_vibration = []
        all_speed = []
        labels = []

        for trial_idx, file_name in enumerate(domain_info['files']):
            file_path = os.path.join(self.data_path, file_name)
            if not os.path.exists(file_path):
                warnings.warn(f"文件不存在: {file_path}", UserWarning)
                continue

            # 加载MAT文件
            try:
                data = sio.loadmat(file_path)
                vibration = data['Channel_1'].flatten()
                speed = data['Channel_2'].flatten()

                # 分段生成样本
                samples = self._segment_signal(vibration, speed)

                for sample in samples:
                    all_vibration.append(sample['vibration'])
                    all_speed.append(sample['speed'])
                    labels.append(self._health_to_label(domain_info['health']))
            except Exception as e:
                warnings.warn(f"加载失败 {file_name}: {e}", UserWarning)

        return {
            'vibration': np.array(all_vibration),
            'speed': np.array(all_speed),
            'labels': np.array(labels),
            'domain_idx': domain_idx
        }

    def _segment_signal(self, vibration, speed):
        """滑动窗口分段"""
        step = int(self.window_size * (1 - self.overlap))
        n_samples = (len(vibration) - self.window_size) // step

        samples = []
        for i in range(n_samples):
            start = i * step
            end = start + self.window_size
            samples.append({
                'vibration': vibration[start:end],
                'speed': speed[start:end]
            })
        return samples

    def _health_to_label(self, health_code):
        """健康状态转标签"""
        mapping = {'H': 0, 'I': 1, 'O': 2}
        return mapping[health_code]

    def generate_episode(self, source_domains, target_domain, k_shot, n_query):
        """
        生成小样本学习episode（核心修复版）
        确保每类严格采样k_shot支持样本和n_query查询样本
        """
        # 加载源域数据
        source_data = {idx: self.load_domain(idx) for idx in source_domains}

        # 加载目标域数据
        target_data = self.load_domain(target_domain)

        # 划分支持集和查询集
        support_set = {'X': [], 'y': []}
        query_set = {'X': [], 'y': []}

        # 每类所需总样本数
        needed_per_class = k_shot + n_query

        for domain_idx, data in source_data.items():
            for class_idx in range(3):  # 3类健康状态
                # 获取该类所有样本
                class_mask = (data['labels'] == class_idx)
                class_samples = data['vibration'][class_mask]
                n_available = len(class_samples)

                # 严格检查：如果样本数不足，强制填充
                if n_available < k_shot:
                    warnings.warn(
                        f"域{domain_idx} 类{class_idx} 样本严重短缺 "
                        f"({n_available} < {k_shot})，使用全部可用样本并填充"
                    )

                # 如果样本不足，使用全部可用样本并复制填充
                if n_available < needed_per_class:
                    # 计算需要复制的次数
                    repeat_times = (needed_per_class + n_available - 1) // n_available
                    class_samples = np.tile(class_samples, (repeat_times, 1))[:needed_per_class]

                # 随机采样（确保索引不越界）
                indices = np.random.permutation(len(class_samples))
                support_idx = indices[:k_shot]
                query_idx = indices[k_shot:k_shot + n_query]

                # 添加到集合
                support_set['X'].append(class_samples[support_idx])
                support_set['y'].extend([class_idx] * k_shot)
                query_set['X'].append(class_samples[query_idx])
                query_set['y'].extend([class_idx] * n_query)

        # 最终验证：确保数据完整
        if not support_set['X'] or not query_set['X']:
            raise ValueError("样本生成失败：支持集或查询集为空")

        support_set['X'] = np.vstack(support_set['X'])
        support_set['y'] = np.array(support_set['y'])
        query_set['X'] = np.vstack(query_set['X'])
        query_set['y'] = np.array(query_set['y'])

        # 维度验证
        assert support_set['X'].shape[0] == len(support_set['y']), \
            f"支持集X和y维度不匹配: X={support_set['X'].shape}, y={support_set['y'].shape}"
        assert query_set['X'].shape[0] == len(query_set['y']), \
            f"查询集X和y维度不匹配: X={query_set['X'].shape}, y={query_set['y'].shape}"

        return support_set, query_set, target_data