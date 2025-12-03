import os
import scipy.io as sio
import numpy as np
from sklearn.preprocessing import StandardScaler


class OttawaDataset:
    """渥太华轴承数据集加载器"""

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
                print(f"Warning: {file_path} not found")
                continue

            # 加载MAT文件
            data = sio.loadmat(file_path)
            vibration = data['Channel_1'].flatten()  # 振动信号
            speed = data['Channel_2'].flatten()  # 转速信号

            # 分段生成样本
            samples = self._segment_signal(vibration, speed)

            for sample in samples:
                all_vibration.append(sample['vibration'])
                all_speed.append(sample['speed'])
                labels.append(self._health_to_label(domain_info['health']))

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
        """生成小样本学习episode"""
        # 加载源域数据
        source_data = {idx: self.load_domain(idx) for idx in source_domains}

        # 加载目标域数据
        target_data = self.load_domain(target_domain)

        # 划分支持集和查询集
        support_set = {'X': [], 'y': []}
        query_set = {'X': [], 'y': []}

        for domain_idx, data in source_data.items():
            for class_idx in range(3):  # 3类健康状态
                class_samples = data['vibration'][data['labels'] == class_idx]
                if len(class_samples) < k_shot + n_query:
                    # 样本不足时复制填充
                    n_needed = (k_shot + n_query) - len(class_samples)
                    class_samples = np.concatenate([
                        class_samples,
                        class_samples[:n_needed]
                    ])

                # 随机采样
                indices = np.random.permutation(len(class_samples))
                support_idx = indices[:k_shot]
                query_idx = indices[k_shot:k_shot + n_query]

                support_set['X'].append(class_samples[support_idx])
                support_set['y'].extend([class_idx] * k_shot)
                query_set['X'].append(class_samples[query_idx])
                query_set['y'].extend([class_idx] * n_query)

        support_set['X'] = np.vstack(support_set['X'])
        support_set['y'] = np.array(support_set['y'])
        query_set['X'] = np.vstack(query_set['X'])
        query_set['y'] = np.array(query_set['y'])

        return support_set, query_set, target_data