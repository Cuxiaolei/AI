import os
import scipy.io as sio
import numpy as np


class OttawaDataset:
    """渥太华轴承数据集加载器（最终版）"""

    def __init__(self, data_path, config):
        self.data_path = data_path
        self.config = config
        self.window_size = config['DATA']['window_size']
        self.overlap = config['DATA']['overlap']

        # 域映射表
        self.domain_map = self._create_domain_map()

        # 按健康状态分类的域索引
        self.health_domains = {
            0: [0, 1, 2, 3],  # 健康类所在的域
            1: [4, 5, 6, 7],  # 内圈类所在的域
            2: [8, 9, 10, 11]  # 外圈类所在的域
        }

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

        for file_name in domain_info['files']:
            file_path = os.path.join(self.data_path, file_name)
            if not os.path.exists(file_path):
                continue

            try:
                data = sio.loadmat(file_path)
                vibration = data['Channel_1'].flatten()
                speed = data['Channel_2'].flatten()

                # 分段生成样本
                samples = self._segment_signal(vibration, speed)

                for sample in samples:
                    all_vibration.append(sample['vibration'])
                    all_speed.append(sample['speed'])
            except:
                pass

        # 标签由域决定（单域单类）
        label = self._health_to_label(domain_info['health'])

        return {
            'vibration': np.vstack(all_vibration),
            'speed': np.vstack(all_speed),
            'labels': np.full(len(all_vibration), label),
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
        生成episode（允许重复采样）
        当可用域不足时，从同一域重复采样不同样本
        """
        # 加载所有源域数据
        source_data = {idx: self.load_domain(idx) for idx in source_domains}

        # 加载目标域数据
        target_data = self.load_domain(target_domain)

        # 划分支持集和查询集
        support_set = {'X': [], 'y': []}
        query_set = {'X': [], 'y': []}

        # 按类采样
        for class_idx in range(3):  # 遍历3类健康状态
            # 获取包含该类样本的所有可用域
            class_domains = self.health_domains[class_idx]
            available_domains = [d for d in class_domains if d in source_domains]

            if len(available_domains) == 0:
                raise ValueError(f"类{class_idx}无可用域")

            # 支持集：从可用域中采样k_shot个域（允许重复）
            support_domains = np.random.choice(
                available_domains,
                size=k_shot,
                replace=True  # ✅ 关键：允许重复
            )

            # 查询集：从可用域中采样n_query个域（允许重复）
            query_domains = np.random.choice(
                available_domains,
                size=n_query,
                replace=True  # ✅ 关键：允许重复
            )

            # 从各域抽取样本（支持集）
            for domain_idx in support_domains:
                domain_data = source_data[domain_idx]
                sample_idx = np.random.randint(len(domain_data['vibration']))
                support_set['X'].append(domain_data['vibration'][sample_idx])
                support_set['y'].append(class_idx)

            # 从各域抽取样本（查询集）
            for domain_idx in query_domains:
                domain_data = source_data[domain_idx]
                sample_idx = np.random.randint(len(domain_data['vibration']))
                query_set['X'].append(domain_data['vibration'][sample_idx])
                query_set['y'].append(class_idx)

        # 转换为numpy数组
        support_set['X'] = np.vstack(support_set['X'])
        support_set['y'] = np.array(support_set['y'])
        query_set['X'] = np.vstack(query_set['X'])
        query_set['y'] = np.array(query_set['y'])

        return support_set, query_set, target_data