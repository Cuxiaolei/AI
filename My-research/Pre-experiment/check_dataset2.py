from utils.data_loader import OttawaDataset
import yaml
import numpy as np

config = {
    'DATA': {'path': '/root/data/Ottawa_Bearing_Dataset', 'window_size': 2048, 'overlap': 0.5}
}

dataset = OttawaDataset(config['DATA']['path'], config)

# 检查每个域的类别分布
print('域类别分布检查:')
for i in range(12):
    data = dataset.load_domain(i)
    counts = [np.sum(data['labels'] == c) for c in range(3)]
    health = dataset.domain_map[i]['health']
    speed = dataset.domain_map[i]['speed']
    print(f'域{i} ({health}-{speed}): 健康={counts[0]}, 内圈={counts[1]}, 外圈={counts[2]}')