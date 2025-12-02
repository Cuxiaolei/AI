import numpy as np
import scipy.io as sio
import yaml
import os
import joblib
from utils.data_loader import OttawaDataset
from models.feature_extractor import FeatureExtractor


class FSDGPredictor:
    """预测器"""

    def __init__(self, config_path, model_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # 加载组件
        self.dataset = OttawaDataset(self.config['DATA']['path'], self.config)
        self.feature_extractor = FeatureExtractor(self.config)

        # 加载训练好的模型
        self.classifier = joblib.load(model_path)

    def predict_file(self, file_path):
        """预测单个文件"""
        # 加载数据
        data = sio.loadmat(file_path)
        vibration = data['Channel_1'].flatten()

        # 分段
        window_size = self.config['DATA']['window_size']
        step = int(window_size * (1 - self.config['DATA']['overlap']))
        n_samples = (len(vibration) - window_size) // step

        # 提取特征
        features = []
        for i in range(min(n_samples, 20)):  # 限制样本数
            start = i * step
            end = start + window_size
            sample = vibration[start:end]
            feat = self.feature_extractor.extract_features([sample])
            features.append(feat[0])

        features = np.array(features)

        # 预测
        predictions = self.classifier.predict(features)
        probs = self.classifier.predict_proba(features)

        # 投票决定最终类别
        final_pred = np.bincount(predictions).argmax()
        confidence = np.mean(probs[:, final_pred])

        health_map = {0: '健康', 1: '内圈缺陷', 2: '外圈缺陷'}

        return {
            'prediction': health_map[final_pred],
            'confidence': confidence,
            'all_votes': {health_map[i]: np.sum(predictions == i)
                          for i in range(3)}
        }


def main():
    config_path = './configs/config.yaml'
    model_path = './results/best_classifier.pkl'

    if not os.path.exists(model_path):
        print("错误: 模型文件不存在，请先运行训练脚本!")
        return

    predictor = FSDGPredictor(config_path, model_path)

    # 测试文件
    test_file = './data/Ottawa Bearing Dataset/H-A-1.mat'
    if os.path.exists(test_file):
        result = predictor.predict_file(test_file)
        print(f"预测结果: {result['prediction']}")
        print(f"置信度: {result['confidence']:.4f}")
        print(f"投票详情: {result['all_votes']}")


if __name__ == '__main__':
    main()