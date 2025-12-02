import numpy as np
import pywt
from scipy.stats import kurtosis, skew
from scipy.fft import fft


class FeatureExtractor:
    """多域特征提取器"""

    def __init__(self, config):
        self.config = config
        self.sample_rate = config['DATA']['sample_rate']

    def extract_features(self, signals):
        """提取多域特征"""
        features = []

        for sig in signals:
            feat = []

            # 1. 时域统计特征
            if self.config['FEATURE']['use_statistical']:
                feat.extend(self._time_domain_features(sig))

            # 2. 频域特征
            if self.config['FEATURE']['use_spectral']:
                feat.extend(self._spectral_features(sig))

            # 3. 时频特征（小波包能量）
            if self.config['FEATURE']['use_time_freq']:
                feat.extend(self._wavelet_features(sig))

            features.append(feat)

        return np.array(features)

    def _time_domain_features(self, signal):
        """时域统计特征"""
        return [
            np.mean(signal),
            np.std(signal),
            np.max(np.abs(signal)),
            np.sqrt(np.mean(signal ** 2)),  # RMS
            kurtosis(signal),
            skew(signal),
            np.max(signal) - np.min(signal),  # 峰峰值
            np.max(np.abs(signal)) / np.sqrt(np.mean(signal ** 2)),  # 峰值因子
            np.sqrt(np.mean(signal ** 2)) / np.mean(np.abs(signal)),  # 波形因子
            np.percentile(np.abs(signal), 75)  # 75分位数
        ]

    def _spectral_features(self, signal):
        """频域特征"""
        n_fft = min(len(signal), 4096)
        spectrum = np.abs(fft(signal, n_fft))[:n_fft // 2]
        freq = np.fft.fftfreq(n_fft, 1 / self.sample_rate)[:n_fft // 2]

        return [
            np.sum(spectrum),  # 频谱和
            np.sum((freq * spectrum)) / np.sum(spectrum),  # 频谱质心
            np.sqrt(np.sum((freq ** 2) * spectrum) / np.sum(spectrum)),  # 频谱均方根频率
            np.sum(((freq - np.mean(freq)) ** 2) * spectrum) / np.sum(spectrum),  # 频谱方差
            np.max(spectrum),  # 主频幅值
            np.argmax(spectrum) / n_fft * (self.sample_rate / 2),  # 主频率
            np.sum((spectrum - np.mean(spectrum)) ** 3) / (len(spectrum) * np.std(spectrum) ** 3),  # 频谱偏度
            np.sum((spectrum - np.mean(spectrum)) ** 4) / (len(spectrum) * np.std(spectrum) ** 4)  # 频谱峭度
        ]

    def _wavelet_features(self, signal):
        """小波包能量特征"""
        wp = pywt.WaveletPacket(signal, 'db4', maxlevel=3)
        nodes = wp.get_level(3, order='natural')
        energies = [np.sum(node.data ** 2) for node in nodes]
        total_energy = np.sum(energies)

        # 归一化能量
        return [e / total_energy for e in energies]