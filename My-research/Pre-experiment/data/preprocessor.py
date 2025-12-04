import numpy as np
import scipy.signal as signal
from typing import Tuple


class SignalPreprocessor:
    """振动信号预处理器"""

    def __init__(self,
                 filtering: bool = True,
                 normalize: bool = True,
                 denoise: bool = False):
        self.filtering = filtering
        self.normalize = normalize
        self.denoise = denoise

    def preprocess(self, sig: np.ndarray, sampling_rate: int) -> np.ndarray:
        """主预处理流程"""

        # 1. 去除直流分量
        sig = sig - np.mean(sig)

        # 2. 带通滤波 (500Hz - 10000Hz)
        if self.filtering:
            sig = self._bandpass_filter(sig, sampling_rate,
                                        lowcut=500, highcut=10000)

        # 3. 去噪
        if self.denoise:
            sig = self._wavelet_denoise(sig)

        # 4. 幅值归一化
        if self.normalize:
            sig = self._normalize_amplitude(sig)

        return sig

    def _bandpass_filter(self, sig: np.ndarray, fs: int,
                         lowcut: int, highcut: int) -> np.ndarray:
        """巴特沃斯带通滤波"""
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq

        # 设计滤波器
        b, a = signal.butter(4, [low, high], btype='band')

        # 双向滤波以消除相位失真
        return signal.filtfilt(b, a, sig)

    def _wavelet_denoise(self, sig: np.ndarray, wavelet: str = 'db4',
                         level: int = 3) -> np.ndarray:
        """小波去噪"""
        import pywt

        coeffs = pywt.wavedec(sig, wavelet, mode='symmetric', level=level)
        sigma = np.median(np.abs(coeffs[-level])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(sig)))

        # 软阈值处理
        coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]

        return pywt.waverec(coeffs, wavelet, mode='symmetric')

    def _normalize_amplitude(self, sig: np.ndarray) -> np.ndarray:
        """幅值归一化到[-1, 1]"""
        return sig / (np.max(np.abs(sig)) + 1e-8)