import numpy as np
from scipy.linalg import sqrtm
from sklearn.metrics import pairwise_kernels


class CoralAligner:
    """CORAL域对齐"""

    def align(self, source_feat, target_feat):
        """CORAL对齐"""
        # 计算协方差矩阵
        cov_s = np.cov(source_feat, rowvar=False) + np.eye(source_feat.shape[1])
        cov_t = np.cov(target_feat, rowvar=False) + np.eye(target_feat.shape[1])

        # CORAL变换矩阵
        cov_s_sqrt = sqrtm(cov_s)
        cov_t_sqrt = sqrtm(cov_t)

        if np.iscomplexobj(cov_s_sqrt):
            cov_s_sqrt = cov_s_sqrt.real
        if np.iscomplexobj(cov_t_sqrt):
            cov_t_sqrt = cov_t_sqrt.real

        # 对齐源域特征
        A = np.dot(np.dot(cov_s_sqrt, np.linalg.inv(cov_t_sqrt)), cov_s_sqrt)
        source_aligned = np.dot(source_feat, np.linalg.inv(A))

        return source_aligned, target_feat


class MMDAligner:
    """MMD域对齐"""

    def align(self, source_feat, target_feat, gamma=1.0):
        """计算MMD损失"""
        K_ss = pairwise_kernels(source_feat, source_feat, metric='rbf', gamma=gamma)
        K_tt = pairwise_kernels(target_feat, target_feat, metric='rbf', gamma=gamma)
        K_st = pairwise_kernels(source_feat, target_feat, metric='rbf', gamma=gamma)

        mmd_loss = np.mean(K_ss) + np.mean(K_tt) - 2 * np.mean(K_st)
        return mmd_loss