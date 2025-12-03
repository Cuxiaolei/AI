from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
import joblib
import os
import numpy as np


class BaseFSClassifier:
    """
    小样本分类器基类
    提供统一的fit/predict接口和模型管理
    """

    def __init__(self, config):
        self.config = config
        self.model = None
        self.is_fitted = False

        # 健康状态映射
        self.health_map = {0: '健康', 1: '内圈缺陷', 2: '外圈缺陷'}

    def fit(self, X, y):
        """训练模型"""
        raise NotImplementedError

    def predict(self, X):
        """预测类别"""
        if not self.is_fitted:
            raise RuntimeError("模型未训练，请先调用fit()")
        return self.model.predict(X)

    def predict_proba(self, X):
        """预测概率"""
        if not self.is_fitted:
            raise RuntimeError("模型未训练，请先调用fit()")

        # 如果模型不支持predict_proba，使用决策函数转换
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        elif hasattr(self.model, 'decision_function'):
            # 对SVM等模型进行概率校准
            decision = self.model.decision_function(X)
            # 简单归一化到[0,1]
            if decision.ndim == 1:
                decision = np.vstack([-decision, decision]).T
            proba = np.exp(decision) / np.sum(np.exp(decision), axis=1, keepdims=True)
            return proba
        else:
            # 回退：使用距离计算伪概率
            if hasattr(self.model, '_fit_X'):
                # KNN的情况
                neigh_dist, neigh_ind = self.model.kneighbors(X)
                proba = np.zeros((len(X), len(self.health_map)))
                for i, (dists, idxs) in enumerate(zip(neigh_dist, neigh_ind)):
                    for j, idx in enumerate(idxs):
                        label = self.model._y[idx]
                        weight = 1 / (dists[j] + 1e-8)
                        proba[i, label] += weight
                proba /= proba.sum(axis=1, keepdims=True)
                return proba
            else:
                # 均匀分布
                return np.ones((len(X), len(self.health_map))) / len(self.health_map)

    def evaluate(self, X_test, y_test):
        """评估模型性能"""
        from utils.metrics import FSMetrics
        metrics = FSMetrics()

        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        return metrics.compute_classification_metrics(y_test, y_pred, y_proba)

    def save(self, path):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'config': self.config,
            'is_fitted': self.is_fitted
        }, path)
        print(f"✅ 模型已保存: {path}")

    def load(self, path):
        """加载模型"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在: {path}")

        saved = joblib.load(path)
        self.model = saved['model']
        self.config = saved['config']
        self.is_fitted = saved['is_fitted']
        print(f"✅ 模型已加载: {path}")
        return self

    def get_params(self):
        """获取模型参数"""
        return self.model.get_params() if self.model else {}


class KNNClassifier(BaseFSClassifier):
    """KNN分类器（小样本默认）"""

    def __init__(self, config):
        super().__init__(config)
        n_neighbors = config['MODEL'].get('n_neighbors', 1)

        # 使用距离权重，在小样本下更鲁棒
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights='distance',  # 距离加权投票
            metric='minkowski',  # 欧氏距离的推广
            p=2  # p=2为欧氏距离
        )

    def fit(self, X, y):
        """训练KNN（构建KD树）"""
        self.model.fit(X, y)
        self.is_fitted = True
        print(f"  KNN模型训练完成: n_neighbors={self.model.n_neighbors}, "
              f"训练样本数={len(X)}")


class SVMClassifier(BaseFSClassifier):
    """SVM分类器（支持概率输出）"""

    def __init__(self, config):
        super().__init__(config)

        # RBF核SVM，参数在小样本下调小
        self.base_model = SVC(
            kernel='rbf',
            C=config['MODEL'].get('svm_C', 1.0),
            gamma='scale',
            probability=True,  # 启用概率输出
            class_weight='balanced'  # 处理类别不平衡
        )

        # 使用概率校准提高可靠性
        self.model = CalibratedClassifierCV(
            self.base_model,
            method='sigmoid',
            cv=3  # 3折交叉验证校准
        )

    def fit(self, X, y):
        """训练SVM"""
        # 小样本下需要最小化CV折数
        if len(X) < 10:
            # 样本太少，禁用CV
            self.model = self.base_model
            self.model.probability = True

        self.model.fit(X, y)
        self.is_fitted = True
        print(
            f"  SVM模型训练完成: 支持向量数={len(self.model.support_vectors_ if hasattr(self.model, 'support_vectors_') else [])}")


class RandomForestClassifierFS(BaseFSClassifier):
    """随机森林分类器（适合非线性特征）"""

    def __init__(self, config):
        super().__init__(config)

        self.model = RandomForestClassifier(
            n_estimators=config['MODEL'].get('rf_n_estimators', 50),
            max_depth=config['MODEL'].get('rf_max_depth', 5),
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1  # 使用所有CPU核心
        )

    def fit(self, X, y):
        """训练随机森林"""
        self.model.fit(X, y)
        self.is_fitted = True

        # 计算特征重要性
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            top3_idx = np.argsort(importance)[-3:][::-1]
            print(f"  随机森林训练完成: 树数量={len(self.model.estimators_)}, "
                  f"Top3特征索引={top3_idx}")


class PrototypicalNetworkClassifier(BaseFSClassifier):
    """
    原型网络分类器（小样本专用）
    基于度量学习，计算到类原型的距离
    """

    def __init__(self, config):
        super().__init__(config)
        self.prototypes = None  # 类原型

    def fit(self, X, y):
        """
        构建类原型（每个类的特征中心）
        """
        self.classes_ = np.unique(y)
        self.prototypes = {}

        for class_id in self.classes_:
            class_samples = X[y == class_id]
            # 原型 = 类内样本均值
            prototype = np.mean(class_samples, axis=0)
            self.prototypes[class_id] = prototype

        self.is_fitted = True
        print(f"  原型网络构建完成: {len(self.prototypes)}个类原型")

    def predict(self, X):
        """最近原型分类"""
        if not self.is_fitted:
            raise RuntimeError("原型未构建，请先调用fit()")

        # 计算到每个原型的距离
        predictions = []
        for x in X:
            distances = {}
            for class_id, prototype in self.prototypes.items():
                dist = np.linalg.norm(x - prototype)
                distances[class_id] = dist

            # 预测为最近原型对应的类别
            pred_class = min(distances, key=distances.get)
            predictions.append(pred_class)

        return np.array(predictions)

    def predict_proba(self, X):
        """基于距离的伪概率"""
        predictions = []
        for x in X:
            distances = []
            for class_id, prototype in self.prototypes.items():
                dist = np.linalg.norm(x - prototype)
                distances.append(dist)

            # 距离转概率（距离越近概率越高）
            distances = np.array(distances)
            inv_dist = 1 / (distances + 1e-8)  # 避免除零
            prob = inv_dist / inv_dist.sum()
            predictions.append(prob)

        return np.array(predictions)

    def save(self, path):
        """保存原型网络"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'prototypes': self.prototypes,
            'classes_': self.classes_,
            'is_fitted': self.is_fitted
        }, path)
        print(f"✅ 原型网络已保存: {path}")

    def load(self, path):
        """加载原型网络"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在: {path}")

        saved = joblib.load(path)
        self.prototypes = saved['prototypes']
        self.classes_ = saved['classes_']
        self.is_fitted = saved['is_fitted']
        print(f"✅ 原型网络已加载: {path}")
        return self


class ClassifierFactory:
    """
    分类器工厂类
    统一创建和管理不同分类器
    """

    @staticmethod
    def create_classifier(config):
        """根据配置创建分类器"""
        classifier_type = config['MODEL']['classifier']

        if classifier_type == 'KNN':
            return KNNClassifier(config)
        elif classifier_type == 'SVM':
            return SVMClassifier(config)
        elif classifier_type == 'RF':
            return RandomForestClassifierFS(config)
        elif classifier_type == 'ProtoNet':
            return PrototypicalNetworkClassifier(config)
        else:
            raise ValueError(f"未知的分类器类型: {classifier_type}")

    @staticmethod
    def list_supported_classifiers():
        """返回支持的分类器列表"""
        return ['KNN', 'SVM', 'RF', 'ProtoNet']

    @staticmethod
    def get_classifier_hyperparameters(classifier_type):
        """获取分类器可调的超参数"""
        hp_dict = {
            'KNN': {'n_neighbors': [1, 3, 5, 7]},
            'SVM': {'C': [0.1, 1.0, 10.0], 'gamma': ['scale', 'auto']},
            'RF': {'n_estimators': [30, 50, 100], 'max_depth': [5, 10, None]},
            'ProtoNet': {}  # 无超参数
        }
        return hp_dict.get(classifier_type, {})