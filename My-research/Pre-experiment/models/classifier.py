from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np


class PrototypicalClassifier:
    """原型网络分类器"""

    def __init__(self, n_neighbors=1):
        self.classifier = KNeighborsClassifier(n_neighbors=n_neighbors)

    def fit(self, support_features, support_labels):
        """基于支持集构建原型"""
        self.classifier.fit(support_features, support_labels)

    def predict(self, query_features):
        return self.classifier.predict(query_features)

    def predict_proba(self, query_features):
        return self.classifier.predict_proba(query_features)


class SVMClassifier:
    """SVM分类器"""

    def __init__(self):
        self.classifier = SVC(kernel='rbf', C=1.0, probability=True)

    def fit(self, support_features, support_labels):
        self.classifier.fit(support_features, support_labels)

    def predict(self, query_features):
        return self.classifier.predict(query_features)