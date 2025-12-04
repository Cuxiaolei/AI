import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, \
    classification_report
from typing import Dict, List


def compute_metrics(y_true: List, y_pred: List) -> Dict:
    """计算评估指标"""

    # 基本指标
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 每个类的指标
    labels = sorted(set(y_true) | set(y_pred))
    report = classification_report(y_true, y_pred,
                                   labels=labels,
                                   output_dict=True)

    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm,
        'classification_report': report,
        'per_class_f1': [report[str(i)]['f1-score'] for i in labels]
    }


def print_metrics(metrics: Dict, label_names: List[str] = None):
    """打印评估结果"""

    print("\n" + "=" * 60)
    print("评估结果:")
    print("=" * 60)

    print(f"准确率 (Accuracy): {metrics['accuracy']:.4f}")
    print(f"F1分数 (Macro): {metrics['f1_macro']:.4f}")
    print(f"F1分数 (Weighted): {metrics['f1_weighted']:.4f}")

    print("\n每个类别的F1分数:")
    if label_names is None:
        label_names = [f"Class {i}" for i in range(len(metrics['per_class_f1']))]

    for name, f1 in zip(label_names, metrics['per_class_f1']):
        print(f"  {name}: {f1:.4f}")

    print("\n混淆矩阵:")
    print(metrics['confusion_matrix'])