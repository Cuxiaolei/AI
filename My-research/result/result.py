#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型评估指标提取脚本（最终版）
核心特性：
1. 固定模型列表及顺序：darm,dpjdg,erm,groupdro,irm,mixstylee,mldg,vrex,ours
2. 表头格式：T1_5_Acc/Pre/Rec/F1（指标缩写，任务在前、数据大小在后）
3. 无数据单元格显示'-'
4. 按数据集分类生成带时间戳的Excel文件
5. 自动设置Excel列宽
"""

import os
import json
import pandas as pd
from datetime import datetime
from openpyxl.utils import get_column_letter

# ======================== 核心配置项 ========================
# 1. 固定模型列表（不可修改顺序）
FIXED_MODEL_LIST = ["darm", "groupdro", "erm", "mixstyle", "mldg", "vrex", "irm", "dpjdg", "mcpdg"]

# 2. 可配置项（按需修改）
ROOT_DIR = "outputs"  # outputs根目录（脚本放在outputs下时无需修改）
TARGET_MODELS = [
    "darm",
    "dpjdg",
    "erm",
    "groupdro",
    "irm",
    "mixstyle",
    "mldg",
    "vrex",
    "mcpdg"
]
TARGET_DATASETS = [
    "phm",
    "pu",
]  # 待提取的数据集
TARGET_TASKS = {  # 各数据集对应的任务
    "phm": ["T1", "T2", "T3", "T4"],
    "pu": ["T5", "T6", "T7", "T8"]
}
TARGET_DATA_SIZES = ["1", "5", "10", "20"]  # 数据大小
OUTPUT_PARENT = "/root/autodl-fs/all-result"  # 输出父文件夹

# 3. 指标名称映射（关键：控制表头缩写）
METRIC_MAPPING = {
    'acc': 'Acc',
    'precision_macro': 'Pre',
    'recall_macro': 'Rec',
    'f1_macro': 'F1'
}

# 4. Excel列宽设置（在这里改数值）
COL_WIDTH_SETTINGS = {
    '模型': 12,        # 第一列“模型”宽度
    'default': 12     # 其他指标列默认宽度
}

# ============================================================

def extract_single_json(json_path):
    """提取单个JSON文件的核心指标（×100保留2位小数）"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return {
            'acc': round(data.get('acc', 0) * 100, 2),
            'precision_macro': round(data.get('precision_macro', 0) * 100, 2),
            'recall_macro': round(data.get('recall_macro', 0) * 100, 2),
            'f1_macro': round(data.get('f1_macro', 0) * 100, 2)
        }
    except Exception as e:
        return None


def collect_all_metrics():
    """收集指标数据（仅提取TARGET_MODELS，保留FIXED_MODEL_LIST结构）"""
    metrics_dict = {
        ds: {
            sz: {
                tk: {md: None for md in FIXED_MODEL_LIST}
                for tk in TARGET_TASKS[ds]
            }
            for sz in TARGET_DATA_SIZES
        }
        for ds in TARGET_DATASETS
    }

    for model in TARGET_MODELS:
        model_path = os.path.join(ROOT_DIR, model)
        if not os.path.exists(model_path):
            print(f"⚠️  模型文件夹 {model} 不存在，跳过")
            continue

        for subfolder in os.listdir(model_path):
            subfolder_parts = subfolder.split('_')
            if len(subfolder_parts) != 4:
                continue

            _, dataset, task, size = subfolder_parts

            if dataset not in TARGET_DATASETS or task not in TARGET_TASKS[dataset] or size not in TARGET_DATA_SIZES:
                continue

            json_path = os.path.join(model_path, subfolder, 'final_test_metrics.json')
            if os.path.exists(json_path):
                metrics = extract_single_json(json_path)
                if metrics:
                    metrics_dict[dataset][size][task][model] = metrics

    return metrics_dict


def generate_excel_for_dataset(dataset, metrics_data, output_dir):
    """生成Excel：表头为T1_5_Acc/Pre/Rec/F1格式（任务在前、数据大小在后）"""
    time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    excel_path = os.path.join(output_dir, f"{dataset}_{time_str}.xlsx")

    has_data = any(
        any(
            any(metrics_data[size][task][model] is not None for model in FIXED_MODEL_LIST)
            for task in TARGET_TASKS[dataset]
        )
        for size in TARGET_DATA_SIZES
    )

    if not has_data:
        print(f"⚠️  数据集 {dataset} 无有效指标数据，跳过Excel生成")
        return None

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        start_row = 0

        for size in TARGET_DATA_SIZES:
            table_data = []
            for model in FIXED_MODEL_LIST:
                row = {'模型': model}
                for task in TARGET_TASKS[dataset]:
                    metrics = metrics_data[size][task][model]
                    for metric_key, metric_short in METRIC_MAPPING.items():
                        col_name = f"{task}_{size}_{metric_short}"
                        row[col_name] = metrics[metric_key] if metrics else '-'
                table_data.append(row)

            df = pd.DataFrame(table_data)
            df.to_excel(writer, sheet_name='指标汇总', startrow=start_row, index=False)

            # ============== 在这里设置列宽 ==============
            workbook = writer.book
            worksheet = writer.sheets['指标汇总']

            for idx, col in enumerate(df.columns):
                col_letter = get_column_letter(idx + 1)
                if col == '模型':
                    width = COL_WIDTH_SETTINGS['模型']
                else:
                    width = COL_WIDTH_SETTINGS['default']
                worksheet.column_dimensions[col_letter].width = width
            # ==========================================

            print(f"✅ 数据集 {dataset} - 数据大小 {size} 子表已写入（起始行：{start_row}）")
            start_row += len(df) + 2

    return excel_path


def main():
    print("=" * 60)
    print("🚀 模型评估指标提取工具（最终版）")
    print(f"📅 执行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔧 配置摘要：")
    print(f"   - 固定模型列表：{FIXED_MODEL_LIST}")
    print(f"   - 待提取模型：{TARGET_MODELS}")
    print(f"   - 表头格式：任务_数据大小_缩写（如T1_5_Acc）")
    print(f"   - 输出目录：{os.path.abspath(os.path.join(ROOT_DIR, OUTPUT_PARENT))}")
    print("=" * 60)

    print("\n📥 开始收集评估指标...")
    metrics_data = collect_all_metrics()

    print("\n📁 开始创建输出目录...")
    output_dirs = {}
    for dataset in TARGET_DATASETS:
        dataset_dir = os.path.join(ROOT_DIR, OUTPUT_PARENT, dataset)
        os.makedirs(dataset_dir, exist_ok=True)
        output_dirs[dataset] = dataset_dir

    print("\n📋 开始生成Excel报告...")
    generated_files = []
    for dataset in TARGET_DATASETS:
        print(f"\n--- 处理数据集：{dataset} ---")
        excel_file = generate_excel_for_dataset(dataset, metrics_data[dataset], output_dirs[dataset])
        if excel_file:
            generated_files.append(excel_file)

    print("\n" + "=" * 60)
    print("📊 任务执行完成！")
    if generated_files:
        print(f"✅ 成功生成 {len(generated_files)} 个Excel文件：")
        for file in generated_files:
            print(f"   - {file}")
    else:
        print("⚠️  未生成任何文件（无符合条件的指标数据）")
    print("=" * 60)


if __name__ == "__main__":
    main()