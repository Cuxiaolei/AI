import yaml
import os
from typing import List, Dict

# 基础配置模板（与原模板格式完全一致）
BASE_TEMPLATE = """
BASE: ../pu_config.yaml

# 任务{task_num}：{task_desc} -> {target_idx}
split:
  # 源域工况
  source_conditions:
{source_conditions}
  # 目标域工况
  target_condition: "{target_condition}"

sampling:
  source:
    normal_per_domain: 300
    fault_per_class_per_domain: {fault_num}


output:
  output_dir: "data./PU"
  task_name: "PU_T{task_num}_300-{fault_num}"
"""

# 工况映射表
CONDITIONS = {
    "H1": "N09_M07_F10",
    "H2": "N15_M01_F10",
    "H3": "N15_M07_F04",
    "H4": "N15_M07_F10"
}

# 任务配置：键为任务编号，值为(源工况索引列表, 目标工况索引, 任务描述)
TASK_CONFIGS = {
    9: ([0, 1, 2], 3, "0，1，2 -> 3"),
    10: ([0, 1, 3], 2, "0，1，3 -> 2"),
    11: ([0, 2, 3], 1, "0，2，3 -> 1"),
    12: ([1, 2, 3], 0, "1，2，3 -> 0")
}

# fault_per_class_per_domain 取值列表
FAULT_VALUES = [1, 5, 10, 20]


def format_source_conditions(source_conds: List[str]) -> str:
    """格式化源工况列表为YAML缩进格式"""
    return "\n".join([f"    - \"{cond}\"" for cond in source_conds])


def generate_yaml_file(task_num: int, fault_num: int):
    """
    生成单个YAML文件

    Args:
        task_num: 任务编号（9-12）
        fault_num: fault_per_class_per_domain取值
        output_dir: 输出目录
    """
    # 创建输出目录
    script_dir = os.path.dirname(os.path.abspath(__file__))  # 脚本所在目录的绝对路径
    output_dir = script_dir  # 输出目录 = 脚本所在目录

    # 获取任务配置
    source_idxs, target_idx, task_desc = TASK_CONFIGS[task_num]

    # 映射工况索引到实际工况值（H1=0, H2=1, H3=2, H4=3）
    condition_list = [CONDITIONS[f"H{i + 1}"] for i in range(4)]
    source_conditions = [condition_list[idx] for idx in source_idxs]
    target_condition = condition_list[target_idx]

    # 格式化源工况字符串
    formatted_source_conds = format_source_conditions(source_conditions)

    # 填充模板
    yaml_content = BASE_TEMPLATE.format(
        task_num=task_num,
        task_desc=task_desc,
        target_idx=target_idx,
        source_conditions=formatted_source_conds,
        target_condition=target_condition,
        fault_num=fault_num
    )

    # 定义文件名
    file_name = f"PU_T{task_num}_300-{fault_num}.yaml"
    file_path = os.path.join(output_dir, file_name)

    # 写入文件（保持原格式）
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(yaml_content.strip())

    print(f"已生成文件: {file_path}")


def generate_all_yaml_files():
    """生成所有16个YAML文件（4个任务 × 4个fault值）"""
    for task_num in TASK_CONFIGS.keys():
        for fault_num in FAULT_VALUES:
            generate_yaml_file(task_num, fault_num)


if __name__ == "__main__":
    # 主函数：执行全部文件生成
    generate_all_yaml_files()
    print("\n所有YAML文件生成完成！")