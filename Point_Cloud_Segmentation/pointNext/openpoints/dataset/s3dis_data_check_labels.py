import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# --------------------------
# 配置参数（与预处理脚本保持一致）
# --------------------------
# 标签映射与类别名称
CLASS_MAPPING = {
    0: "前景(铁塔/绝缘子)",
    1: "背景(建筑物/地面等)",
    2: "前景(导线/地线/引流线)"
}
VALID_LABELS = {0, 1, 2}  # 仅允许的三类标签
FEATURE_DIM = 10  # S3DIS 格式固定为 N×10 维（3坐标+3颜色+3法线+1标签）
REQUIRED_DIRS = ["merged"]  # 必须存在的子目录
REQUIRED_SPLIT_FILES = ["train_scenes.txt", "val_scenes.txt", "test_scenes.txt"]  # 数据集划分文件


def validate_s3dis_data(root_dir: str) -> None:
    """
    验证 S3DIS 格式预处理数据的完整性、格式正确性与数据质量

    Args:
        root_dir: 预处理数据根目录（包含 merged 文件夹和划分文件）
    """
    # 1. 基础目录检查
    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"根目录不存在：{root_dir}")

    # 检查必要子目录（merged）
    dir_check_result = {}
    for dir_name in REQUIRED_DIRS:
        dir_path = root_path / dir_name
        dir_check_result[dir_name] = dir_path.exists()
        if not dir_check_result[dir_name]:
            raise NotADirectoryError(f"必需子目录缺失：{dir_path}（预处理流程应生成该目录）")

    # 2. 数据集划分文件检查
    split_files_info = {}
    missing_split_files = []
    for file_name in REQUIRED_SPLIT_FILES:
        file_path = root_path / file_name
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                # 读取场景列表（去除空行）
                scenes = [line.strip() for line in f if line.strip()]
            split_files_info[file_name] = {
                "exists": True,
                "scene_count": len(scenes),
                "scenes": scenes
            }
        else:
            split_files_info[file_name] = {"exists": False, "scene_count": 0, "scenes": []}
            missing_split_files.append(file_name)

    # 3. 获取所有场景文件（仅识别 Area_*.npy 格式）
    merged_dir = root_path / "merged"
    scene_files = list(merged_dir.glob("Area_*.npy"))
    if not scene_files:
        raise ValueError(f"merged 目录下无有效场景文件（需为 Area_*.npy 格式）：{merged_dir}")
    scene_files.sort()  # 按文件名排序，确保结果可复现

    # 4. 初始化统计变量
    total_metrics = {
        "total_scenes": 0,  # 总场景数
        "valid_scenes": 0,  # 有效场景数（无异常）
        "empty_scenes": 0,  # 空场景数（点数=0）
        "abnormal_scenes": 0,  # 异常场景数（格式/数据错误）
        "total_points": 0,  # 全量总点数
        "class_distribution": {  # 各类别总点数
            0: 0,
            1: 0,
            2: 0
        }
    }

    # 存储每个场景的详细信息（用于生成CSV报告）
    scene_detail_records = []

    # --------------------------
    # 5. 逐场景检查核心逻辑
    # --------------------------
    print("=" * 70)
    print(f"开始检查 S3DIS 格式预处理数据 | 根目录：{root_dir}")
    print("=" * 70)
    print(f"基础检查结果：")
    print(f"  - merged 目录：存在（{merged_dir}）")
    print(f"  - 场景文件数量：{len(scene_files)} 个（Area_*.npy）")
    print(f"  - 划分文件状态：{'全部存在' if not missing_split_files else f'缺失 {missing_split_files}'}")
    for split_file, info in split_files_info.items():
        if info["exists"]:
            print(f"    · {split_file}：{info['scene_count']} 个场景")
        else:
            print(f"    · {split_file}：缺失")
    print("\n" + "-" * 70)
    print("逐场景详细检查（进度）：")
    print("-" * 70)

    for scene_file in tqdm(scene_files, desc="处理场景", unit="个"):
        scene_name = scene_file.name  # 场景文件名（如 Area_1.npy）
        scene_path = str(scene_file)
        total_metrics["total_scenes"] += 1

        # 初始化当前场景记录
        scene_record = {
            "场景文件名": scene_name,
            "场景路径": scene_path,
            "总点数": 0,
            "类别0_点数": 0, "类别0_占比(%)": 0.0,
            "类别1_点数": 0, "类别1_占比(%)": 0.0,
            "类别2_点数": 0, "类别2_占比(%)": 0.0,
            "X坐标_范围": "无数据", "Y坐标_范围": "无数据", "Z坐标_范围": "无数据",
            "颜色值_范围": "无数据", "法线值_范围": "无数据",
            "数据状态": "异常",
            "异常原因": ""
        }

        # 加载场景数据（捕获加载错误）
        try:
            scene_data = np.load(scene_path)
        except Exception as e:
            scene_record["异常原因"] = f"文件加载失败：{str(e)[:30]}..."
            scene_detail_records.append(scene_record)
            total_metrics["abnormal_scenes"] += 1
            continue

        # 检查数据维度（必须为 N×10 维）
        if scene_data.ndim != 2 or scene_data.shape[1] != FEATURE_DIM:
            scene_record["异常原因"] = f"维度错误（需 N×10，实际 {scene_data.shape}）"
            scene_detail_records.append(scene_record)
            total_metrics["abnormal_scenes"] += 1
            continue

        # 检查点数（空场景判断）
        num_points = scene_data.shape[0]
        scene_record["总点数"] = num_points
        if num_points == 0:
            scene_record["异常原因"] = "空场景（点数为0）"
            scene_detail_records.append(scene_record)
            total_metrics["empty_scenes"] += 1
            total_metrics["abnormal_scenes"] += 1
            continue

        # --------------------------
        # 拆分 10 维特征并检查
        # --------------------------
        # 3维坐标（0-2列：X/Y/Z，原始坐标，无归一化）
        coords = scene_data[:, 0:3].astype(np.float32)
        x_min, x_max = round(coords[:, 0].min(), 4), round(coords[:, 0].max(), 4)
        y_min, y_max = round(coords[:, 1].min(), 4), round(coords[:, 1].max(), 4)
        z_min, z_max = round(coords[:, 2].min(), 4), round(coords[:, 2].max(), 4)
        scene_record["X坐标_范围"] = f"{x_min} ~ {x_max}"
        scene_record["Y坐标_范围"] = f"{y_min} ~ {y_max}"
        scene_record["Z坐标_范围"] = f"{z_min} ~ {z_max}"

        # 3维颜色（3-5列：R/G/B，需 0-255 整数）
        colors = scene_data[:, 3:6]
        color_min, color_max = int(colors.min()), int(colors.max())
        scene_record["颜色值_范围"] = f"{color_min} ~ {color_max}"
        # 颜色范围检查
        if color_min < 0 or color_max > 255:
            scene_record["异常原因"] = f"颜色值超出 0-255 范围（{color_min}~{color_max}）"
            scene_detail_records.append(scene_record)
            total_metrics["abnormal_scenes"] += 1
            continue

        # 3维法线（6-8列：法向量X/Y/Z，通常范围 [-1,1]）
        normals = scene_data[:, 6:9].astype(np.float32)
        normal_min, normal_max = round(normals.min(), 4), round(normals.max(), 4)
        scene_record["法线值_范围"] = f"{normal_min} ~ {normal_max}"
        # 法线全0检查（提示 Open3D 安装问题）
        if np.allclose(normals, 0):
            scene_record["异常原因"] = "法线全为0（可能未安装 Open3D 或计算失败）"
            # 法线全0不归类为“错误”，仅标记警告
            scene_record["数据状态"] = "警告"
        else:
            scene_record["数据状态"] = "正常"

        # 1维标签（9列：仅允许 0/1/2）
        labels = scene_data[:, 9].astype(np.uint8)
        unique_labels = set(np.unique(labels))
        invalid_labels = unique_labels - VALID_LABELS
        if invalid_labels:
            scene_record["异常原因"] = f"存在无效标签：{sorted(invalid_labels)}（仅允许 0/1/2）"
            scene_detail_records.append(scene_record)
            total_metrics["abnormal_scenes"] += 1
            continue

        # --------------------------
        # 统计类别分布
        # --------------------------
        class_0_cnt = int((labels == 0).sum())
        class_1_cnt = int((labels == 1).sum())
        class_2_cnt = int((labels == 2).sum())
        # 更新场景记录
        scene_record["类别0_点数"] = class_0_cnt
        scene_record["类别0_占比(%)"] = round(class_0_cnt / num_points * 100, 2)
        scene_record["类别1_点数"] = class_1_cnt
        scene_record["类别1_占比(%)"] = round(class_1_cnt / num_points * 100, 2)
        scene_record["类别2_点数"] = class_2_cnt
        scene_record["类别2_占比(%)"] = round(class_2_cnt / num_points * 100, 2)
        # 更新全量统计
        total_metrics["total_points"] += num_points
        total_metrics["class_distribution"][0] += class_0_cnt
        total_metrics["class_distribution"][1] += class_1_cnt
        total_metrics["class_distribution"][2] += class_2_cnt

        # 标记有效场景（无错误，警告场景也计入有效）
        if scene_record["数据状态"] in ["正常", "警告"]:
            total_metrics["valid_scenes"] += 1

        # 保存当前场景记录
        scene_detail_records.append(scene_record)

    # --------------------------
    # 6. 生成最终报告
    # --------------------------
    print("\n" + "=" * 70)
    print("S3DIS 数据检查最终报告")
    print("=" * 70)

    # 6.1 核心统计
    print(f"1. 场景统计：")
    print(f"   - 总场景数：{total_metrics['total_scenes']}")
    print(f"   - 有效场景数：{total_metrics['valid_scenes']}（正常+警告）")
    print(f"   - 空场景数：{total_metrics['empty_scenes']}")
    print(f"   - 异常场景数：{total_metrics['abnormal_scenes']}（需修复）")

    print(f"\n2. 点数统计：")
    if total_metrics["total_points"] > 0:
        print(f"   - 全量总点数：{total_metrics['total_points']:,}")
        print(f"   - 平均每场景点数：{total_metrics['total_points'] // total_metrics['total_scenes']:,}")
    else:
        print(f"   - 全量总点数：0（无有效数据）")

    print(f"\n3. 类别分布（全量）：")
    total_points = total_metrics["total_points"]
    for cls_id, cls_name in CLASS_MAPPING.items():
        cls_cnt = total_metrics["class_distribution"][cls_id]
        cls_ratio = (cls_cnt / total_points * 100) if total_points > 0 else 0.0
        print(f"   - 类别 {cls_id}（{cls_name}）：{cls_cnt:,} 点（{cls_ratio:.2f}%）")

    # 6.2 异常场景提醒
    if total_metrics["abnormal_scenes"] > 0:
        print(f"\n4. 异常提醒：")
        print(f"   ⚠️  共 {total_metrics['abnormal_scenes']} 个异常场景，需优先处理：")
        abnormal_records = [r for r in scene_detail_records if r["数据状态"] == "异常"]
        for idx, rec in enumerate(abnormal_records[:3]):  # 显示前3个异常场景
            print(f"     {idx + 1}. {rec['场景文件名']}：{rec['异常原因']}")
        if len(abnormal_records) > 3:
            print(f"     ... 还有 {len(abnormal_records) - 3} 个异常场景（详见CSV报告）")

    # 6.3 生成 CSV 详细报告
    if scene_detail_records:
        report_df = pd.DataFrame(scene_detail_records)
        report_path = root_path / "s3dis_data_check_report.csv"
        # 保存 CSV（支持中文，用 utf-8-sig 编码）
        report_df.to_csv(report_path, index=False, encoding="utf-8-sig")
        print(f"\n5. 详细报告：")
        print(f"   📄 详细报告已保存至：{report_path}")
        print(f"   报告包含：每个场景的点数、类别分布、坐标/颜色/法线范围、数据状态等信息")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    # --------------------------
    # 请修改为你的 S3DIS 数据根目录
    # （即预处理脚本中的 S3DIS_OUTPUT_DIR，包含 merged 文件夹）
    # --------------------------
    S3DIS_ROOT_DIR = r"D:\user\Documents\ai\paper\1_process\dataSet\s3dis_pointNeXt\output"
    # S3DIS_ROOT_DIR = r"D:\user\Documents\ai\paper\1_process\dataSet\s3dis_pointNeXt\s3dis_有法向量"

    # 执行检查
    try:
        validate_s3dis_data(S3DIS_ROOT_DIR)
    except (FileNotFoundError, NotADirectoryError, ValueError) as e:
        print(f"\n❌ 检查脚本启动失败：{str(e)}")
    except Exception as e:
        print(f"\n❌ 检查过程中发生未知错误：{str(e)}")
