import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd


def check_preprocessed_data(output_dir: str) -> None:
    """
    检查预处理后的数据基本信息并生成报告

    Args:
        output_dir: 预处理数据的输出目录
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        raise ValueError(f"输出目录不存在: {output_dir}")

    # 收集所有场景信息
    scene_info = []
    total_points = 0
    total_scenes = 0
    total_class0 = 0
    total_class1 = 0
    total_class2 = 0

    # 获取所有场景目录
    # 只查找 train 和 val 目录下的场景
    scene_dirs = []
    for split in ["train", "val"]:
        split_dir = output_path / split
        if split_dir.exists() and split_dir.is_dir():
            # 查找该目录下以 scene 开头的子目录
            scenes = [d for d in split_dir.iterdir() if d.is_dir() and d.name.startswith("scene")]
            scene_dirs.extend(scenes)

    print(f"开始检查数据，共发现 {len(scene_dirs)} 个场景目录...")

    # 遍历每个场景
    for scene_dir in tqdm(scene_dirs, desc="检查场景"):
        scene_name = scene_dir.name
        total_scenes += 1

        # 检查必要的文件是否存在
        required_files = ["coord.npy", "color.npy", "normal.npy", "segment20.npy"]
        missing_files = [f for f in required_files if not (scene_dir / f).exists()]

        if missing_files:
            print(f"[警告] 场景 {scene_name} 缺少文件: {missing_files}")
            continue

        # 加载数据
        try:
            coords = np.load(scene_dir / "coord.npy")
            colors = np.load(scene_dir / "color.npy")
            normals = np.load(scene_dir / "normal.npy")
            segments = np.load(scene_dir / "segment20.npy")
        except Exception as e:
            print(f"[错误] 加载场景 {scene_name} 时出错: {str(e)}")
            continue

        # 检查数据一致性
        num_points = coords.shape[0]
        if not (colors.shape[0] == num_points and
                normals.shape[0] == num_points and
                segments.shape[0] == num_points):
            print(f"[警告] 场景 {scene_name} 数据长度不一致: "
                  f"coord={coords.shape[0]}, color={colors.shape[0]}, "
                  f"normal={normals.shape[0]}, segment={segments.shape[0]}")

        # 统计类别分布
        class0 = int((segments == 0).sum())
        class1 = int((segments == 1).sum())
        class2 = int((segments == 2).sum())

        # 检查坐标范围（原始坐标，未归一化）
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        z_min, z_max = coords[:, 2].min(), coords[:, 2].max()

        # 检查颜色范围
        color_min = colors.min()
        color_max = colors.max()
        if not (color_min >= 0 and color_max <= 255):
            print(f"[警告] 场景 {scene_name} 颜色值超出范围: min={color_min}, max={color_max}")

        # 检查法线范围
        normal_min = normals.min()
        normal_max = normals.max()

        # 累计总数
        total_points += num_points
        total_class0 += class0
        total_class1 += class1
        total_class2 += class2

        # 保存场景信息
        scene_info.append({
            "场景名称": scene_name,
            "点数量": num_points,
            "类别0数量": class0,
            "类别1数量": class1,
            "类别2数量": class2,
            "类别0占比": class0 / num_points if num_points > 0 else 0,
            "类别1占比": class1 / num_points if num_points > 0 else 0,
            "类别2占比": class2 / num_points if num_points > 0 else 0,
            "X范围": (x_min, x_max),
            "Y范围": (y_min, y_max),
            "Z范围": (z_min, z_max),
            "颜色值范围": (color_min, color_max),
            "法线值范围": (normal_min, normal_max)
        })

    # 生成报告
    print("\n" + "=" * 50)
    print("数据检查报告")
    print("=" * 50)
    print(f"总场景数: {total_scenes}")
    print(f"总点数量: {total_points}")

    # 增加对总点数量为0的判断
    if total_points > 0:
        print(f"类别0总数量: {total_class0} ({total_class0 / total_points:.2%})")
        print(f"类别1总数量: {total_class1} ({total_class1 / total_points:.2%})")
        print(f"类别2总数量: {total_class2} ({total_class2 / total_points:.2%})")
    else:
        print("没有有效的点数据，无法计算类别占比")

    # 保存详细信息到CSV
    if scene_info:
        df = pd.DataFrame(scene_info)
        report_path = output_path / "preprocessing_report.csv"
        df.to_csv(report_path, index=False, encoding="utf-8-sig")
        print(f"\n详细报告已保存至: {report_path}")

        # 显示前5个场景的信息
        print("\n前5个场景的详细信息:")
        print(df.head().to_string())


if __name__ == "__main__":
    # 请修改为你的预处理数据输出目录
    OUTPUT_DIR = r"D:\user\Documents\ai\paper\1_process\dataSet\data_scannet_tower\Scencnet"
    # OUTPUT_DIR = r"D:\user\Documents\ai\paper\1_process\dataSet\data_scannet_tower\tower"
    check_preprocessed_data(OUTPUT_DIR)
