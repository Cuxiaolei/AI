import os


def generate_single_yaml(output_dir, dataset_info, t_val, num_val, model, conunt):
    # 解析数据集配置
    ds_name = dataset_info["name"]
    ds_path_prefix = dataset_info["path_prefix"]
    ds_class = dataset_info["class"]


    # 构建YAML内容模板
    yaml_content = f"""
BASE: ../{model}.yaml

data:
  train_h5: /root/data/{ds_path_prefix}/{ds_path_prefix}_{t_val}_{num_val}/train.h5
  test_h5: /root/data/{ds_path_prefix}/{ds_path_prefix}_{t_val}_{num_val}/test.h5
  dataset_name: {ds_name}
  num_classes: {ds_class}

output:
  exp_name: {model}_{ds_name}_{t_val}_{num_val}-{conunt}
"""

    # 生成文件路径
    filename = f"{model}_{ds_name}_{t_val}_{num_val}-{conunt}.yaml"
    file_path = os.path.join(output_dir, filename)

    # 写入文件（UTF-8编码避免乱码）
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    return file_path


def main():
    """主函数：控制48个YAML文件的生成流程"""
    # ===================== 核心配置（严格按要求定义） =====================
    script_dir = os.path.dirname(os.path.abspath(__file__))  # 脚本所在目录的绝对路径
    output_dir = script_dir  # 输出目录 = 脚本所在目录
    num_vals = [
        # "1",
        "5",
        # "10"
    ]  # 4个数值后缀（固定）
    conunt = 5

    # 修改这个
    model = "mcpdgE2"

    # 3个数据集的完整配置（每个4个T值）
    datasets = [
        # 1. PHM数据集：T1-T4 → 4个T值 × 4个数值 = 16个文件
        {
            "name": "phm",
            "class": 8,
            "path_prefix": "phm-spur",
            "t_list": ["T1", "T2", "T3", "T4"],
            "spur_num": 8  # PHM_spur固定
        },
        # 2. PU数据集：T5-T8 → 4个T值 × 4个数值 = 16个文件
        # {
        #     "name": "pu",
        #     "class": 9,
        #     "path_prefix": "pu",
        #     "t_list": ["T5", "T6", "T7", "T8"],
        #     "spur_num": ""  # 非PHM数据集无需spur
        # }
    ]

    # ===================== 执行生成流程 =====================
    # 1. 创建输出目录（不存在则自动创建）
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 输出目录：{os.path.abspath(output_dir)}")
    generated_count = 0
    generated_files = []

    # 2. 遍历所有组合生成文件
    for ds in datasets:
        print(f"\n🔄 开始生成【{ds['name']}】数据集文件（T值：{','.join(ds['t_list'])}）...")
        for t_val in ds["t_list"]:
            for num_val in num_vals:
                file_path = generate_single_yaml(output_dir, ds, t_val, num_val, model, conunt)
                generated_files.append(file_path)
                generated_count += 1
                print(f"✅ 生成：{os.path.basename(file_path)}")

    # 3. 输出汇总信息
    print(f"\n🎉 生成完成！总计生成 {generated_count} 个YAML文件")
    print(f"📂 文件存储路径：{os.path.abspath(output_dir)}")
    print(f"📊 明细：PHM(16个) + PU(16个) = 48个")


if __name__ == "__main__":
    main()