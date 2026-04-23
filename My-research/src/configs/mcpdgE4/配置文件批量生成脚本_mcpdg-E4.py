import os


def generate_single_yaml(output_dir, dataset_info, t_val, num_val, model, count):
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
  exp_name: {model}_{ds_name}_{t_val}_{num_val}-{count}
"""

    # 生成文件路径 —— 这里已修复！！！
    filename = f"{model}_{ds_name}_{t_val}_{num_val}-{count}.yaml"
    file_path = os.path.join(output_dir, filename)  # 修复了 os.path -> os.path.join

    # 写入文件（UTF-8编码避免乱码）
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    return file_path


def main():
    """主函数：count 1-10 循环批量生成YAML"""
    # ===================== 核心配置 =====================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = script_dir
    num_vals = [
        # "1",
        "5",
        # "10"
    ]
    # count 改为 1~10
    count_list = list(range(1, 11))

    # 模型名称
    model = "mcpdgE4"

    # 数据集配置
    datasets = [
        {
            "name": "phm",
            "class": 8,
            "path_prefix": "phm-spur",
            "t_list": ["T1", "T2", "T3", "T4"],
            "spur_num": 8
        },
        {
            "name": "pu",
            "class": 9,
            "path_prefix": "pu",
            "t_list": ["T5", "T6", "T7", "T8"],
            "spur_num": ""
        }
    ]

    # ===================== 执行生成 =====================
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 输出目录：{os.path.abspath(output_dir)}")
    generated_count = 0

    # 四层循环：数据集 → T → num → count
    for ds in datasets:
        print(f"\n🔄 开始生成【{ds['name']}】数据集文件...")
        for t_val in ds["t_list"]:
            for num_val in num_vals:
                for count in count_list:
                    file_path = generate_single_yaml(output_dir, ds, t_val, num_val, model, count)
                    generated_count += 1
                    print(f"✅ 生成：{os.path.basename(file_path)}")

    # 汇总
    print(f"\n🎉 全部生成完成！总计：{generated_count} 个 YAML 文件")


if __name__ == "__main__":
    main()