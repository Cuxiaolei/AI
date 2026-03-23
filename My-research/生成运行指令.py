def generate_model_file_paths():
    """
    生成7个模型的所有文件路径列表（每个模型48个路径，数据集间空行分隔）
    返回：str - 格式化后的路径列表字符串
    """
    # 1. 配置基础参数
    models = ["darm", "dpjdg", "erm", "groupdro", "irm", "mixstyle", "mldg", "vrex", "mcpdg"]  # 7个模型
    num_vals = ["1", "5", "10", "20"]  # 数值后缀
    datasets = [
        {"name": "phm", "t_list": ["T1", "T2", "T3", "T4"]},  # PHM数据集
        {"name": "pu", "t_list": ["T5", "T6", "T7", "T8"]}  # PU数据集
    ]

    # 2. 生成每个模型的路径列表
    final_output = []
    for model in models:
        # 添加模型标题（可选，便于区分）
        final_output.append(f"echo {model} 模型")

        # 遍历每个数据集，生成路径
        for ds_idx, ds in enumerate(datasets):
            ds_paths = []
            for t_val in ds["t_list"]:
                for num_val in num_vals:
                    path = f'python -m src.main --configs src/configs/{model}/{model}_{ds["name"]}_{t_val}_{num_val}.yaml'
                    ds_paths.append(path)

            # 将当前数据集的路径加入结果
            final_output.extend(ds_paths)

            # 数据集之间添加空行（最后一个数据集不添加）
            if ds_idx < len(datasets) - 1:
                final_output.append("")

        # 模型之间添加分隔空行（最后一个模型不添加）
        if model != models[-1]:
            final_output.append("")

    # 3. 将列表转为格式化的字符串
    return "\n".join(final_output)


def save_to_file(content, filename="运行指令-model_file_paths.txt"):
    """
    将生成的路径列表保存到文件

    Args:
        content (str): 生成的路径内容
        filename (str): 保存的文件名
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✅ 路径列表已保存到文件：{filename}")


if __name__ == "__main__":
    # 生成路径列表
    path_content = generate_model_file_paths()

    # 打印到控制台（可直接复制）
    print("=" * 50)
    print("生成的完整路径列表：")
    print("=" * 50)
    print(path_content)

    # 保存到文件（方便后续使用）
    save_to_file(path_content)

    # 输出汇总信息
    total_lines = len([line for line in path_content.split("\n") if line.strip() and not line.startswith("###")])
    print(f"\n🎉 生成完成！总计 {total_lines} 个文件路径")