# 导出所有命令到 commands.txt
CONFIG_PATH = "src/configs/mcpdg.yaml"
ROOT_DATA = "/root/data"
dataset_info = [
    ("phm-spur", "T1"),
    ("phm-spur", "T3"),
    ("pu", "T5"),
    ("pu", "T8")
]
replica_suffix = ["1", "5"]

with open("commands.txt", "w", encoding="utf-8") as f:
    # 第一组
    f.write("==================== 第一组实验：imbalance_power=0.5 ====================\n")
    ip = 0.5
    alpha_list = [round(i*0.1,1) for i in range(11)]
    for alpha in alpha_list:
        for data_dir, t in dataset_info:
            for suf in replica_suffix:
                folder = f"{data_dir}_{t}_{suf}"
                train = f"{ROOT_DATA}/{data_dir}/{folder}/train.h5"
                test = f"{ROOT_DATA}/{data_dir}/{folder}/test.h5"
                an = str(alpha).replace(".","")
                ipn = str(ip).replace(".","")
                exp = f"mcpdg_{data_dir.lower()}_{t}_{suf}_al{an}_ip{ipn}"
                cmd = (
                    f'python -m src.main --configs {CONFIG_PATH} '
                    f'data.train_h5="{train}" '
                    f'data.test_h5="{test}" '
                    f'model.proto_residual_alpha={alpha} '
                    f'imbalance_power={ip} '
                    f'output.exp_name="{exp}" '
                    f'|| echo "==== 实验【${exp}】运行异常，跳过当前任务，继续执行下一组 ===="\n'
                )
                f.write(cmd)

    # 第二组
    f.write("\n==================== 第二组实验：alpha=0.5 ====================\n")
    alpha = 0.5
    ip_list = [round(i*0.1,1) for i in range(11)]
    for ip in ip_list:
        for data_dir, t in dataset_info:
            for suf in replica_suffix:
                folder = f"{data_dir}_{t}_{suf}"
                train = f"{ROOT_DATA}/{data_dir}/{folder}/train.h5"
                test = f"{ROOT_DATA}/{data_dir}/{folder}/test.h5"
                an = str(alpha).replace(".","")
                ipn = str(ip).replace(".","")
                exp = f"mcpdg_{data_dir.lower()}_{t}_{suf}_al{an}_ip{ipn}"
                cmd = (
                    f'python -m src.main --configs {CONFIG_PATH} '
                    f'data.train_h5="{train}" '
                    f'data.test_h5="{test}" '
                    f'model.proto_residual_alpha={alpha} '
                    f'imbalance_power={ip} '
                    f'output.exp_name="{exp}" '
                    f'|| echo "==== 实验【${exp}】运行异常，跳过当前任务，继续执行下一组 ===="\n'
                )
                f.write(cmd)

    f.write("\n# 所有任务执行完毕后关机\nshutdown -h now\n")

print("所有运行命令已保存至 commands.txt")