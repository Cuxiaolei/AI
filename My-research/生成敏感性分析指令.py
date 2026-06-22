CONFIG_PATH = "src/configs/mcpdg.yaml"
ROOT_DATA = "/root/data"
ERROR_LOG = "experiment_error.log"

dataset_info = [
    ("phm-spur", "T1"),
    ("phm-spur", "T3"),
    ("pu", "T5"),
    ("pu", "T8")
]
replica_suffix = ["1", "5"]

with open("mingan.sh", "w", encoding="utf-8") as f:
    f.write("#!/bin/bash\n")
    f.write("set -e\n\n")
    f.write(f'ERROR_LOG="{ERROR_LOG}"\n')
    f.write(f'> $ERROR_LOG\n\n')

    # ========== 第一组：imbalance_power=0.5，遍历 alpha 0~1.0 step 0.1 ==========
    f.write("# ==================== 第一组实验：imbalance_power=0.5 ====================\n")
    f.write('echo -e "\\n==================== 第一组实验：imbalance_power=0.5 ====================\\n"\n\n')
    fixed_ip = 0.5
    alpha_list = [round(i * 0.1, 1) for i in range(11)]

    for alpha in alpha_list:
        for data_dir, t_val in dataset_info:
            for suf in replica_suffix:
                folder = f"{data_dir}_{t_val}_{suf}"
                train_h5 = f"{ROOT_DATA}/{data_dir}/{folder}/train.h5"
                test_h5 = f"{ROOT_DATA}/{data_dir}/{folder}/test.h5"

                alpha_str = str(alpha).replace(".", "")
                ip_str = str(fixed_ip).replace(".", "")
                exp_name = f"mcpdg_{data_dir.lower()}_{t_val}_{suf}_al{alpha_str}_ip{ip_str}"

                # 数据集参数
                if data_dir.startswith("phm"):
                    dataset = "phm"
                    num_class = 8
                else:
                    dataset = "pu"
                    num_class = 9

                # 实验开始打印
                f.write(f'echo "实验（{exp_name}）准备开始"\n')

                # 训练命令 + 错误容错
                cmd = (
                    f'python -m src.main --configs {CONFIG_PATH} '
                    f'data.train_h5="{train_h5}" '
                    f'data.test_h5="{test_h5}" '
                    f'data.dataset_name="{dataset}" '
                    f'data.num_classes={num_class} '
                    f'model.proto_residual_alpha={alpha} '
                    f'imbalance_power={fixed_ip} '
                    f'output.exp_name="{exp_name}" '
                    f'|| {{ echo "[`date +%Y-%m-%d\ %H:%M:%S`] {exp_name} 执行失败" >> $ERROR_LOG; echo "==== 实验【{exp_name}】运行异常，跳过当前任务，继续执行下一组 ===="; }}\n\n'
                )
                f.write(cmd)

    # ========== 第二组：alpha=0.5，遍历 imbalance_power 0~1.0 step 0.1 ==========
    f.write("# ==================== 第二组实验：alpha=0.5 ====================\n")
    f.write('echo -e "\\n==================== 第二组实验：alpha=0.5 ====================\\n"\n\n')
    fixed_alpha = 0.5
    ip_list = [round(i * 0.1, 1) for i in range(11)]

    for ip in ip_list:
        for data_dir, t_val in dataset_info:
            for suf in replica_suffix:
                folder = f"{data_dir}_{t_val}_{suf}"
                train_h5 = f"{ROOT_DATA}/{data_dir}/{folder}/train.h5"
                test_h5 = f"{ROOT_DATA}/{data_dir}/{folder}/test.h5"

                alpha_str = str(fixed_alpha).replace(".", "")
                ip_str = str(ip).replace(".", "")
                exp_name = f"mcpdg_{data_dir.lower()}_{t_val}_{suf}_al{alpha_str}_ip{ip_str}"

                # 数据集参数
                if data_dir.startswith("phm"):
                    dataset = "phm"
                    num_class = 8
                else:
                    dataset = "pu"
                    num_class = 9

                f.write(f'echo "实验（{exp_name}）准备开始"\n')

                cmd = (
                    f'python -m src.main --configs {CONFIG_PATH} '
                    f'data.train_h5="{train_h5}" '
                    f'data.test_h5="{test_h5}" '
                    f'data.dataset_name="{dataset}" '
                    f'data.num_classes={num_class} '
                    f'model.proto_residual_alpha={fixed_alpha} '
                    f'imbalance_power={ip} '
                    f'output.exp_name="{exp_name}" '
                    f'|| {{ echo "[`date +%Y-%m-%d\ %H:%M:%S`] {exp_name} 执行失败" >> $ERROR_LOG; echo "==== 实验【{exp_name}】运行异常，跳过当前任务，继续执行下一组 ===="; }}\n\n'
                )
                f.write(cmd)

    # 全部跑完关机
    f.write('echo "======== 所有实验任务执行完毕，错误日志见：$ERROR_LOG，准备关机 ========"\n')
    f.write('shutdown -h now\n')

print("已生成逐条展开的调试脚本：run_debug.sh")