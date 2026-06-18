#!/bin/bash

# 模型列表
models=(
    "darm"
    "dfdn"
    "dpjdg"
    "erm"
    "masfd"
    "mcpdg"
    "mldg"
    "sdagn"
    "vrex"
)

# 固定任务标识
task_tag="T2_5"

# 循环执行所有模型
for model in "${models[@]}"; do
    config_path="src/configs/${model}/${model}_phm_${task_tag}-1.yaml"
    exp_name="${model}_phm_${task_tag}"

    echo "==================== 开始训练模型：${model} ===================="
    python -m src.main --configs "${config_path}" \
        output.save_checkpoint=false \
        output.exp_name="${exp_name}"
done

echo "所有9个模型训练执行完毕"