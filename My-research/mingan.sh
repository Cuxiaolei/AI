#!/bin/bash

# 全局基础配置
CONFIG_PATH="src/configs/mcpdg.yaml"
ROOT_DATA="/root/data"

# 数据集列表：数据集类型 前缀编号 副本(1/5)
# PHM: T1 T3
# PU: T5 T8
dataset_info=(
    "phm-spur" "T1"
    "phm-spur" "T3"
    "pu" "T5"
    "pu" "T8"
)
# 副本后缀
replicas=("1" "5")

# ========== 第一波：IMBALANCE_POWER固定0.5，ALPHA从0到1，步长0.1 ==========
echo -e "\n==================== 第一组实验：imbalance_power=0.5，alpha遍历0~1(0.1步长) ====================\n"
IMBALANCE_POWER=0.5
# alpha序列：0.0 0.1 ... 1.0
for ALPHA in $(seq 0 0.1 1.0)
do
    for item in "${dataset_info[@]}"
    do
        # 拆分数据集类型与编号
        dataset_type=$(echo $item | awk '{print $1}')
        T_val=$(echo $item | awk '{print $2}')
        for rep in "${replicas[@]}"
        do
            data_dir="${ROOT_DATA}/${dataset_type}/${dataset_type}_${T_val}_${rep}"
            TRAIN_H5="${data_dir}/train.h5"
            TEST_H5="${data_dir}/test.h5"
            EXP_NAME="mcpdg_${dataset_type}_${T_val}_${rep}_al${ALPHA#*.}_ip${IMBALANCE_POWER#*.}"

            echo "【第一波】开始运行: ${EXP_NAME}"
            python -m src.main --configs "${CONFIG_PATH}" \
                data.train_h5="${TRAIN_H5}" \
                data.test_h5="${TEST_H5}" \
                model.proto_residual_alpha="${ALPHA}" \
                imbalance_power="${IMBALANCE_POWER}" \
                output.exp_name="${EXP_NAME}"
        done
    done
done

# ========== 第二波：ALPHA固定0.5，IMBALANCE_POWER从0到1，步长0.1 ==========
echo -e "\n==================== 第二组实验：alpha=0.5，imbalance_power遍历0~1(0.1步长) ====================\n"
ALPHA=0.5
for IMBALANCE_POWER in $(seq 0 0.1 1.0)
do
    for item in "${dataset_info[@]}"
    do
        dataset_type=$(echo $item | awk '{print $1}')
        T_val=$(echo $item | awk '{print $2}')
        for rep in "${replicas[@]}"
        do
            data_dir="${ROOT_DATA}/${dataset_type}/${dataset_type}_${T_val}_${rep}"
            TRAIN_H5="${data_dir}/train.h5"
            TEST_H5="${data_dir}/test.h5"
            EXP_NAME="mcpdg_${dataset_type}_${T_val}_${rep}_al${ALPHA#*.}_ip${IMBALANCE_POWER#*.}"

            echo "【第二波】开始运行: ${EXP_NAME}"
            python -m src.main --configs "${CONFIG_PATH}" \
                data.train_h5="${TRAIN_H5}" \
                data.test_h5="${TEST_H5}" \
                model.proto_residual_alpha="${ALPHA}" \
                imbalance_power="${IMBALANCE_POWER}" \
                output.exp_name="${EXP_NAME}"
        done
    done
done

echo "======== 所有实验任务全部执行完成 ========"
shutdown -h now