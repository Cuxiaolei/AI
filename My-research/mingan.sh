#!/bin/bash
set -e

CONFIG_PATH="src/configs/mcpdg.yaml"
ROOT_DATA="data"

# 数据集配对：数据集目录名 T编号
dataset_info=(
    "phm-spur" "T1"
    "phm-spur" "T3"
    "pu" "T5"
    "pu" "T8"
)
# 两个副本后缀#
replica_suffix=("1" "5")

# ====================== 第一波：imbalance_power固定0.5，遍历alpha 0~1.0 步长0.1 ======================
echo -e "\n==================== 第一组实验：imbalance_power=0.5 ====================\n"
IMBALANCE_POWER=0.5
for ALPHA in $(seq 0 0.1 1.0)
do
    # 成对遍历数据集
    for ((i=0; i<${#dataset_info[@]}; i+=2))
    do
        DATA_DIR=${dataset_info[$i]}
        T_VAL=${dataset_info[$i+1]}
        # 遍历两个副本
        for suffix in "${replica_suffix[@]}"
        do
            folder="${DATA_DIR}_${T_VAL}_${suffix}"
            TRAIN_H5="${ROOT_DATA}/${DATA_DIR}/${folder}/train.h5"
            TEST_H5="${ROOT_DATA}/${DATA_DIR}/${folder}/test.h5"
            # 实验命名
            EXP_NAME="mcpdg_${DATA_DIR,,}_${T_VAL}_${suffix}_al${ALPHA#*.}_ip${IMBALANCE_POWER#*.}"

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

# ====================== 第二波：alpha固定0.5，遍历imbalance_power 0~1.0 步长0.1 ======================
echo -e "\n==================== 第二组实验：alpha=0.5 ====================\n"
ALPHA=0.5
for IMBALANCE_POWER in $(seq 0 0.1 1.0)
do
    for ((i=0; i<${#dataset_info[@]}; i+=2))
    do
        DATA_DIR=${dataset_info[$i]}
        T_VAL=${dataset_info[$i+1]}
        for suffix in "${replica_suffix[@]}"
        do
            folder="${DATA_DIR}_${T_VAL}_${suffix}"
            TRAIN_H5="${ROOT_DATA}/${DATA_DIR}/${folder}/train.h5"
            TEST_H5="${ROOT_DATA}/${DATA_DIR}/${folder}/test.h5"
            EXP_NAME="mcpdg_${DATA_DIR,,}_${T_VAL}_${suffix}_al${ALPHA#*.}_ip${IMBALANCE_POWER#*.}"

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