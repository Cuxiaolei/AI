#!/bin/bash
set -e

ERROR_LOG="experiment_error.log"
> $ERROR_LOG

# ==================== 第一组实验：imbalance_power=0.5 ====================
echo -e "\n==================== 第一组实验：imbalance_power=0.5 ====================\n"

#echo "实验（mcpdg_phm-spur_T1_1_al00_ip05）准备开始"
#python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T1_1/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T1_1/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.0 imbalance_power=0.5 output.exp_name="mcpdg_phm-spur_T1_1_al00_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T1_1_al00_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T1_1_al00_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }
#
#echo "实验（mcpdg_phm-spur_T1_5_al00_ip05）准备开始"
#python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T1_5/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T1_5/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.0 imbalance_power=0.5 output.exp_name="mcpdg_phm-spur_T1_5_al00_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T1_5_al00_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T1_5_al00_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }
#
#echo "实验（mcpdg_phm-spur_T3_1_al00_ip05）准备开始"
#python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T3_1/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T3_1/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.0 imbalance_power=0.5 output.exp_name="mcpdg_phm-spur_T3_1_al00_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T3_1_al00_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T3_1_al00_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }
#
#echo "实验（mcpdg_phm-spur_T3_5_al00_ip05）准备开始"
#python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T3_5/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T3_5/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.0 imbalance_power=0.5 output.exp_name="mcpdg_phm-spur_T3_5_al00_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T3_5_al00_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T3_5_al00_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T5_1_al00_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T5_1/train.h5" data.test_h5="/root/data/pu/pu_T5_1/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.0 imbalance_power=0.5 output.exp_name="mcpdg_pu_T5_1_al00_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T5_1_al00_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T5_1_al00_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T5_5_al00_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T5_5/train.h5" data.test_h5="/root/data/pu/pu_T5_5/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.0 imbalance_power=0.5 output.exp_name="mcpdg_pu_T5_5_al00_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T5_5_al00_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T5_5_al00_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T8_1_al00_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T8_1/train.h5" data.test_h5="/root/data/pu/pu_T8_1/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.0 imbalance_power=0.5 output.exp_name="mcpdg_pu_T8_1_al00_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T8_1_al00_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T8_1_al00_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T8_5_al00_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T8_5/train.h5" data.test_h5="/root/data/pu/pu_T8_5/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.0 imbalance_power=0.5 output.exp_name="mcpdg_pu_T8_5_al00_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T8_5_al00_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T8_5_al00_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T1_1_al01_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T1_1/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T1_1/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.1 imbalance_power=0.5 output.exp_name="mcpdg_phm-spur_T1_1_al01_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T1_1_al01_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T1_1_al01_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T1_5_al01_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T1_5/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T1_5/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.1 imbalance_power=0.5 output.exp_name="mcpdg_phm-spur_T1_5_al01_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T1_5_al01_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T1_5_al01_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T3_1_al01_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T3_1/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T3_1/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.1 imbalance_power=0.5 output.exp_name="mcpdg_phm-spur_T3_1_al01_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T3_1_al01_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T3_1_al01_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T3_5_al01_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T3_5/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T3_5/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.1 imbalance_power=0.5 output.exp_name="mcpdg_phm-spur_T3_5_al01_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T3_5_al01_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T3_5_al01_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T5_1_al01_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T5_1/train.h5" data.test_h5="/root/data/pu/pu_T5_1/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.1 imbalance_power=0.5 output.exp_name="mcpdg_pu_T5_1_al01_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T5_1_al01_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T5_1_al01_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T5_5_al01_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T5_5/train.h5" data.test_h5="/root/data/pu/pu_T5_5/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.1 imbalance_power=0.5 output.exp_name="mcpdg_pu_T5_5_al01_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T5_5_al01_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T5_5_al01_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T8_1_al01_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T8_1/train.h5" data.test_h5="/root/data/pu/pu_T8_1/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.1 imbalance_power=0.5 output.exp_name="mcpdg_pu_T8_1_al01_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T8_1_al01_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T8_1_al01_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T8_5_al01_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T8_5/train.h5" data.test_h5="/root/data/pu/pu_T8_5/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.1 imbalance_power=0.5 output.exp_name="mcpdg_pu_T8_5_al01_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T8_5_al01_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T8_5_al01_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T1_1_al02_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T1_1/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T1_1/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.2 imbalance_power=0.5 output.exp_name="mcpdg_phm-spur_T1_1_al02_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T1_1_al02_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T1_1_al02_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T1_5_al02_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T1_5/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T1_5/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.2 imbalance_power=0.5 output.exp_name="mcpdg_phm-spur_T1_5_al02_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T1_5_al02_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T1_5_al02_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T3_1_al02_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T3_1/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T3_1/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.2 imbalance_power=0.5 output.exp_name="mcpdg_phm-spur_T3_1_al02_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T3_1_al02_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T3_1_al02_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T3_5_al02_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T3_5/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T3_5/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.2 imbalance_power=0.5 output.exp_name="mcpdg_phm-spur_T3_5_al02_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T3_5_al02_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T3_5_al02_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T5_1_al02_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T5_1/train.h5" data.test_h5="/root/data/pu/pu_T5_1/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.2 imbalance_power=0.5 output.exp_name="mcpdg_pu_T5_1_al02_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T5_1_al02_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T5_1_al02_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T5_5_al02_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T5_5/train.h5" data.test_h5="/root/data/pu/pu_T5_5/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.2 imbalance_power=0.5 output.exp_name="mcpdg_pu_T5_5_al02_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T5_5_al02_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T5_5_al02_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T8_1_al02_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T8_1/train.h5" data.test_h5="/root/data/pu/pu_T8_1/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.2 imbalance_power=0.5 output.exp_name="mcpdg_pu_T8_1_al02_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T8_1_al02_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T8_1_al02_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T8_5_al02_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T8_5/train.h5" data.test_h5="/root/data/pu/pu_T8_5/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.2 imbalance_power=0.5 output.exp_name="mcpdg_pu_T8_5_al02_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T8_5_al02_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T8_5_al02_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T1_1_al03_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T1_1/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T1_1/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.3 imbalance_power=0.5 output.exp_name="mcpdg_phm-spur_T1_1_al03_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T1_1_al03_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T1_1_al03_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T1_5_al03_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T1_5/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T1_5/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.3 imbalance_power=0.5 output.exp_name="mcpdg_phm-spur_T1_5_al03_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T1_5_al03_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T1_5_al03_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T3_1_al03_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T3_1/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T3_1/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.3 imbalance_power=0.5 output.exp_name="mcpdg_phm-spur_T3_1_al03_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T3_1_al03_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T3_1_al03_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T3_5_al03_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T3_5/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T3_5/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.3 imbalance_power=0.5 output.exp_name="mcpdg_phm-spur_T3_5_al03_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T3_5_al03_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T3_5_al03_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T5_1_al03_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T5_1/train.h5" data.test_h5="/root/data/pu/pu_T5_1/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.3 imbalance_power=0.5 output.exp_name="mcpdg_pu_T5_1_al03_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T5_1_al03_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T5_1_al03_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T5_5_al03_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T5_5/train.h5" data.test_h5="/root/data/pu/pu_T5_5/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.3 imbalance_power=0.5 output.exp_name="mcpdg_pu_T5_5_al03_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T5_5_al03_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T5_5_al03_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T8_1_al03_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T8_1/train.h5" data.test_h5="/root/data/pu/pu_T8_1/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.3 imbalance_power=0.5 output.exp_name="mcpdg_pu_T8_1_al03_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T8_1_al03_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T8_1_al03_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T8_5_al03_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T8_5/train.h5" data.test_h5="/root/data/pu/pu_T8_5/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.3 imbalance_power=0.5 output.exp_name="mcpdg_pu_T8_5_al03_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T8_5_al03_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T8_5_al03_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T1_1_al04_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T1_1/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T1_1/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.4 imbalance_power=0.5 output.exp_name="mcpdg_phm-spur_T1_1_al04_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T1_1_al04_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T1_1_al04_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T1_5_al04_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T1_5/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T1_5/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.4 imbalance_power=0.5 output.exp_name="mcpdg_phm-spur_T1_5_al04_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T1_5_al04_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T1_5_al04_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T3_1_al04_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T3_1/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T3_1/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.4 imbalance_power=0.5 output.exp_name="mcpdg_phm-spur_T3_1_al04_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T3_1_al04_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T3_1_al04_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T3_5_al04_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T3_5/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T3_5/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.4 imbalance_power=0.5 output.exp_name="mcpdg_phm-spur_T3_5_al04_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T3_5_al04_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T3_5_al04_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T5_1_al04_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T5_1/train.h5" data.test_h5="/root/data/pu/pu_T5_1/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.4 imbalance_power=0.5 output.exp_name="mcpdg_pu_T5_1_al04_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T5_1_al04_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T5_1_al04_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T5_5_al04_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T5_5/train.h5" data.test_h5="/root/data/pu/pu_T5_5/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.4 imbalance_power=0.5 output.exp_name="mcpdg_pu_T5_5_al04_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T5_5_al04_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T5_5_al04_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T8_1_al04_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T8_1/train.h5" data.test_h5="/root/data/pu/pu_T8_1/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.4 imbalance_power=0.5 output.exp_name="mcpdg_pu_T8_1_al04_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T8_1_al04_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T8_1_al04_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T8_5_al04_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T8_5/train.h5" data.test_h5="/root/data/pu/pu_T8_5/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.4 imbalance_power=0.5 output.exp_name="mcpdg_pu_T8_5_al04_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T8_5_al04_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T8_5_al04_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T1_1_al05_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T1_1/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T1_1/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.5 imbalance_power=0.5 output.exp_name="mcpdg_phm-spur_T1_1_al05_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T1_1_al05_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T1_1_al05_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T1_5_al05_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T1_5/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T1_5/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.5 imbalance_power=0.5 output.exp_name="mcpdg_phm-spur_T1_5_al05_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T1_5_al05_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T1_5_al05_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T3_1_al05_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T3_1/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T3_1/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.5 imbalance_power=0.5 output.exp_name="mcpdg_phm-spur_T3_1_al05_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T3_1_al05_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T3_1_al05_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T3_5_al05_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T3_5/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T3_5/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.5 imbalance_power=0.5 output.exp_name="mcpdg_phm-spur_T3_5_al05_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T3_5_al05_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T3_5_al05_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T5_1_al05_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T5_1/train.h5" data.test_h5="/root/data/pu/pu_T5_1/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.5 imbalance_power=0.5 output.exp_name="mcpdg_pu_T5_1_al05_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T5_1_al05_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T5_1_al05_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T5_5_al05_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T5_5/train.h5" data.test_h5="/root/data/pu/pu_T5_5/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.5 imbalance_power=0.5 output.exp_name="mcpdg_pu_T5_5_al05_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T5_5_al05_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T5_5_al05_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T8_1_al05_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T8_1/train.h5" data.test_h5="/root/data/pu/pu_T8_1/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.5 imbalance_power=0.5 output.exp_name="mcpdg_pu_T8_1_al05_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T8_1_al05_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T8_1_al05_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T8_5_al05_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T8_5/train.h5" data.test_h5="/root/data/pu/pu_T8_5/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.5 imbalance_power=0.5 output.exp_name="mcpdg_pu_T8_5_al05_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T8_5_al05_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T8_5_al05_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T1_1_al06_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T1_1/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T1_1/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.6 imbalance_power=0.5 output.exp_name="mcpdg_phm-spur_T1_1_al06_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T1_1_al06_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T1_1_al06_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T1_5_al06_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T1_5/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T1_5/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.6 imbalance_power=0.5 output.exp_name="mcpdg_phm-spur_T1_5_al06_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T1_5_al06_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T1_5_al06_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T3_1_al06_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T3_1/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T3_1/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.6 imbalance_power=0.5 output.exp_name="mcpdg_phm-spur_T3_1_al06_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T3_1_al06_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T3_1_al06_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T3_5_al06_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T3_5/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T3_5/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.6 imbalance_power=0.5 output.exp_name="mcpdg_phm-spur_T3_5_al06_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T3_5_al06_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T3_5_al06_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T5_1_al06_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T5_1/train.h5" data.test_h5="/root/data/pu/pu_T5_1/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.6 imbalance_power=0.5 output.exp_name="mcpdg_pu_T5_1_al06_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T5_1_al06_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T5_1_al06_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T5_5_al06_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T5_5/train.h5" data.test_h5="/root/data/pu/pu_T5_5/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.6 imbalance_power=0.5 output.exp_name="mcpdg_pu_T5_5_al06_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T5_5_al06_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T5_5_al06_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T8_1_al06_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T8_1/train.h5" data.test_h5="/root/data/pu/pu_T8_1/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.6 imbalance_power=0.5 output.exp_name="mcpdg_pu_T8_1_al06_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T8_1_al06_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T8_1_al06_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T8_5_al06_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T8_5/train.h5" data.test_h5="/root/data/pu/pu_T8_5/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.6 imbalance_power=0.5 output.exp_name="mcpdg_pu_T8_5_al06_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T8_5_al06_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T8_5_al06_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T1_1_al07_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T1_1/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T1_1/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.7 imbalance_power=0.5 output.exp_name="mcpdg_phm-spur_T1_1_al07_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T1_1_al07_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T1_1_al07_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T1_5_al07_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T1_5/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T1_5/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.7 imbalance_power=0.5 output.exp_name="mcpdg_phm-spur_T1_5_al07_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T1_5_al07_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T1_5_al07_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T3_1_al07_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T3_1/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T3_1/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.7 imbalance_power=0.5 output.exp_name="mcpdg_phm-spur_T3_1_al07_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T3_1_al07_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T3_1_al07_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T3_5_al07_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T3_5/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T3_5/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.7 imbalance_power=0.5 output.exp_name="mcpdg_phm-spur_T3_5_al07_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T3_5_al07_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T3_5_al07_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T5_1_al07_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T5_1/train.h5" data.test_h5="/root/data/pu/pu_T5_1/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.7 imbalance_power=0.5 output.exp_name="mcpdg_pu_T5_1_al07_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T5_1_al07_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T5_1_al07_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T5_5_al07_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T5_5/train.h5" data.test_h5="/root/data/pu/pu_T5_5/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.7 imbalance_power=0.5 output.exp_name="mcpdg_pu_T5_5_al07_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T5_5_al07_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T5_5_al07_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T8_1_al07_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T8_1/train.h5" data.test_h5="/root/data/pu/pu_T8_1/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.7 imbalance_power=0.5 output.exp_name="mcpdg_pu_T8_1_al07_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T8_1_al07_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T8_1_al07_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T8_5_al07_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T8_5/train.h5" data.test_h5="/root/data/pu/pu_T8_5/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.7 imbalance_power=0.5 output.exp_name="mcpdg_pu_T8_5_al07_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T8_5_al07_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T8_5_al07_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T1_1_al08_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T1_1/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T1_1/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.8 imbalance_power=0.5 output.exp_name="mcpdg_phm-spur_T1_1_al08_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T1_1_al08_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T1_1_al08_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T1_5_al08_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T1_5/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T1_5/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.8 imbalance_power=0.5 output.exp_name="mcpdg_phm-spur_T1_5_al08_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T1_5_al08_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T1_5_al08_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T3_1_al08_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T3_1/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T3_1/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.8 imbalance_power=0.5 output.exp_name="mcpdg_phm-spur_T3_1_al08_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T3_1_al08_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T3_1_al08_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T3_5_al08_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T3_5/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T3_5/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.8 imbalance_power=0.5 output.exp_name="mcpdg_phm-spur_T3_5_al08_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T3_5_al08_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T3_5_al08_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T5_1_al08_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T5_1/train.h5" data.test_h5="/root/data/pu/pu_T5_1/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.8 imbalance_power=0.5 output.exp_name="mcpdg_pu_T5_1_al08_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T5_1_al08_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T5_1_al08_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T5_5_al08_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T5_5/train.h5" data.test_h5="/root/data/pu/pu_T5_5/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.8 imbalance_power=0.5 output.exp_name="mcpdg_pu_T5_5_al08_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T5_5_al08_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T5_5_al08_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T8_1_al08_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T8_1/train.h5" data.test_h5="/root/data/pu/pu_T8_1/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.8 imbalance_power=0.5 output.exp_name="mcpdg_pu_T8_1_al08_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T8_1_al08_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T8_1_al08_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T8_5_al08_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T8_5/train.h5" data.test_h5="/root/data/pu/pu_T8_5/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.8 imbalance_power=0.5 output.exp_name="mcpdg_pu_T8_5_al08_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T8_5_al08_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T8_5_al08_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T1_1_al09_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T1_1/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T1_1/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.9 imbalance_power=0.5 output.exp_name="mcpdg_phm-spur_T1_1_al09_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T1_1_al09_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T1_1_al09_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T1_5_al09_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T1_5/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T1_5/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.9 imbalance_power=0.5 output.exp_name="mcpdg_phm-spur_T1_5_al09_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T1_5_al09_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T1_5_al09_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T3_1_al09_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T3_1/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T3_1/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.9 imbalance_power=0.5 output.exp_name="mcpdg_phm-spur_T3_1_al09_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T3_1_al09_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T3_1_al09_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T3_5_al09_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T3_5/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T3_5/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.9 imbalance_power=0.5 output.exp_name="mcpdg_phm-spur_T3_5_al09_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T3_5_al09_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T3_5_al09_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T5_1_al09_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T5_1/train.h5" data.test_h5="/root/data/pu/pu_T5_1/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.9 imbalance_power=0.5 output.exp_name="mcpdg_pu_T5_1_al09_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T5_1_al09_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T5_1_al09_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T5_5_al09_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T5_5/train.h5" data.test_h5="/root/data/pu/pu_T5_5/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.9 imbalance_power=0.5 output.exp_name="mcpdg_pu_T5_5_al09_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T5_5_al09_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T5_5_al09_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T8_1_al09_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T8_1/train.h5" data.test_h5="/root/data/pu/pu_T8_1/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.9 imbalance_power=0.5 output.exp_name="mcpdg_pu_T8_1_al09_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T8_1_al09_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T8_1_al09_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T8_5_al09_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T8_5/train.h5" data.test_h5="/root/data/pu/pu_T8_5/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.9 imbalance_power=0.5 output.exp_name="mcpdg_pu_T8_5_al09_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T8_5_al09_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T8_5_al09_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T1_1_al10_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T1_1/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T1_1/test.h5" data.dataset_name="phm" model.proto_residual_alpha=1.0 imbalance_power=0.5 output.exp_name="mcpdg_phm-spur_T1_1_al10_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T1_1_al10_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T1_1_al10_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T1_5_al10_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T1_5/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T1_5/test.h5" data.dataset_name="phm" model.proto_residual_alpha=1.0 imbalance_power=0.5 output.exp_name="mcpdg_phm-spur_T1_5_al10_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T1_5_al10_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T1_5_al10_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T3_1_al10_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T3_1/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T3_1/test.h5" data.dataset_name="phm" model.proto_residual_alpha=1.0 imbalance_power=0.5 output.exp_name="mcpdg_phm-spur_T3_1_al10_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T3_1_al10_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T3_1_al10_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T3_5_al10_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T3_5/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T3_5/test.h5" data.dataset_name="phm" model.proto_residual_alpha=1.0 imbalance_power=0.5 output.exp_name="mcpdg_phm-spur_T3_5_al10_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T3_5_al10_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T3_5_al10_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T5_1_al10_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T5_1/train.h5" data.test_h5="/root/data/pu/pu_T5_1/test.h5" data.dataset_name="pu" model.proto_residual_alpha=1.0 imbalance_power=0.5 output.exp_name="mcpdg_pu_T5_1_al10_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T5_1_al10_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T5_1_al10_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T5_5_al10_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T5_5/train.h5" data.test_h5="/root/data/pu/pu_T5_5/test.h5" data.dataset_name="pu" model.proto_residual_alpha=1.0 imbalance_power=0.5 output.exp_name="mcpdg_pu_T5_5_al10_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T5_5_al10_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T5_5_al10_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T8_1_al10_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T8_1/train.h5" data.test_h5="/root/data/pu/pu_T8_1/test.h5" data.dataset_name="pu" model.proto_residual_alpha=1.0 imbalance_power=0.5 output.exp_name="mcpdg_pu_T8_1_al10_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T8_1_al10_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T8_1_al10_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T8_5_al10_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T8_5/train.h5" data.test_h5="/root/data/pu/pu_T8_5/test.h5" data.dataset_name="pu" model.proto_residual_alpha=1.0 imbalance_power=0.5 output.exp_name="mcpdg_pu_T8_5_al10_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T8_5_al10_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T8_5_al10_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

# ==================== 第二组实验：alpha=0.5 ====================
echo -e "\n==================== 第二组实验：alpha=0.5 ====================\n"

echo "实验（mcpdg_phm-spur_T1_1_al05_ip00）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T1_1/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T1_1/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.5 imbalance_power=0.0 output.exp_name="mcpdg_phm-spur_T1_1_al05_ip00" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T1_1_al05_ip00 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T1_1_al05_ip00】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T1_5_al05_ip00）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T1_5/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T1_5/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.5 imbalance_power=0.0 output.exp_name="mcpdg_phm-spur_T1_5_al05_ip00" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T1_5_al05_ip00 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T1_5_al05_ip00】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T3_1_al05_ip00）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T3_1/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T3_1/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.5 imbalance_power=0.0 output.exp_name="mcpdg_phm-spur_T3_1_al05_ip00" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T3_1_al05_ip00 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T3_1_al05_ip00】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T3_5_al05_ip00）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T3_5/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T3_5/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.5 imbalance_power=0.0 output.exp_name="mcpdg_phm-spur_T3_5_al05_ip00" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T3_5_al05_ip00 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T3_5_al05_ip00】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T5_1_al05_ip00）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T5_1/train.h5" data.test_h5="/root/data/pu/pu_T5_1/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.5 imbalance_power=0.0 output.exp_name="mcpdg_pu_T5_1_al05_ip00" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T5_1_al05_ip00 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T5_1_al05_ip00】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T5_5_al05_ip00）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T5_5/train.h5" data.test_h5="/root/data/pu/pu_T5_5/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.5 imbalance_power=0.0 output.exp_name="mcpdg_pu_T5_5_al05_ip00" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T5_5_al05_ip00 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T5_5_al05_ip00】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T8_1_al05_ip00）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T8_1/train.h5" data.test_h5="/root/data/pu/pu_T8_1/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.5 imbalance_power=0.0 output.exp_name="mcpdg_pu_T8_1_al05_ip00" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T8_1_al05_ip00 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T8_1_al05_ip00】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T8_5_al05_ip00）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T8_5/train.h5" data.test_h5="/root/data/pu/pu_T8_5/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.5 imbalance_power=0.0 output.exp_name="mcpdg_pu_T8_5_al05_ip00" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T8_5_al05_ip00 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T8_5_al05_ip00】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T1_1_al05_ip01）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T1_1/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T1_1/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.5 imbalance_power=0.1 output.exp_name="mcpdg_phm-spur_T1_1_al05_ip01" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T1_1_al05_ip01 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T1_1_al05_ip01】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T1_5_al05_ip01）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T1_5/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T1_5/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.5 imbalance_power=0.1 output.exp_name="mcpdg_phm-spur_T1_5_al05_ip01" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T1_5_al05_ip01 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T1_5_al05_ip01】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T3_1_al05_ip01）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T3_1/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T3_1/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.5 imbalance_power=0.1 output.exp_name="mcpdg_phm-spur_T3_1_al05_ip01" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T3_1_al05_ip01 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T3_1_al05_ip01】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T3_5_al05_ip01）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T3_5/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T3_5/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.5 imbalance_power=0.1 output.exp_name="mcpdg_phm-spur_T3_5_al05_ip01" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T3_5_al05_ip01 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T3_5_al05_ip01】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T5_1_al05_ip01）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T5_1/train.h5" data.test_h5="/root/data/pu/pu_T5_1/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.5 imbalance_power=0.1 output.exp_name="mcpdg_pu_T5_1_al05_ip01" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T5_1_al05_ip01 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T5_1_al05_ip01】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T5_5_al05_ip01）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T5_5/train.h5" data.test_h5="/root/data/pu/pu_T5_5/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.5 imbalance_power=0.1 output.exp_name="mcpdg_pu_T5_5_al05_ip01" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T5_5_al05_ip01 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T5_5_al05_ip01】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T8_1_al05_ip01）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T8_1/train.h5" data.test_h5="/root/data/pu/pu_T8_1/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.5 imbalance_power=0.1 output.exp_name="mcpdg_pu_T8_1_al05_ip01" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T8_1_al05_ip01 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T8_1_al05_ip01】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T8_5_al05_ip01）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T8_5/train.h5" data.test_h5="/root/data/pu/pu_T8_5/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.5 imbalance_power=0.1 output.exp_name="mcpdg_pu_T8_5_al05_ip01" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T8_5_al05_ip01 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T8_5_al05_ip01】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T1_1_al05_ip02）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T1_1/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T1_1/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.5 imbalance_power=0.2 output.exp_name="mcpdg_phm-spur_T1_1_al05_ip02" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T1_1_al05_ip02 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T1_1_al05_ip02】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T1_5_al05_ip02）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T1_5/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T1_5/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.5 imbalance_power=0.2 output.exp_name="mcpdg_phm-spur_T1_5_al05_ip02" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T1_5_al05_ip02 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T1_5_al05_ip02】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T3_1_al05_ip02）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T3_1/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T3_1/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.5 imbalance_power=0.2 output.exp_name="mcpdg_phm-spur_T3_1_al05_ip02" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T3_1_al05_ip02 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T3_1_al05_ip02】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T3_5_al05_ip02）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T3_5/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T3_5/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.5 imbalance_power=0.2 output.exp_name="mcpdg_phm-spur_T3_5_al05_ip02" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T3_5_al05_ip02 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T3_5_al05_ip02】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T5_1_al05_ip02）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T5_1/train.h5" data.test_h5="/root/data/pu/pu_T5_1/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.5 imbalance_power=0.2 output.exp_name="mcpdg_pu_T5_1_al05_ip02" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T5_1_al05_ip02 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T5_1_al05_ip02】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T5_5_al05_ip02）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T5_5/train.h5" data.test_h5="/root/data/pu/pu_T5_5/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.5 imbalance_power=0.2 output.exp_name="mcpdg_pu_T5_5_al05_ip02" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T5_5_al05_ip02 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T5_5_al05_ip02】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T8_1_al05_ip02）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T8_1/train.h5" data.test_h5="/root/data/pu/pu_T8_1/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.5 imbalance_power=0.2 output.exp_name="mcpdg_pu_T8_1_al05_ip02" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T8_1_al05_ip02 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T8_1_al05_ip02】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T8_5_al05_ip02）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T8_5/train.h5" data.test_h5="/root/data/pu/pu_T8_5/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.5 imbalance_power=0.2 output.exp_name="mcpdg_pu_T8_5_al05_ip02" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T8_5_al05_ip02 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T8_5_al05_ip02】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T1_1_al05_ip03）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T1_1/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T1_1/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.5 imbalance_power=0.3 output.exp_name="mcpdg_phm-spur_T1_1_al05_ip03" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T1_1_al05_ip03 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T1_1_al05_ip03】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T1_5_al05_ip03）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T1_5/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T1_5/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.5 imbalance_power=0.3 output.exp_name="mcpdg_phm-spur_T1_5_al05_ip03" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T1_5_al05_ip03 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T1_5_al05_ip03】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T3_1_al05_ip03）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T3_1/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T3_1/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.5 imbalance_power=0.3 output.exp_name="mcpdg_phm-spur_T3_1_al05_ip03" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T3_1_al05_ip03 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T3_1_al05_ip03】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T3_5_al05_ip03）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T3_5/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T3_5/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.5 imbalance_power=0.3 output.exp_name="mcpdg_phm-spur_T3_5_al05_ip03" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T3_5_al05_ip03 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T3_5_al05_ip03】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T5_1_al05_ip03）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T5_1/train.h5" data.test_h5="/root/data/pu/pu_T5_1/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.5 imbalance_power=0.3 output.exp_name="mcpdg_pu_T5_1_al05_ip03" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T5_1_al05_ip03 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T5_1_al05_ip03】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T5_5_al05_ip03）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T5_5/train.h5" data.test_h5="/root/data/pu/pu_T5_5/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.5 imbalance_power=0.3 output.exp_name="mcpdg_pu_T5_5_al05_ip03" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T5_5_al05_ip03 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T5_5_al05_ip03】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T8_1_al05_ip03）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T8_1/train.h5" data.test_h5="/root/data/pu/pu_T8_1/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.5 imbalance_power=0.3 output.exp_name="mcpdg_pu_T8_1_al05_ip03" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T8_1_al05_ip03 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T8_1_al05_ip03】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T8_5_al05_ip03）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T8_5/train.h5" data.test_h5="/root/data/pu/pu_T8_5/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.5 imbalance_power=0.3 output.exp_name="mcpdg_pu_T8_5_al05_ip03" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T8_5_al05_ip03 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T8_5_al05_ip03】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T1_1_al05_ip04）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T1_1/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T1_1/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.5 imbalance_power=0.4 output.exp_name="mcpdg_phm-spur_T1_1_al05_ip04" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T1_1_al05_ip04 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T1_1_al05_ip04】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T1_5_al05_ip04）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T1_5/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T1_5/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.5 imbalance_power=0.4 output.exp_name="mcpdg_phm-spur_T1_5_al05_ip04" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T1_5_al05_ip04 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T1_5_al05_ip04】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T3_1_al05_ip04）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T3_1/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T3_1/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.5 imbalance_power=0.4 output.exp_name="mcpdg_phm-spur_T3_1_al05_ip04" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T3_1_al05_ip04 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T3_1_al05_ip04】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T3_5_al05_ip04）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T3_5/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T3_5/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.5 imbalance_power=0.4 output.exp_name="mcpdg_phm-spur_T3_5_al05_ip04" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T3_5_al05_ip04 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T3_5_al05_ip04】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T5_1_al05_ip04）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T5_1/train.h5" data.test_h5="/root/data/pu/pu_T5_1/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.5 imbalance_power=0.4 output.exp_name="mcpdg_pu_T5_1_al05_ip04" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T5_1_al05_ip04 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T5_1_al05_ip04】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T5_5_al05_ip04）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T5_5/train.h5" data.test_h5="/root/data/pu/pu_T5_5/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.5 imbalance_power=0.4 output.exp_name="mcpdg_pu_T5_5_al05_ip04" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T5_5_al05_ip04 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T5_5_al05_ip04】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T8_1_al05_ip04）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T8_1/train.h5" data.test_h5="/root/data/pu/pu_T8_1/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.5 imbalance_power=0.4 output.exp_name="mcpdg_pu_T8_1_al05_ip04" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T8_1_al05_ip04 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T8_1_al05_ip04】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T8_5_al05_ip04）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T8_5/train.h5" data.test_h5="/root/data/pu/pu_T8_5/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.5 imbalance_power=0.4 output.exp_name="mcpdg_pu_T8_5_al05_ip04" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T8_5_al05_ip04 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T8_5_al05_ip04】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T1_1_al05_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T1_1/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T1_1/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.5 imbalance_power=0.5 output.exp_name="mcpdg_phm-spur_T1_1_al05_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T1_1_al05_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T1_1_al05_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T1_5_al05_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T1_5/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T1_5/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.5 imbalance_power=0.5 output.exp_name="mcpdg_phm-spur_T1_5_al05_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T1_5_al05_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T1_5_al05_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T3_1_al05_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T3_1/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T3_1/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.5 imbalance_power=0.5 output.exp_name="mcpdg_phm-spur_T3_1_al05_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T3_1_al05_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T3_1_al05_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T3_5_al05_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T3_5/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T3_5/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.5 imbalance_power=0.5 output.exp_name="mcpdg_phm-spur_T3_5_al05_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T3_5_al05_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T3_5_al05_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T5_1_al05_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T5_1/train.h5" data.test_h5="/root/data/pu/pu_T5_1/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.5 imbalance_power=0.5 output.exp_name="mcpdg_pu_T5_1_al05_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T5_1_al05_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T5_1_al05_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T5_5_al05_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T5_5/train.h5" data.test_h5="/root/data/pu/pu_T5_5/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.5 imbalance_power=0.5 output.exp_name="mcpdg_pu_T5_5_al05_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T5_5_al05_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T5_5_al05_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T8_1_al05_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T8_1/train.h5" data.test_h5="/root/data/pu/pu_T8_1/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.5 imbalance_power=0.5 output.exp_name="mcpdg_pu_T8_1_al05_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T8_1_al05_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T8_1_al05_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T8_5_al05_ip05）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T8_5/train.h5" data.test_h5="/root/data/pu/pu_T8_5/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.5 imbalance_power=0.5 output.exp_name="mcpdg_pu_T8_5_al05_ip05" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T8_5_al05_ip05 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T8_5_al05_ip05】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T1_1_al05_ip06）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T1_1/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T1_1/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.5 imbalance_power=0.6 output.exp_name="mcpdg_phm-spur_T1_1_al05_ip06" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T1_1_al05_ip06 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T1_1_al05_ip06】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T1_5_al05_ip06）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T1_5/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T1_5/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.5 imbalance_power=0.6 output.exp_name="mcpdg_phm-spur_T1_5_al05_ip06" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T1_5_al05_ip06 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T1_5_al05_ip06】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T3_1_al05_ip06）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T3_1/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T3_1/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.5 imbalance_power=0.6 output.exp_name="mcpdg_phm-spur_T3_1_al05_ip06" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T3_1_al05_ip06 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T3_1_al05_ip06】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T3_5_al05_ip06）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T3_5/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T3_5/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.5 imbalance_power=0.6 output.exp_name="mcpdg_phm-spur_T3_5_al05_ip06" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T3_5_al05_ip06 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T3_5_al05_ip06】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T5_1_al05_ip06）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T5_1/train.h5" data.test_h5="/root/data/pu/pu_T5_1/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.5 imbalance_power=0.6 output.exp_name="mcpdg_pu_T5_1_al05_ip06" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T5_1_al05_ip06 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T5_1_al05_ip06】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T5_5_al05_ip06）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T5_5/train.h5" data.test_h5="/root/data/pu/pu_T5_5/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.5 imbalance_power=0.6 output.exp_name="mcpdg_pu_T5_5_al05_ip06" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T5_5_al05_ip06 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T5_5_al05_ip06】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T8_1_al05_ip06）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T8_1/train.h5" data.test_h5="/root/data/pu/pu_T8_1/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.5 imbalance_power=0.6 output.exp_name="mcpdg_pu_T8_1_al05_ip06" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T8_1_al05_ip06 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T8_1_al05_ip06】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T8_5_al05_ip06）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T8_5/train.h5" data.test_h5="/root/data/pu/pu_T8_5/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.5 imbalance_power=0.6 output.exp_name="mcpdg_pu_T8_5_al05_ip06" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T8_5_al05_ip06 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T8_5_al05_ip06】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T1_1_al05_ip07）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T1_1/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T1_1/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.5 imbalance_power=0.7 output.exp_name="mcpdg_phm-spur_T1_1_al05_ip07" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T1_1_al05_ip07 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T1_1_al05_ip07】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T1_5_al05_ip07）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T1_5/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T1_5/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.5 imbalance_power=0.7 output.exp_name="mcpdg_phm-spur_T1_5_al05_ip07" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T1_5_al05_ip07 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T1_5_al05_ip07】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T3_1_al05_ip07）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T3_1/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T3_1/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.5 imbalance_power=0.7 output.exp_name="mcpdg_phm-spur_T3_1_al05_ip07" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T3_1_al05_ip07 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T3_1_al05_ip07】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T3_5_al05_ip07）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T3_5/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T3_5/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.5 imbalance_power=0.7 output.exp_name="mcpdg_phm-spur_T3_5_al05_ip07" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T3_5_al05_ip07 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T3_5_al05_ip07】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T5_1_al05_ip07）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T5_1/train.h5" data.test_h5="/root/data/pu/pu_T5_1/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.5 imbalance_power=0.7 output.exp_name="mcpdg_pu_T5_1_al05_ip07" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T5_1_al05_ip07 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T5_1_al05_ip07】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T5_5_al05_ip07）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T5_5/train.h5" data.test_h5="/root/data/pu/pu_T5_5/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.5 imbalance_power=0.7 output.exp_name="mcpdg_pu_T5_5_al05_ip07" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T5_5_al05_ip07 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T5_5_al05_ip07】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T8_1_al05_ip07）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T8_1/train.h5" data.test_h5="/root/data/pu/pu_T8_1/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.5 imbalance_power=0.7 output.exp_name="mcpdg_pu_T8_1_al05_ip07" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T8_1_al05_ip07 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T8_1_al05_ip07】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T8_5_al05_ip07）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T8_5/train.h5" data.test_h5="/root/data/pu/pu_T8_5/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.5 imbalance_power=0.7 output.exp_name="mcpdg_pu_T8_5_al05_ip07" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T8_5_al05_ip07 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T8_5_al05_ip07】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T1_1_al05_ip08）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T1_1/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T1_1/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.5 imbalance_power=0.8 output.exp_name="mcpdg_phm-spur_T1_1_al05_ip08" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T1_1_al05_ip08 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T1_1_al05_ip08】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T1_5_al05_ip08）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T1_5/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T1_5/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.5 imbalance_power=0.8 output.exp_name="mcpdg_phm-spur_T1_5_al05_ip08" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T1_5_al05_ip08 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T1_5_al05_ip08】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T3_1_al05_ip08）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T3_1/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T3_1/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.5 imbalance_power=0.8 output.exp_name="mcpdg_phm-spur_T3_1_al05_ip08" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T3_1_al05_ip08 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T3_1_al05_ip08】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T3_5_al05_ip08）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T3_5/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T3_5/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.5 imbalance_power=0.8 output.exp_name="mcpdg_phm-spur_T3_5_al05_ip08" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T3_5_al05_ip08 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T3_5_al05_ip08】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T5_1_al05_ip08）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T5_1/train.h5" data.test_h5="/root/data/pu/pu_T5_1/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.5 imbalance_power=0.8 output.exp_name="mcpdg_pu_T5_1_al05_ip08" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T5_1_al05_ip08 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T5_1_al05_ip08】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T5_5_al05_ip08）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T5_5/train.h5" data.test_h5="/root/data/pu/pu_T5_5/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.5 imbalance_power=0.8 output.exp_name="mcpdg_pu_T5_5_al05_ip08" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T5_5_al05_ip08 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T5_5_al05_ip08】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T8_1_al05_ip08）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T8_1/train.h5" data.test_h5="/root/data/pu/pu_T8_1/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.5 imbalance_power=0.8 output.exp_name="mcpdg_pu_T8_1_al05_ip08" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T8_1_al05_ip08 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T8_1_al05_ip08】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T8_5_al05_ip08）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T8_5/train.h5" data.test_h5="/root/data/pu/pu_T8_5/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.5 imbalance_power=0.8 output.exp_name="mcpdg_pu_T8_5_al05_ip08" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T8_5_al05_ip08 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T8_5_al05_ip08】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T1_1_al05_ip09）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T1_1/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T1_1/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.5 imbalance_power=0.9 output.exp_name="mcpdg_phm-spur_T1_1_al05_ip09" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T1_1_al05_ip09 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T1_1_al05_ip09】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T1_5_al05_ip09）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T1_5/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T1_5/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.5 imbalance_power=0.9 output.exp_name="mcpdg_phm-spur_T1_5_al05_ip09" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T1_5_al05_ip09 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T1_5_al05_ip09】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T3_1_al05_ip09）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T3_1/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T3_1/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.5 imbalance_power=0.9 output.exp_name="mcpdg_phm-spur_T3_1_al05_ip09" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T3_1_al05_ip09 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T3_1_al05_ip09】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T3_5_al05_ip09）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T3_5/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T3_5/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.5 imbalance_power=0.9 output.exp_name="mcpdg_phm-spur_T3_5_al05_ip09" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T3_5_al05_ip09 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T3_5_al05_ip09】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T5_1_al05_ip09）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T5_1/train.h5" data.test_h5="/root/data/pu/pu_T5_1/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.5 imbalance_power=0.9 output.exp_name="mcpdg_pu_T5_1_al05_ip09" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T5_1_al05_ip09 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T5_1_al05_ip09】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T5_5_al05_ip09）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T5_5/train.h5" data.test_h5="/root/data/pu/pu_T5_5/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.5 imbalance_power=0.9 output.exp_name="mcpdg_pu_T5_5_al05_ip09" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T5_5_al05_ip09 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T5_5_al05_ip09】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T8_1_al05_ip09）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T8_1/train.h5" data.test_h5="/root/data/pu/pu_T8_1/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.5 imbalance_power=0.9 output.exp_name="mcpdg_pu_T8_1_al05_ip09" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T8_1_al05_ip09 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T8_1_al05_ip09】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T8_5_al05_ip09）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T8_5/train.h5" data.test_h5="/root/data/pu/pu_T8_5/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.5 imbalance_power=0.9 output.exp_name="mcpdg_pu_T8_5_al05_ip09" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T8_5_al05_ip09 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T8_5_al05_ip09】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T1_1_al05_ip10）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T1_1/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T1_1/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.5 imbalance_power=1.0 output.exp_name="mcpdg_phm-spur_T1_1_al05_ip10" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T1_1_al05_ip10 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T1_1_al05_ip10】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T1_5_al05_ip10）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T1_5/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T1_5/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.5 imbalance_power=1.0 output.exp_name="mcpdg_phm-spur_T1_5_al05_ip10" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T1_5_al05_ip10 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T1_5_al05_ip10】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T3_1_al05_ip10）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T3_1/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T3_1/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.5 imbalance_power=1.0 output.exp_name="mcpdg_phm-spur_T3_1_al05_ip10" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T3_1_al05_ip10 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T3_1_al05_ip10】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_phm-spur_T3_5_al05_ip10）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/phm-spur/phm-spur_T3_5/train.h5" data.test_h5="/root/data/phm-spur/phm-spur_T3_5/test.h5" data.dataset_name="phm" model.proto_residual_alpha=0.5 imbalance_power=1.0 output.exp_name="mcpdg_phm-spur_T3_5_al05_ip10" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_phm-spur_T3_5_al05_ip10 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_phm-spur_T3_5_al05_ip10】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T5_1_al05_ip10）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T5_1/train.h5" data.test_h5="/root/data/pu/pu_T5_1/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.5 imbalance_power=1.0 output.exp_name="mcpdg_pu_T5_1_al05_ip10" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T5_1_al05_ip10 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T5_1_al05_ip10】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T5_5_al05_ip10）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T5_5/train.h5" data.test_h5="/root/data/pu/pu_T5_5/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.5 imbalance_power=1.0 output.exp_name="mcpdg_pu_T5_5_al05_ip10" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T5_5_al05_ip10 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T5_5_al05_ip10】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T8_1_al05_ip10）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T8_1/train.h5" data.test_h5="/root/data/pu/pu_T8_1/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.5 imbalance_power=1.0 output.exp_name="mcpdg_pu_T8_1_al05_ip10" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T8_1_al05_ip10 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T8_1_al05_ip10】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "实验（mcpdg_pu_T8_5_al05_ip10）准备开始"
python -m src.main --configs src/configs/mcpdg.yaml data.train_h5="/root/data/pu/pu_T8_5/train.h5" data.test_h5="/root/data/pu/pu_T8_5/test.h5" data.dataset_name="pu" model.proto_residual_alpha=0.5 imbalance_power=1.0 output.exp_name="mcpdg_pu_T8_5_al05_ip10" || { echo "[`date +%Y-%m-%d\ %H:%M:%S`] mcpdg_pu_T8_5_al05_ip10 执行失败" >> $ERROR_LOG; echo "==== 实验【mcpdg_pu_T8_5_al05_ip10】运行异常，跳过当前任务，继续执行下一组 ===="; }

echo "======== 所有实验任务执行完毕，错误日志见：$ERROR_LOG，准备关机 ========"
shutdown -h now
