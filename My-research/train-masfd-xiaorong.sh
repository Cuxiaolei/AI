#!/bin/bash
# 批量执行指令的脚本
set -e  # 关键：任意指令失败则脚本立即退出（可选，增强安全性）

echo masfdE1 模型
python -m src.main --configs src/configs/masfdE1/masfdE1_phm_T1_5-1.yaml
python -m src.main --configs src/configs/masfdE1/masfdE1_phm_T1_5-2.yaml
python -m src.main --configs src/configs/masfdE1/masfdE1_phm_T1_5-3.yaml
python -m src.main --configs src/configs/masfdE1/masfdE1_phm_T1_5-4.yaml
python -m src.main --configs src/configs/masfdE1/masfdE1_phm_T1_5-5.yaml
python -m src.main --configs src/configs/masfdE1/masfdE1_phm_T2_5-1.yaml
python -m src.main --configs src/configs/masfdE1/masfdE1_phm_T2_5-2.yaml
python -m src.main --configs src/configs/masfdE1/masfdE1_phm_T2_5-3.yaml
python -m src.main --configs src/configs/masfdE1/masfdE1_phm_T2_5-4.yaml
python -m src.main --configs src/configs/masfdE1/masfdE1_phm_T2_5-5.yaml
python -m src.main --configs src/configs/masfdE1/masfdE1_phm_T3_5-1.yaml
python -m src.main --configs src/configs/masfdE1/masfdE1_phm_T3_5-2.yaml
python -m src.main --configs src/configs/masfdE1/masfdE1_phm_T3_5-3.yaml
python -m src.main --configs src/configs/masfdE1/masfdE1_phm_T3_5-4.yaml
python -m src.main --configs src/configs/masfdE1/masfdE1_phm_T3_5-5.yaml
python -m src.main --configs src/configs/masfdE1/masfdE1_phm_T4_5-1.yaml
python -m src.main --configs src/configs/masfdE1/masfdE1_phm_T4_5-2.yaml
python -m src.main --configs src/configs/masfdE1/masfdE1_phm_T4_5-3.yaml
python -m src.main --configs src/configs/masfdE1/masfdE1_phm_T4_5-4.yaml
python -m src.main --configs src/configs/masfdE1/masfdE1_phm_T4_5-5.yaml

echo masfdE2 模型
python -m src.main --configs src/configs/masfdE2/masfdE2_phm_T1_5-1.yaml
python -m src.main --configs src/configs/masfdE2/masfdE2_phm_T1_5-2.yaml
python -m src.main --configs src/configs/masfdE2/masfdE2_phm_T1_5-3.yaml
python -m src.main --configs src/configs/masfdE2/masfdE2_phm_T1_5-4.yaml
python -m src.main --configs src/configs/masfdE2/masfdE2_phm_T1_5-5.yaml
python -m src.main --configs src/configs/masfdE2/masfdE2_phm_T2_5-1.yaml
python -m src.main --configs src/configs/masfdE2/masfdE2_phm_T2_5-2.yaml
python -m src.main --configs src/configs/masfdE2/masfdE2_phm_T2_5-3.yaml
python -m src.main --configs src/configs/masfdE2/masfdE2_phm_T2_5-4.yaml
python -m src.main --configs src/configs/masfdE2/masfdE2_phm_T2_5-5.yaml
python -m src.main --configs src/configs/masfdE2/masfdE2_phm_T3_5-1.yaml
python -m src.main --configs src/configs/masfdE2/masfdE2_phm_T3_5-2.yaml
python -m src.main --configs src/configs/masfdE2/masfdE2_phm_T3_5-3.yaml
python -m src.main --configs src/configs/masfdE2/masfdE2_phm_T3_5-4.yaml
python -m src.main --configs src/configs/masfdE2/masfdE2_phm_T3_5-5.yaml
python -m src.main --configs src/configs/masfdE2/masfdE2_phm_T4_5-1.yaml
python -m src.main --configs src/configs/masfdE2/masfdE2_phm_T4_5-2.yaml
python -m src.main --configs src/configs/masfdE2/masfdE2_phm_T4_5-3.yaml
python -m src.main --configs src/configs/masfdE2/masfdE2_phm_T4_5-4.yaml
python -m src.main --configs src/configs/masfdE2/masfdE2_phm_T4_5-5.yaml

echo masfdE3 模型
python -m src.main --configs src/configs/masfdE3/masfdE3_phm_T1_5-1.yaml
python -m src.main --configs src/configs/masfdE3/masfdE3_phm_T1_5-2.yaml
python -m src.main --configs src/configs/masfdE3/masfdE3_phm_T1_5-3.yaml
python -m src.main --configs src/configs/masfdE3/masfdE3_phm_T1_5-4.yaml
python -m src.main --configs src/configs/masfdE3/masfdE3_phm_T1_5-5.yaml
python -m src.main --configs src/configs/masfdE3/masfdE3_phm_T2_5-1.yaml
python -m src.main --configs src/configs/masfdE3/masfdE3_phm_T2_5-2.yaml
python -m src.main --configs src/configs/masfdE3/masfdE3_phm_T2_5-3.yaml
python -m src.main --configs src/configs/masfdE3/masfdE3_phm_T2_5-4.yaml
python -m src.main --configs src/configs/masfdE3/masfdE3_phm_T2_5-5.yaml
python -m src.main --configs src/configs/masfdE3/masfdE3_phm_T3_5-1.yaml
python -m src.main --configs src/configs/masfdE3/masfdE3_phm_T3_5-2.yaml
python -m src.main --configs src/configs/masfdE3/masfdE3_phm_T3_5-3.yaml
python -m src.main --configs src/configs/masfdE3/masfdE3_phm_T3_5-4.yaml
python -m src.main --configs src/configs/masfdE3/masfdE3_phm_T3_5-5.yaml
python -m src.main --configs src/configs/masfdE3/masfdE3_phm_T4_5-1.yaml
python -m src.main --configs src/configs/masfdE3/masfdE3_phm_T4_5-2.yaml
python -m src.main --configs src/configs/masfdE3/masfdE3_phm_T4_5-3.yaml
python -m src.main --configs src/configs/masfdE3/masfdE3_phm_T4_5-4.yaml
python -m src.main --configs src/configs/masfdE3/masfdE3_phm_T4_5-5.yaml

echo masfdE4 模型
python -m src.main --configs src/configs/masfdE4/masfdE4_phm_T1_5-1.yaml
python -m src.main --configs src/configs/masfdE4/masfdE4_phm_T1_5-2.yaml
python -m src.main --configs src/configs/masfdE4/masfdE4_phm_T1_5-3.yaml
python -m src.main --configs src/configs/masfdE4/masfdE4_phm_T1_5-4.yaml
python -m src.main --configs src/configs/masfdE4/masfdE4_phm_T1_5-5.yaml
python -m src.main --configs src/configs/masfdE4/masfdE4_phm_T2_5-1.yaml
python -m src.main --configs src/configs/masfdE4/masfdE4_phm_T2_5-2.yaml
python -m src.main --configs src/configs/masfdE4/masfdE4_phm_T2_5-3.yaml
python -m src.main --configs src/configs/masfdE4/masfdE4_phm_T2_5-4.yaml
python -m src.main --configs src/configs/masfdE4/masfdE4_phm_T2_5-5.yaml
python -m src.main --configs src/configs/masfdE4/masfdE4_phm_T3_5-1.yaml
python -m src.main --configs src/configs/masfdE4/masfdE4_phm_T3_5-2.yaml
python -m src.main --configs src/configs/masfdE4/masfdE4_phm_T3_5-3.yaml
python -m src.main --configs src/configs/masfdE4/masfdE4_phm_T3_5-4.yaml
python -m src.main --configs src/configs/masfdE4/masfdE4_phm_T3_5-5.yaml
python -m src.main --configs src/configs/masfdE4/masfdE4_phm_T4_5-1.yaml
python -m src.main --configs src/configs/masfdE4/masfdE4_phm_T4_5-2.yaml
python -m src.main --configs src/configs/masfdE4/masfdE4_phm_T4_5-3.yaml
python -m src.main --configs src/configs/masfdE4/masfdE4_phm_T4_5-4.yaml
python -m src.main --configs src/configs/masfdE4/masfdE4_phm_T4_5-5.yaml

echo masfdE5 模型
python -m src.main --configs src/configs/masfdE5/masfdE5_phm_T1_5-1.yaml
python -m src.main --configs src/configs/masfdE5/masfdE5_phm_T1_5-2.yaml
python -m src.main --configs src/configs/masfdE5/masfdE5_phm_T1_5-3.yaml
python -m src.main --configs src/configs/masfdE5/masfdE5_phm_T1_5-4.yaml
python -m src.main --configs src/configs/masfdE5/masfdE5_phm_T1_5-5.yaml
python -m src.main --configs src/configs/masfdE5/masfdE5_phm_T2_5-1.yaml
python -m src.main --configs src/configs/masfdE5/masfdE5_phm_T2_5-2.yaml
python -m src.main --configs src/configs/masfdE5/masfdE5_phm_T2_5-3.yaml
python -m src.main --configs src/configs/masfdE5/masfdE5_phm_T2_5-4.yaml
python -m src.main --configs src/configs/masfdE5/masfdE5_phm_T2_5-5.yaml
python -m src.main --configs src/configs/masfdE5/masfdE5_phm_T3_5-1.yaml
python -m src.main --configs src/configs/masfdE5/masfdE5_phm_T3_5-2.yaml
python -m src.main --configs src/configs/masfdE5/masfdE5_phm_T3_5-3.yaml
python -m src.main --configs src/configs/masfdE5/masfdE5_phm_T3_5-4.yaml
python -m src.main --configs src/configs/masfdE5/masfdE5_phm_T3_5-5.yaml
python -m src.main --configs src/configs/masfdE5/masfdE5_phm_T4_5-1.yaml
python -m src.main --configs src/configs/masfdE5/masfdE5_phm_T4_5-2.yaml
python -m src.main --configs src/configs/masfdE5/masfdE5_phm_T4_5-3.yaml
python -m src.main --configs src/configs/masfdE5/masfdE5_phm_T4_5-4.yaml
python -m src.main --configs src/configs/masfdE5/masfdE5_phm_T4_5-5.yaml

echo masfdE6 模型
python -m src.main --configs src/configs/masfdE6/masfdE6_phm_T1_5-1.yaml
python -m src.main --configs src/configs/masfdE6/masfdE6_phm_T1_5-2.yaml
python -m src.main --configs src/configs/masfdE6/masfdE6_phm_T1_5-3.yaml
python -m src.main --configs src/configs/masfdE6/masfdE6_phm_T1_5-4.yaml
python -m src.main --configs src/configs/masfdE6/masfdE6_phm_T1_5-5.yaml
python -m src.main --configs src/configs/masfdE6/masfdE6_phm_T2_5-1.yaml
python -m src.main --configs src/configs/masfdE6/masfdE6_phm_T2_5-2.yaml
python -m src.main --configs src/configs/masfdE6/masfdE6_phm_T2_5-3.yaml
python -m src.main --configs src/configs/masfdE6/masfdE6_phm_T2_5-4.yaml
python -m src.main --configs src/configs/masfdE6/masfdE6_phm_T2_5-5.yaml
python -m src.main --configs src/configs/masfdE6/masfdE6_phm_T3_5-1.yaml
python -m src.main --configs src/configs/masfdE6/masfdE6_phm_T3_5-2.yaml
python -m src.main --configs src/configs/masfdE6/masfdE6_phm_T3_5-3.yaml
python -m src.main --configs src/configs/masfdE6/masfdE6_phm_T3_5-4.yaml
python -m src.main --configs src/configs/masfdE6/masfdE6_phm_T3_5-5.yaml
python -m src.main --configs src/configs/masfdE6/masfdE6_phm_T4_5-1.yaml
python -m src.main --configs src/configs/masfdE6/masfdE6_phm_T4_5-2.yaml
python -m src.main --configs src/configs/masfdE6/masfdE6_phm_T4_5-3.yaml
python -m src.main --configs src/configs/masfdE6/masfdE6_phm_T4_5-4.yaml
python -m src.main --configs src/configs/masfdE6/masfdE6_phm_T4_5-5.yaml

echo masfdE7 模型
python -m src.main --configs src/configs/masfdE7/masfdE7_phm_T1_5-1.yaml
python -m src.main --configs src/configs/masfdE7/masfdE7_phm_T1_5-2.yaml
python -m src.main --configs src/configs/masfdE7/masfdE7_phm_T1_5-3.yaml
python -m src.main --configs src/configs/masfdE7/masfdE7_phm_T1_5-4.yaml
python -m src.main --configs src/configs/masfdE7/masfdE7_phm_T1_5-5.yaml
python -m src.main --configs src/configs/masfdE7/masfdE7_phm_T2_5-1.yaml
python -m src.main --configs src/configs/masfdE7/masfdE7_phm_T2_5-2.yaml
python -m src.main --configs src/configs/masfdE7/masfdE7_phm_T2_5-3.yaml
python -m src.main --configs src/configs/masfdE7/masfdE7_phm_T2_5-4.yaml
python -m src.main --configs src/configs/masfdE7/masfdE7_phm_T2_5-5.yaml
python -m src.main --configs src/configs/masfdE7/masfdE7_phm_T3_5-1.yaml
python -m src.main --configs src/configs/masfdE7/masfdE7_phm_T3_5-2.yaml
python -m src.main --configs src/configs/masfdE7/masfdE7_phm_T3_5-3.yaml
python -m src.main --configs src/configs/masfdE7/masfdE7_phm_T3_5-4.yaml
python -m src.main --configs src/configs/masfdE7/masfdE7_phm_T3_5-5.yaml
python -m src.main --configs src/configs/masfdE7/masfdE7_phm_T4_5-1.yaml
python -m src.main --configs src/configs/masfdE7/masfdE7_phm_T4_5-2.yaml
python -m src.main --configs src/configs/masfdE7/masfdE7_phm_T4_5-3.yaml
python -m src.main --configs src/configs/masfdE7/masfdE7_phm_T4_5-4.yaml
python -m src.main --configs src/configs/masfdE7/masfdE7_phm_T4_5-5.yaml
