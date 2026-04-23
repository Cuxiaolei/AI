#!/bin/bash
# 批量执行指令的脚本
set -e  # 关键：任意指令失败则脚本立即退出（可选，增强安全性）

#echo mcpdgE1 模型
#python -m src.main --configs src/configs/mcpdgE1/mcpdgE1_phm_T1_5-1.yaml
#python -m src.main --configs src/configs/mcpdgE1/mcpdgE1_phm_T1_5-2.yaml
#python -m src.main --configs src/configs/mcpdgE1/mcpdgE1_phm_T1_5-3.yaml
#python -m src.main --configs src/configs/mcpdgE1/mcpdgE1_phm_T1_5-4.yaml
#python -m src.main --configs src/configs/mcpdgE1/mcpdgE1_phm_T1_5-5.yaml
#python -m src.main --configs src/configs/mcpdgE1/mcpdgE1_phm_T2_5-1.yaml
#python -m src.main --configs src/configs/mcpdgE1/mcpdgE1_phm_T2_5-2.yaml
#python -m src.main --configs src/configs/mcpdgE1/mcpdgE1_phm_T2_5-3.yaml
#python -m src.main --configs src/configs/mcpdgE1/mcpdgE1_phm_T2_5-4.yaml
#python -m src.main --configs src/configs/mcpdgE1/mcpdgE1_phm_T2_5-5.yaml
#python -m src.main --configs src/configs/mcpdgE1/mcpdgE1_phm_T3_5-1.yaml
#python -m src.main --configs src/configs/mcpdgE1/mcpdgE1_phm_T3_5-2.yaml
#python -m src.main --configs src/configs/mcpdgE1/mcpdgE1_phm_T3_5-3.yaml
#python -m src.main --configs src/configs/mcpdgE1/mcpdgE1_phm_T3_5-4.yaml
#python -m src.main --configs src/configs/mcpdgE1/mcpdgE1_phm_T3_5-5.yaml
#python -m src.main --configs src/configs/mcpdgE1/mcpdgE1_phm_T4_5-1.yaml
#python -m src.main --configs src/configs/mcpdgE1/mcpdgE1_phm_T4_5-2.yaml
#python -m src.main --configs src/configs/mcpdgE1/mcpdgE1_phm_T4_5-3.yaml
#python -m src.main --configs src/configs/mcpdgE1/mcpdgE1_phm_T4_5-4.yaml
#python -m src.main --configs src/configs/mcpdgE1/mcpdgE1_phm_T4_5-5.yaml
#
#echo mcpdgE2 模型
#python -m src.main --configs src/configs/mcpdgE2/mcpdgE2_phm_T1_5-1.yaml
#python -m src.main --configs src/configs/mcpdgE2/mcpdgE2_phm_T1_5-2.yaml
#python -m src.main --configs src/configs/mcpdgE2/mcpdgE2_phm_T1_5-3.yaml
#python -m src.main --configs src/configs/mcpdgE2/mcpdgE2_phm_T1_5-4.yaml
#python -m src.main --configs src/configs/mcpdgE2/mcpdgE2_phm_T1_5-5.yaml
#python -m src.main --configs src/configs/mcpdgE2/mcpdgE2_phm_T2_5-1.yaml
#python -m src.main --configs src/configs/mcpdgE2/mcpdgE2_phm_T2_5-2.yaml
#python -m src.main --configs src/configs/mcpdgE2/mcpdgE2_phm_T2_5-3.yaml
#python -m src.main --configs src/configs/mcpdgE2/mcpdgE2_phm_T2_5-4.yaml
#python -m src.main --configs src/configs/mcpdgE2/mcpdgE2_phm_T2_5-5.yaml
#python -m src.main --configs src/configs/mcpdgE2/mcpdgE2_phm_T3_5-1.yaml
#python -m src.main --configs src/configs/mcpdgE2/mcpdgE2_phm_T3_5-2.yaml
#python -m src.main --configs src/configs/mcpdgE2/mcpdgE2_phm_T3_5-3.yaml
#python -m src.main --configs src/configs/mcpdgE2/mcpdgE2_phm_T3_5-4.yaml
#python -m src.main --configs src/configs/mcpdgE2/mcpdgE2_phm_T3_5-5.yaml
#python -m src.main --configs src/configs/mcpdgE2/mcpdgE2_phm_T4_5-1.yaml
#python -m src.main --configs src/configs/mcpdgE2/mcpdgE2_phm_T4_5-2.yaml
#python -m src.main --configs src/configs/mcpdgE2/mcpdgE2_phm_T4_5-3.yaml
#python -m src.main --configs src/configs/mcpdgE2/mcpdgE2_phm_T4_5-4.yaml
#python -m src.main --configs src/configs/mcpdgE2/mcpdgE2_phm_T4_5-5.yaml
#
#echo mcpdgE3 模型
#python -m src.main --configs src/configs/mcpdgE3/mcpdgE3_phm_T1_5-1.yaml
#python -m src.main --configs src/configs/mcpdgE3/mcpdgE3_phm_T1_5-2.yaml
#python -m src.main --configs src/configs/mcpdgE3/mcpdgE3_phm_T1_5-3.yaml
#python -m src.main --configs src/configs/mcpdgE3/mcpdgE3_phm_T1_5-4.yaml
#python -m src.main --configs src/configs/mcpdgE3/mcpdgE3_phm_T1_5-5.yaml
#python -m src.main --configs src/configs/mcpdgE3/mcpdgE3_phm_T2_5-1.yaml
#python -m src.main --configs src/configs/mcpdgE3/mcpdgE3_phm_T2_5-2.yaml
#python -m src.main --configs src/configs/mcpdgE3/mcpdgE3_phm_T2_5-3.yaml
#python -m src.main --configs src/configs/mcpdgE3/mcpdgE3_phm_T2_5-4.yaml
#python -m src.main --configs src/configs/mcpdgE3/mcpdgE3_phm_T2_5-5.yaml
#python -m src.main --configs src/configs/mcpdgE3/mcpdgE3_phm_T3_5-1.yaml
#python -m src.main --configs src/configs/mcpdgE3/mcpdgE3_phm_T3_5-2.yaml
#python -m src.main --configs src/configs/mcpdgE3/mcpdgE3_phm_T3_5-3.yaml
#python -m src.main --configs src/configs/mcpdgE3/mcpdgE3_phm_T3_5-4.yaml
#python -m src.main --configs src/configs/mcpdgE3/mcpdgE3_phm_T3_5-5.yaml
#python -m src.main --configs src/configs/mcpdgE3/mcpdgE3_phm_T4_5-1.yaml
#python -m src.main --configs src/configs/mcpdgE3/mcpdgE3_phm_T4_5-2.yaml
#python -m src.main --configs src/configs/mcpdgE3/mcpdgE3_phm_T4_5-3.yaml
#python -m src.main --configs src/configs/mcpdgE3/mcpdgE3_phm_T4_5-4.yaml
#python -m src.main --configs src/configs/mcpdgE3/mcpdgE3_phm_T4_5-5.yaml
#
#echo mcpdgE4 模型
#python -m src.main --configs src/configs/mcpdgE4/mcpdgE4_phm_T1_5-1.yaml
#python -m src.main --configs src/configs/mcpdgE4/mcpdgE4_phm_T1_5-2.yaml
#python -m src.main --configs src/configs/mcpdgE4/mcpdgE4_phm_T1_5-3.yaml
#python -m src.main --configs src/configs/mcpdgE4/mcpdgE4_phm_T1_5-4.yaml
#python -m src.main --configs src/configs/mcpdgE4/mcpdgE4_phm_T1_5-5.yaml
#python -m src.main --configs src/configs/mcpdgE4/mcpdgE4_phm_T2_5-1.yaml
#python -m src.main --configs src/configs/mcpdgE4/mcpdgE4_phm_T2_5-2.yaml
#python -m src.main --configs src/configs/mcpdgE4/mcpdgE4_phm_T2_5-3.yaml
#python -m src.main --configs src/configs/mcpdgE4/mcpdgE4_phm_T2_5-4.yaml
#python -m src.main --configs src/configs/mcpdgE4/mcpdgE4_phm_T2_5-5.yaml
#python -m src.main --configs src/configs/mcpdgE4/mcpdgE4_phm_T3_5-1.yaml
#python -m src.main --configs src/configs/mcpdgE4/mcpdgE4_phm_T3_5-2.yaml
#python -m src.main --configs src/configs/mcpdgE4/mcpdgE4_phm_T3_5-3.yaml
#python -m src.main --configs src/configs/mcpdgE4/mcpdgE4_phm_T3_5-4.yaml
#python -m src.main --configs src/configs/mcpdgE4/mcpdgE4_phm_T3_5-5.yaml
#python -m src.main --configs src/configs/mcpdgE4/mcpdgE4_phm_T4_5-1.yaml
#python -m src.main --configs src/configs/mcpdgE4/mcpdgE4_phm_T4_5-2.yaml
#python -m src.main --configs src/configs/mcpdgE4/mcpdgE4_phm_T4_5-3.yaml
#python -m src.main --configs src/configs/mcpdgE4/mcpdgE4_phm_T4_5-4.yaml
#python -m src.main --configs src/configs/mcpdgE4/mcpdgE4_phm_T4_5-5.yaml

#echo mcpdgE5 模型
#python -m src.main --configs src/configs/mcpdgE5/mcpdgE5_phm_T1_5-1.yaml
#python -m src.main --configs src/configs/mcpdgE5/mcpdgE5_phm_T1_5-2.yaml
#python -m src.main --configs src/configs/mcpdgE5/mcpdgE5_phm_T1_5-3.yaml
#python -m src.main --configs src/configs/mcpdgE5/mcpdgE5_phm_T1_5-4.yaml
#python -m src.main --configs src/configs/mcpdgE5/mcpdgE5_phm_T1_5-5.yaml
#python -m src.main --configs src/configs/mcpdgE5/mcpdgE5_phm_T2_5-1.yaml
#python -m src.main --configs src/configs/mcpdgE5/mcpdgE5_phm_T2_5-2.yaml
#python -m src.main --configs src/configs/mcpdgE5/mcpdgE5_phm_T2_5-3.yaml
#python -m src.main --configs src/configs/mcpdgE5/mcpdgE5_phm_T2_5-4.yaml
#python -m src.main --configs src/configs/mcpdgE5/mcpdgE5_phm_T2_5-5.yaml
#python -m src.main --configs src/configs/mcpdgE5/mcpdgE5_phm_T3_5-1.yaml
#python -m src.main --configs src/configs/mcpdgE5/mcpdgE5_phm_T3_5-2.yaml
#python -m src.main --configs src/configs/mcpdgE5/mcpdgE5_phm_T3_5-3.yaml
#python -m src.main --configs src/configs/mcpdgE5/mcpdgE5_phm_T3_5-4.yaml
#python -m src.main --configs src/configs/mcpdgE5/mcpdgE5_phm_T3_5-5.yaml
#python -m src.main --configs src/configs/mcpdgE5/mcpdgE5_phm_T4_5-1.yaml
#python -m src.main --configs src/configs/mcpdgE5/mcpdgE5_phm_T4_5-2.yaml
#python -m src.main --configs src/configs/mcpdgE5/mcpdgE5_phm_T4_5-3.yaml
#python -m src.main --configs src/configs/mcpdgE5/mcpdgE5_phm_T4_5-4.yaml
#python -m src.main --configs src/configs/mcpdgE5/mcpdgE5_phm_T4_5-5.yaml
#
#echo mcpdgE6 模型
#python -m src.main --configs src/configs/mcpdgE6/mcpdgE6_phm_T1_5-1.yaml
#python -m src.main --configs src/configs/mcpdgE6/mcpdgE6_phm_T1_5-2.yaml
#python -m src.main --configs src/configs/mcpdgE6/mcpdgE6_phm_T1_5-3.yaml
#python -m src.main --configs src/configs/mcpdgE6/mcpdgE6_phm_T1_5-4.yaml
#python -m src.main --configs src/configs/mcpdgE6/mcpdgE6_phm_T1_5-5.yaml
#python -m src.main --configs src/configs/mcpdgE6/mcpdgE6_phm_T2_5-1.yaml
#python -m src.main --configs src/configs/mcpdgE6/mcpdgE6_phm_T2_5-2.yaml
#python -m src.main --configs src/configs/mcpdgE6/mcpdgE6_phm_T2_5-3.yaml
#python -m src.main --configs src/configs/mcpdgE6/mcpdgE6_phm_T2_5-4.yaml
#python -m src.main --configs src/configs/mcpdgE6/mcpdgE6_phm_T2_5-5.yaml
#python -m src.main --configs src/configs/mcpdgE6/mcpdgE6_phm_T3_5-1.yaml
#python -m src.main --configs src/configs/mcpdgE6/mcpdgE6_phm_T3_5-2.yaml
#python -m src.main --configs src/configs/mcpdgE6/mcpdgE6_phm_T3_5-3.yaml
#python -m src.main --configs src/configs/mcpdgE6/mcpdgE6_phm_T3_5-4.yaml
#python -m src.main --configs src/configs/mcpdgE6/mcpdgE6_phm_T3_5-5.yaml
#python -m src.main --configs src/configs/mcpdgE6/mcpdgE6_phm_T4_5-1.yaml
#python -m src.main --configs src/configs/mcpdgE6/mcpdgE6_phm_T4_5-2.yaml
#python -m src.main --configs src/configs/mcpdgE6/mcpdgE6_phm_T4_5-3.yaml
#python -m src.main --configs src/configs/mcpdgE6/mcpdgE6_phm_T4_5-4.yaml
#python -m src.main --configs src/configs/mcpdgE6/mcpdgE6_phm_T4_5-5.yaml

echo mcpdgE7 模型
python -m src.main --configs src/configs/mcpdgE7/mcpdgE7_phm_T1_5-1.yaml
python -m src.main --configs src/configs/mcpdgE7/mcpdgE7_phm_T1_5-2.yaml
python -m src.main --configs src/configs/mcpdgE7/mcpdgE7_phm_T1_5-3.yaml
python -m src.main --configs src/configs/mcpdgE7/mcpdgE7_phm_T1_5-4.yaml
python -m src.main --configs src/configs/mcpdgE7/mcpdgE7_phm_T1_5-5.yaml
python -m src.main --configs src/configs/mcpdgE7/mcpdgE7_phm_T2_5-1.yaml
python -m src.main --configs src/configs/mcpdgE7/mcpdgE7_phm_T2_5-2.yaml
python -m src.main --configs src/configs/mcpdgE7/mcpdgE7_phm_T2_5-3.yaml
python -m src.main --configs src/configs/mcpdgE7/mcpdgE7_phm_T2_5-4.yaml
python -m src.main --configs src/configs/mcpdgE7/mcpdgE7_phm_T2_5-5.yaml
python -m src.main --configs src/configs/mcpdgE7/mcpdgE7_phm_T3_5-1.yaml
python -m src.main --configs src/configs/mcpdgE7/mcpdgE7_phm_T3_5-2.yaml
python -m src.main --configs src/configs/mcpdgE7/mcpdgE7_phm_T3_5-3.yaml
python -m src.main --configs src/configs/mcpdgE7/mcpdgE7_phm_T3_5-4.yaml
python -m src.main --configs src/configs/mcpdgE7/mcpdgE7_phm_T3_5-5.yaml
python -m src.main --configs src/configs/mcpdgE7/mcpdgE7_phm_T4_5-1.yaml
python -m src.main --configs src/configs/mcpdgE7/mcpdgE7_phm_T4_5-2.yaml
python -m src.main --configs src/configs/mcpdgE7/mcpdgE7_phm_T4_5-3.yaml
python -m src.main --configs src/configs/mcpdgE7/mcpdgE7_phm_T4_5-4.yaml
python -m src.main --configs src/configs/mcpdgE7/mcpdgE7_phm_T4_5-5.yaml