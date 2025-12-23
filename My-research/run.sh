#!/usr/bin/env bash
# 执行训练脚本（-u 禁用输出缓冲，保证日志实时打印）
python -u run.py --config configs/base.yaml configs/experiments/pu_dsfsfd_lodo.yaml

# 训练脚本执行完毕后，立即关机（-h 表示halt/关机，now 表示立即执行）
shutdown -h now