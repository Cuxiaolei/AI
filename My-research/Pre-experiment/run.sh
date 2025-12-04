#!/bin/bash
# 一键运行完整实验

echo "开始2D-ResNet小样本域泛化实验"

训练
echo "⏱️  开始训练"

# 步骤1：快速验证
python main.py --mode train --epochs 10 --save_dir ./debug

# 步骤2：核心训练
python main.py --mode continual --save_dir ./exp_main

# 步骤3：最终评估
python main.py --mode test --resume ./exp_main/best_model.pth --save_dir ./exp_final