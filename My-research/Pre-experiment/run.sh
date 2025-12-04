#!/bin/bash

# 设置Python路径
export PYTHONPATH="./:$PYTHONPATH"

# 开始训练（使用 -m 参数）
echo "Starting training..."
python -m main --mode continual --config configs/config.yaml --save_dir ./outputs

echo "Training completed!"