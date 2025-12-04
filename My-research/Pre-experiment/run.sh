#!/bin/bash
echo "Starting training..."
# 修改 run.sh 为：
python main.py --mode continual --config configs/config.yaml --save_dir ./outputs
echo "Training completed!"