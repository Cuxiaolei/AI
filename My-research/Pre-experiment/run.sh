#!/bin/bash
echo "Starting training..."
python run.py --mode continual --config configs/config.yaml --save_dir ./outputs
echo "Training completed!"