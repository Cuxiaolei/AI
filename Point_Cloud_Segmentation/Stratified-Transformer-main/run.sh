#!/bin/bash

# 依次执行训练命令，前一个完成后才会执行下一个
echo "开始执行训练任务..."

python3 train.py --config config/scannetv2/scannetv2_stratified_transformer.yaml

python3 test.py --config config/scannetv2/scannetv2_stratified_transformer.yaml

python3 train.py --config config/scannetv2/scannetv2_swin3d_transformer.yaml

python3 test.py --config config/scannetv2/scannetv2_swin3d_transformer.yaml

echo "所有训练任务执行完毕！"
