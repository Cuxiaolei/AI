#!/bin/bash
# 一键运行完整实验

echo "开始2D-ResNet小样本域泛化实验"


# 3. 清理旧结果
if [ -d "./results" ]; then
    echo "🗑️  清理旧结果..."
    rm -rf ./results
fi

mkdir -p ./results

# 4. 运行训练
echo "⏱️  开始训练"
python main.py

echo "实验完成！"
echo "结果位置:"
echo "   模型: ./results/resnet2d_model.pt"
echo "   报告: ./results/detailed_results.txt"