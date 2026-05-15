#!/bin/bash

# ====================================================
# CIFAR 自动化对比实验脚本 (针对 CPU/WSL 优化)
# ====================================================

# 1. 基础配置
DATASETS=("cifar10" "cifar100")
MODELS=("simple_cnn" "resnet18" "resnet50" "vgg16" "densenet121")

# 2. 性能优化参数
BATCH_SIZE=256    # 增大 batch 以提升 CPU 利用率
NUM_WORKERS=8     # 开启 8 个并行线程读取数据
EPOCHS=10         # 每个模型跑 10 轮进行初步对比 (你可以自行修改)

echo "🚀 开始实验"

for DATASET in "${DATASETS[@]}"
do
    echo "=========================================="
    echo "📂 当前数据集: $DATASET"
    echo "=========================================="

    for MODEL in "${MODELS[@]}"
    do
        echo "------------------------------------------"
        echo "正在训练模型: $MODEL | 数据集: $DATASET"
        echo "------------------------------------------"

        # 执行训练
        # 结果会自动保存在 lightning_logs 文件夹中
        python train.py \
            --dataset $DATASET \
            --model $MODEL \
            --max_epochs $EPOCHS \
            --batch_size $BATCH_SIZE \
            --num_workers $NUM_WORKERS \
            --lr 1e-3

        echo "✅ 模型 $MODEL 在 $DATASET 上的任务已完成。"
    done
done

echo "🎉 运行完毕！查看 lightning_logs 文件夹获取结果。"