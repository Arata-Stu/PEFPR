#!/bin/bash

# --- 1. 基本設定 ---
PROJECT_NAME="yolox-s"
RUN_NAME="yolox-s-fintune"
DATASET_ROOT="/media/arata-22/AT_SSD/dataset/dsec"

# --- 2. トレーニング設定 ---
BATCH_SIZE=16
NUM_WORKERS=12
PRETRAINED_PATH="null" # 学習済みモデルを使用する場合はパスを指定

# --- 3. モデルアーキテクチャ設定 (YOLOX-S相当) ---
# nano: 0.33, 0.25 非推奨
# tiny: 0.33, 0.375
# s: 0.33, 0.50
# m: 0.67, 0.75
# l: 1.0, 1.0
# x: 1.33, 1.25
MODEL_DEPTH=0.33
MODEL_WIDTH=0.50

# --- 4. 実行コマンドの構築 ---
python3 train.py \
    wandb.project=${PROJECT_NAME} \
    wandb.name=${RUN_NAME} \
    dataset.dataset_root=${DATASET_ROOT} \
    dataset.use_events=false \
    training.batch_size=${BATCH_SIZE} \
    training.num_workers=${NUM_WORKERS} \
    training.pretrained_path=${PRETRAINED_PATH} \
    model.backbone.depth=${MODEL_DEPTH} \
    model.backbone.width=${MODEL_WIDTH}