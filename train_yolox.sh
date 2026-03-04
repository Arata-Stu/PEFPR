#!/bin/bash

# --- 1. 実行モードの取得 ---
MODE=${1:-"finetune"}

# --- 2. 共通パス設定 ---
PROJECT_NAME="yolox-s"
DATASET_ROOT="/media/arata-22/AT_SSD/dataset/dsec"
MODEL_DEPTH=0.33
MODEL_WIDTH=0.50

# --- 3. 学習モード別プリセット設定 ---
if [ "$MODE" = "finetune" ]; then
    RUN_NAME="yolox-s-finetune"
    # Finetuning用の推奨設定
    PRETRAINED_PATH="./weights/yolox_s.pth"
    BATCH_SIZE=16
    LR=0.0002
    EPOCHS=50
    WARMUP=0.3
    EMA=0.9999
    CLIP=0.1
elif [ "$MODE" = "scratch" ]; then
    RUN_NAME="yolox-s-scratch"
    # Scratch（最初から）用の推奨設定
    PRETRAINED_PATH="null"
    BATCH_SIZE=16           # Scratchはリソースを食う場合があるため調整
    LR=0.01                # 最初からは高めのLRが一般的
    EPOCHS=300             # Scratchは長く回す必要がある
    WARMUP=5               # ウォームアップも長めに
    EMA=0.9998
    CLIP=1.0
else
    echo "Error: Unknown mode '$MODE'. Use 'finetune' or 'scratch'."
    exit 1
fi

# --- 4. 実行コマンド ---
echo "Running in [${MODE}] mode..."

python3 train.py \
    wandb.project=${PROJECT_NAME} \
    wandb.name=${RUN_NAME} \
    dataset.dataset_root=${DATASET_ROOT} \
    dataset.use_events=false \
    model.backbone.depth=${MODEL_DEPTH} \
    model.backbone.width=${MODEL_WIDTH} \
    training.pretrained_path=${PRETRAINED_PATH} \
    training.batch_size=${BATCH_SIZE} \
    training.lr=${LR} \
    training.tot_num_epochs=${EPOCHS} \
    training.warmup_epochs=${WARMUP} \
    training.ema_decay=${EMA} \
    training.num_workers=12 \
    training.use_amp=true \
    training.clip=CLIP