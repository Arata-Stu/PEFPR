#!/bin/bash

# --- 1. 実行モードと基本引数の取得 ---
MODE=${1:-"finetune"}
shift $(( $# > 0 ? 1 : 0 ))

# --- 2. プロジェクト共通設定 ---
PROJECT_NAME="perp_yolox-s"
DATASET_ROOT="/path/to/dsec"
MODEL_TYPE="perp_yolox"  
MODEL_DEPTH=0.33
MODEL_WIDTH=0.50

# --- 3. 学習モード別プリセット設定 ---
if [ "$MODE" = "finetune" ]; then
    RUN_NAME="perp_yolox-s-finetune"
    PRETRAINED_PATH="./weights/yolox/yolox_s.pth"
    BATCH_SIZE=16
    LR=0.0002
    EPOCHS=50
    WARM_UP=0.3
    EMA=0.9999
    CLIP=0.1
elif [ "$MODE" = "scratch" ]; then
    RUN_NAME="perp_yolox-s-scratch"
    PRETRAINED_PATH="null"
    BATCH_SIZE=16
    LR=0.001
    EPOCHS=300
    WARM_UP=5
    EMA=0.9998
    CLIP=1.0
else
    echo "Error: Unknown mode '$MODE'. Use 'finetune' or 'scratch'."
    exit 1
fi

# --- 4. 実行コマンド ---
echo "Running in [${MODE}] mode with model [${MODEL_TYPE}]..."

python3 train.py \
    model=${MODEL_TYPE} \
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
    training.warmup_epochs=${WARM_UP} \
    training.ema_decay=${EMA} \
    training.num_workers=12 \
    training.use_amp=true \
    training.clip=${CLIP} \
    "$@"