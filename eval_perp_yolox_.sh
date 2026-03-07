#!/bin/bash

# --- 1. 基本設定 ---
DATASET_ROOT="/path/to/dsec"
CKPT_PATH="./checkpoints/yolox_s_best.pth"

# --- 2. モデル設定---
MODEL_TYPE="perp_yolox"
MODEL_DEPTH=0.33
MODEL_WIDTH=0.50

# --- 3. 評価用データ・デバイス設定 ---
USE_EVENTS=true
USE_IMAGE=true

# --- 4. 実行コマンド ---
echo "Starting evaluation for model [${MODEL_TYPE}]..."
echo "Using Checkpoint: ${CKPT_PATH}"

python3 eval_perp.py \
    dataset=dsec_det \
    model=${MODEL_TYPE} \
    dataset.dataset_root=${DATASET_ROOT} \
    dataset.use_events=${USE_EVENTS} \
    dataset.use_image=${USE_IMAGE} \
    ckpt_path=${CKPT_PATH} \
    model.backbone.depth=${MODEL_DEPTH} \
    model.backbone.width=${MODEL_WIDTH} \
    "$@"