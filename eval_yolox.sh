#!/bin/bash

# --- 1. 基本設定（ここを書き換える） ---
DATASET_ROOT="/media/arata-22/AT_SSD/dataset/dsec"
CKPT_PATH="checkpoints/.pth"

# --- 2. モデル設定 --
MODEL_DEPTH=0.33
MODEL_WIDTH=0.50

# --- 3. データセット・デバイス設定 ---
USE_EVENTS=false
USE_IMAGE=true

# --- 4. 実行コマンドの構築 ---
python3 eval.py \
    dataset=dsec_det \
    model=yolox \
    dataset.dataset_root=${DATASET_ROOT} \
    dataset.use_events=${USE_EVENTS} \
    dataset.use_image=${USE_IMAGE} \
    ckpt_path=${CKPT_PATH} \
    model.backbone.depth=${MODEL_DEPTH} \
    model.backbone.width=${MODEL_WIDTH}