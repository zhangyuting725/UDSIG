#!/bin/bash
# Stage 3: personalized image generation with uncertainty-aware retrieval.
#
# Dataset defaults to MovieLens (paper Sec. 4.1.1: 598 users / 6,041 items, ratings >= 4).
# To run another dataset, override the four variables below.
set -euo pipefail

# MovieLens (default) / POG: SCENE=pog DATA_NAME=POG ITEM_NUM=14582 / SER: SCENE=ser DATA_NAME=SER ITEM_NUM=14345
SCENE="${SCENE:-movie}"
DATA_NAME="${DATA_NAME:-MovieLens}"
ITEM_NUM="${ITEM_NUM:-6041}"
DATA_ROOT="${DATA_ROOT:-./data/MovieLens}"

SAVE_FOLDER="${SAVE_FOLDER:-${DATA_ROOT}/results_tmp}"
PREF_MODEL_PATH="${PREF_MODEL_PATH:-${DATA_ROOT}/results/DIN_model.pt}"
# Step of the Stage 2 checkpoint to load; defaults to MAX_TRAIN_STEPS of train_refl_sdxl.sh.
CKPT_STEP="${CKPT_STEP:-50}"
GEN_CKPT="${GEN_CKPT:-${DATA_ROOT}/results/${CKPT_STEP}/checkpoint.pt}"

python3 inference.py \
    --scene "${SCENE}" \
    --pref_model_type DIN \
    --item_num "${ITEM_NUM}" \
    --data_name "${DATA_NAME}" \
    --data_path "${DATA_ROOT}" \
    --top_k 3 \
    --save_folder "${SAVE_FOLDER}" \
    --image_feat_path "${DATA_ROOT}/image_feat.pt" \
    --pref_model_path "${PREF_MODEL_PATH}" \
    --generation_model_path "${GEN_CKPT}" \
    --device cuda
