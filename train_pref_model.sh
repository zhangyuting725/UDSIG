#!/bin/bash
# Stage 1: train the preference (reward) model + build the semantic retrieval index.
#
# Dataset defaults to MovieLens (paper Sec. 4.1.1: 598 users / 6,041 items, ratings >= 4).
# To run another dataset, override the four variables below.
set -euo pipefail

# MovieLens (default) / POG: SCENE=pog DATA_NAME=POG ITEM_NUM=14582 / SER: SCENE=ser DATA_NAME=SER ITEM_NUM=14345
SCENE="${SCENE:-movie}"
DATA_NAME="${DATA_NAME:-MovieLens}"
ITEM_NUM="${ITEM_NUM:-6041}"
DATA_ROOT="${DATA_ROOT:-./data/MovieLens}"

# Paper Sec. 4.1.2: UM and RM are trained with batch size 64 and learning rate 1e-3.
python3 train_preference_model.py \
    --scene "${SCENE}" \
    --pref_model_type DIN \
    --item_num "${ITEM_NUM}" \
    --data_name "${DATA_NAME}" \
    --data_path "${DATA_ROOT}" \
    --save_folder "${DATA_ROOT}/results" \
    --image_feat_path "${DATA_ROOT}/image_feat.pt" \
    --device cuda
