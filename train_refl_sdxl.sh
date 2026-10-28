#!/bin/bash
# Stage 2: ReFL fine-tuning of SDXL (only the IP-Adapter parameters are trainable).
#
# Dataset defaults to MovieLens (paper Sec. 4.1.1: 598 users / 6,041 items, ratings >= 4).
# To run another dataset, override the four variables below.
set -euo pipefail

# MovieLens (default) / POG: SCENE=pog DATA_NAME=POG ITEM_NUM=14582 / SER: SCENE=ser DATA_NAME=SER ITEM_NUM=14345
SCENE="${SCENE:-movie}"
DATA_NAME="${DATA_NAME:-MovieLens}"
ITEM_NUM="${ITEM_NUM:-6041}"
DATA_ROOT="${DATA_ROOT:-./data/MovieLens}"

OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/results}"
SDXL_PATH="${SDXL_PATH:-stabilityai/stable-diffusion-xl-base-1.0}"
XL_VAE_PATH="${XL_VAE_PATH:-madebyollin/sdxl-vae-fp16-fix}"
IPADAPTER_PATH="${IPADAPTER_PATH:-h94/IP-Adapter}"
CLIP_PATH="${CLIP_PATH:-openai/clip-vit-large-patch14}"

# Hyper-parameters as reported in the paper:
#   effective batch size 8  = --train_batch_size 1 x --gradient_accumulation_steps 8  (Sec. 4.1.2)
#   learning rate 1e-5, Adam optimizer                                                (Sec. 4.1.2)
#   k (retrieved images) = 3  ->  --top_k 3                                           (Sec. 4.6.1)
#   alpha (teacher-reward weight) = 0.8  ->  --alpha (default 0.8)                    (Sec. 4.6.2)
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-50}"
# Must be <= MAX_TRAIN_STEPS, otherwise no checkpoint is ever written.
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-10}"

accelerate launch --mixed_precision=fp16 refl_sdxl_retrival.py \
  --scene "${SCENE}" \
  --data_name "${DATA_NAME}" \
  --data_path "${DATA_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --image_feat_path "${DATA_ROOT}/image_feat.pt" \
  --pretrained_model_name_or_path="${SDXL_PATH}" \
  --xl_vae_path="${XL_VAE_PATH}" \
  --ipadapter_path="${IPADAPTER_PATH}" \
  --diffusion_clip_path="${CLIP_PATH}" \
  --uncertain_model_type DIN \
  --reward_model_type DIN \
  --top_k 3 \
  --reward_model_path "${DATA_ROOT}/results/DIN_model.pt" \
  --uncertain_model_path "${DATA_ROOT}/results/DIN_model.pt" \
  --hidden_dim=128 \
  --item_num "${ITEM_NUM}" \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=8 \
  --gradient_checkpointing \
  --max_train_steps "${MAX_TRAIN_STEPS}" \
  --checkpointing_steps "${CHECKPOINTING_STEPS}" \
  --learning_rate 1e-05 \
  --max_grad_norm 1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --apply_reward_loss \
  --seed 42 \
  --apply_pre_loss
