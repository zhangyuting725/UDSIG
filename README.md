# Uncertain Sparse Gen — Preference-Conditioned Image Generation

A pipeline that fine-tunes Stable Diffusion XL (SDXL) with reinforcement learning from preference feedback (ReFL) to generate personalized images based on user interaction history.

---

## Overview

The released pipeline consists of three sequential stages:

```
train_pref_model.sh  →  train_refl_sdxl.sh  →  inference.sh
 (preference model)      (ReFL fine-tuning)     (generation)
```

The evaluation scripts (Stage 4) are kept internal and are not part of this release.

---

## Project Structure

```
uncertain_image_generation/
├── train_preference_model.py   # Stage 1: train CTR preference model
├── refl_sdxl_retrival.py       # Stage 2: ReFL SDXL fine-tuning entry point
├── inference.py                # Stage 3: personalized image generation
├── data_utils.py               # Shared data utilities and dataset classes
├── models/
│   ├── preference_model.py     # DIN / SASRec preference models
│   ├── gen_model.py            # PreferenceConditionedDiffusion wrapper
│   └── ReFL_SDXL.py            # ReFL trainer (Accelerate + SDXL)
├── train_pref_model.sh         # Stage 1 (dataset set at the top of the script)
├── train_refl_sdxl.sh          # Stage 2
└── inference.sh                # Stage 3
```

---

## Required Data Files

All data files should be placed under a dataset directory:

| File | Description |
|------|-------------|
| `image_feat.pt` | Precomputed image feature tensor, shape `[item_num, feature_dim]` |
| `image_path.pkl` | List of image file paths (length = `item_num`) |
| `train.pkl` | Training interaction data: `{uids, iids, his_iids, ...}` |
| `valid.pkl` | Validation interaction data: same format as train |
| `test.pkl` | Test interaction data: same format as train |
| `reward_model.pt` | Pretrained reward model weights (for evaluation stage) |
| `<image_dir>/` | Directory containing item images referenced by `image_path.pkl` |

---

## Pretrained Models

| Model | Purpose |
|-------|---------|
| SDXL base | Diffusion backbone |
| SDXL VAE (fp16) | VAE for SDXL |
| IP-Adapter | Image prompt conditioning |
| CLIP ViT-L/14 | Text/image encoding |

---

## Stage 1 — Train Preference Model

```bash
bash train_pref_model.sh   # MovieLens is the default dataset

# Other datasets — override the four variables at the top of the script:
SCENE=pog DATA_NAME=POG ITEM_NUM=14582 DATA_ROOT=/path/to/POG bash train_pref_model.sh
SCENE=ser DATA_NAME=SER ITEM_NUM=14345 DATA_ROOT=/path/to/SER bash train_pref_model.sh
```

Trains a CTR (click-through rate) preference model on user interaction history.

**Key arguments** (`train_preference_model.py`):

| Argument | Default | Description |
|----------|---------|-------------|
| `--scene` | `movie` | Dataset scene name |
| `--pref_model_type` | `DIN` | Model type: `DIN` or `SASRec` |
| `--item_num` | `6041` | Number of items |
| `--data_name` | `MovieLens` | Dataset name |
| `--data_path` | — | Path to dataset directory |
| `--save_folder` | — | Output directory for model checkpoint |
| `--image_feat_path` | — | Path to `image_feat.pt` |
| `--device` | `cuda` | Device |

**Output:** `{save_folder}/DIN_model.pt` (or `SASRec_model.pt`)

---

## Stage 2 — ReFL Fine-tuning of SDXL

```bash
bash train_refl_sdxl.sh   # MovieLens is the default dataset

# Other datasets — override the four variables at the top of the script:
SCENE=pog DATA_NAME=POG ITEM_NUM=14582 DATA_ROOT=/path/to/POG bash train_refl_sdxl.sh
SCENE=ser DATA_NAME=SER ITEM_NUM=14345 DATA_ROOT=/path/to/SER bash train_refl_sdxl.sh
```

Paper hyper-parameters (Sec. 4.1.2 / 4.6) are wired into these configs:

| Symbol | Value | Where |
|--------|-------|-------|
| effective batch size | 8 | `--train_batch_size 1` × `--gradient_accumulation_steps 8` |
| learning rate | 1e-5, Adam | `--learning_rate 1e-05` |
| k (retrieved images) | 3 | `--top_k 3` (Sec. 4.6.1) |
| α (teacher-reward weight) | 0.8 | `--alpha` (default `0.8`), scales the reward loss `L_r` (Sec. 4.6.2) |
| optimizer steps | not reported in the paper | `MAX_TRAIN_STEPS` env var (scripts default to 50) |

Fine-tunes SDXL using Reinforcement Learning from preference Feedback (ReFL). The preference model from Stage 1 is used as the reward signal. Launched via HuggingFace Accelerate with fp16 mixed precision.

**Key arguments** (`refl_sdxl_retrival.py` → `models/ReFL_SDXL.py`):

| Argument | Default | Description |
|----------|---------|-------------|
| `--scene` | `movie` | Dataset scene name |
| `--data_name` | `MovieLens` | Dataset name |
| `--data_path` | — | Path to dataset directory |
| `--output_dir` | — | Output directory for checkpoints |
| `--pretrained_model_name_or_path` | — | Path to SDXL base model |
| `--xl_vae_path` | — | Path to SDXL fp16 VAE |
| `--ipadapter_path` | — | Path to IP-Adapter weights |
| `--diffusion_clip_path` | — | Path to CLIP model |
| `--reward_model_type` | `DIN` | Reward model type |
| `--reward_model_path` | — | Path to reward model from Stage 1 |
| `--uncertain_model_type` | `DIN` | Uncertainty model type |
| `--uncertain_model_path` | — | Path to uncertainty model (can reuse reward model) |
| `--hidden_dim` | `128` | Hidden dimension of preference models |
| `--item_num` | `6041` | Number of items |
| `--top_k` | `3` | Number of retrieved reference images (Sec. 4.6.1); must match between training and inference |
| `--train_batch_size` | `1` | Training batch size (per device) |
| `--max_train_steps` | `None` | Total optimizer steps; set through `MAX_TRAIN_STEPS` in the configs |
| `--checkpointing_steps` | `100` | Save checkpoint every N steps |
| `--learning_rate` | `1e-5` | Learning rate |
| `--apply_reward_loss` | flag | Enable reward-based loss |
| `--apply_pre_loss` | flag | Enable preference loss |
| `--alpha` | `0.8` | Weight of the reward loss in `L = α·L_r + L_p` (paper Eq. 11) |
| `--use_ema` | flag | Use EMA for model weights |
| `--seed` | `42` | Random seed |

**Output:** `{output_dir}/{step}/checkpoint.pt`

---

## Stage 3 — Inference (Personalized Image Generation)

```bash
bash inference.sh   # MovieLens is the default dataset

# Other datasets — override the four variables at the top of the script:
SCENE=pog DATA_NAME=POG ITEM_NUM=14582 DATA_ROOT=/path/to/POG bash inference.sh
```

Loads the fine-tuned SDXL checkpoint and generates personalized images for each user in the test set, guided by the preference model's uncertainty-based retrieval.

**Key arguments** (`inference.py`):

| Argument | Default | Description |
|----------|---------|-------------|
| `--scene` | `movie` | Dataset scene name |
| `--pref_model_type` | `DIN` | Preference model type |
| `--item_num` | `6041` | Number of items |
| `--data_name` | `MovieLens` | Dataset name |
| `--data_path` | — | Path to dataset directory |
| `--save_folder` | — | Output directory for generated images and `all_res.npy` |
| `--image_feat_path` | — | Path to `image_feat.pt` |
| `--pref_model_path` | — | Path to preference model from Stage 1 |
| `--generation_model_path` | — | Path to ReFL checkpoint from Stage 2 |
| `--device` | `cuda` | Device |

**Output:**
- `{save_folder}/{uid}/{iid}_generated.jpg` — generated images
- `{save_folder}/all_res.npy` — dict mapping `uid → iid → {gen_image_path, his_iids, ...}`

---

## Stage 4 — Evaluation (internal, not released)

The evaluation code is kept internal and is not included in this release. For reference, it computes:

- **Image quality metrics** — FID, CLIP Score, DINO Image Score, CLIP Image Score, Hist LPIPS, Hist DINO Image Score, SSIM, MS-SSIM between generated and ground-truth images.
- **Reward-model ranking evaluation** — for each user-item pair, samples 99 negative items, scores all candidates (original + generated + negatives) with the reward model, and compares rank positions. Reported metrics: `Better rank rate`, `Avg original image rank`, `Avg generated image rank`, `Avg rank improvement` = mean of `(rank_ori - rank_gen) / (1 + num_negative_samples)`.

Stage 3 writes `{save_folder}/all_res.npy`, which is the input to both evaluations.

---

## Configuring Paths

All paths in the three shell scripts share the same dataset root. To switch datasets, update the following variables consistently across all scripts:

| Variable | Scripts | Description |
|----------|---------|-------------|
| `--data_path` | all | Dataset root directory |
| `--save_folder` | train / inference | Output directory |
| `--image_feat_path` | train / train_refl / inference | Precomputed image features |
| `--reward_model_path` | train_refl | Reward / preference model checkpoint |
| `--generation_model_path` | inference | ReFL fine-tuned checkpoint |
| `--pretrained_model_name_or_path` | train_refl | SDXL base model path |
