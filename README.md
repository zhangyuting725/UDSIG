# Uncertain Sparse Gen — Preference-Conditioned Image Generation

A pipeline that fine-tunes Stable Diffusion XL (SDXL) with reinforcement learning from preference feedback (ReFL) to generate personalized images based on user interaction history.

---

## Overview

The released pipeline consists of three sequential stages:

```
train_pref_model.sh  →  train_refl_sdxl.sh  →  inference.sh
 (preference model)      (ReFL fine-tuning)     (generation)
```

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
| `image_path.pkl` | List of image file names (length = `item_num`), relative to `poster/` |
| `movie_info.npy` | Per-item metadata dict keyed by item id; `movie_info[iid]['description']` is the text prompt |
| `train.pkl` | Training interactions: a DataFrame with `userId` and `movie_seq` |
| `valid.pkl` | Validation interactions: same format as train |
| `test.pkl` | Test interactions: same format as train |
| `near_images.npy` | Semantic neighbours per item, shape `[item_num, 100]` |
| `near_images_sim.npy` | Cosine similarity of those neighbours, shape `[item_num, 100]` |
| `poster/` | Directory containing the item images referenced by `image_path.pkl` |

`near_images.npy` / `near_images_sim.npy` are built by `data_utils.get_top_k_similar_images()` during Stage 1 (CLIP text embeddings of `movie_info[*]['description']`, top-100 neighbours per item); if they already exist they are reused as-is.

Stage 1 also writes `{save_folder}/DIN_model.pt`, which Stages 2 and 3 load as the preference / reward model — it is an output, not an input.

### Data Sampling (training and inference)

Every sample is built by `IPAdapterDataset` (`data_utils.py`) as follows:

- **Target item** — drawn **uniformly at random from the whole catalogue** (`random.randint(0, item_num - 1)`), *not* from `data['iids']`. This is intentional data augmentation: the image-prompt target is decoupled from the interaction sequence, so the model cannot simply memorise (history → next item) pairs.
- **Sparse history** — 1–3 interactions are randomly sub-sampled from the user's history (`k = random.randint(1, 3)`) to simulate sparse / cold-start users.
- Training and inference share this dataset class with no train/eval branch, so the `(uid, iid)` pairs written to `all_res.npy` are these **randomly sampled pairs**, not fixed held-out pairs.

To evaluate on fixed held-out pairs instead, restore `tgt_iid = self.data['iids'][idx]` in `data_utils.py`.

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
