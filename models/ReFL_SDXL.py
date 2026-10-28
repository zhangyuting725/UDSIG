import argparse
import functools
import gc
import logging
import math
import os
import random
import shutil
import numpy as np
import pickle
import accelerate
import datasets
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import concatenate_datasets, load_dataset
from packaging import version
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import AutoencoderKL, DiffusionPipeline,DDPMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from transformers import CLIPVisionModelWithProjection, CLIPProcessor
import torch.nn as nn

from models.preference_model import DIN, SASRec

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))  
parent_dir = os.path.dirname(current_dir)  
sys.path.append(parent_dir)
import data_utils

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.26.0")

logger = get_logger(__name__)

DATASET_NAME_MAPPING = {}





def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--grad_scale", type=float, default=1e-3, help="Scale divided for grad loss value."
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="The directory where the training data is stored.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sdxl-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=100,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam self.optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam self.optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam self.optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `self.noise_scheduler.config.prediction_type` is chosen.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )

    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    # Personal usage arguments
    parser.add_argument(
        "--apply_pre_loss", action="store_true", help="Whether or not to apply pretrained loss."
    )
    parser.add_argument(
        "--apply_reward_loss", action="store_true", help="Whether or not to apply reward loss."
    )
    parser.add_argument(
        "--alpha", type=float, default=0.8,
        help="Weight of the reward loss in the objective L = alpha * L_r + L_p (paper Eq. 11, "
             "Sec. 4.6.2). Only scales the reward term; L_p keeps a coefficient of 1."
    )
    parser.add_argument(
        "--image_base_dir",
        default="",
        help="The base directory where stored all the images."
    )
    parser.add_argument(
        "--mapping_batch_size",
        default=128,
        type=int,
        help="Batch size to map huggingface data."
    )


    #-------------my
    parser.add_argument("--device", type=str, default="cpu", help="Specify the device, e.g., 'cuda' or 'cpu'")

    # Dataset configuration
    parser.add_argument("--data_name", type=str)
    parser.add_argument("--data_path", type=str, default="", help="Path to the dataset")
    parser.add_argument("--item_num", type=int, default=1000, help="Number of items")

    # Pretrained Model paths 
   
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="", help="Path to pretrained model")
    parser.add_argument("--xl_vae_path", type=str, default="", help="Path to the fp16")
    parser.add_argument("--diffusion_clip_path",type =str,default ='', help="Path to the clip model")
    parser.add_argument("--ipadapter_path",type=str, default="", help="Path to IP-Adapter model")
    # Model type selection
    
    # Feature paths
    parser.add_argument("--image_feat_path", type=str, help="Path to image features")
    
    parser.add_argument("--uncertain_model_type", type=str, default="DIN", choices=['DIN', 'SASRec'])
    parser.add_argument("--uncertain_model_path", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=3,
                       help="Number of reference images retrieved per target and fed to IP-Adapter (must match inference)")
    # Reward model parameters
    parser.add_argument("--reward_model_type", type=str, default="DIN", choices=['DIN', 'SASRec'])
    parser.add_argument("--reward_model_path", type=str, default=None,
                       help="Path to pretrained reward model checkpoint (for CTR-based models)")
    parser.add_argument("--hidden_dim",type=int, default=128,help="hidden dimension for reward model")
    parser.add_argument("--SASRec_num_layers", type=int, default=2, help="Number of Transformer layers for SASRec")
    parser.add_argument("--SASRec_num_heads", type=int, default=4, help="Number of attention heads for SASRec")
    parser.add_argument("--SASRec_dropout", type=float, default=0.2, help="Dropout rate for SASRec")
    parser.add_argument("--max_seq_len", type=int, default=15, help="Maximum sequence length for SASRec-CTR")
    
    

    parser.add_argument("--scene", type=str, default='movie',  help="dataset scene")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args




class Trainer(object):
    def __init__(self, args) -> None:
        self.pretrained_model_name_or_path = args.pretrained_model_name_or_path
        self.diffusion_clip_path = args.diffusion_clip_path
        logging_dir = os.path.join(args.output_dir, args.logging_dir)

        accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with=args.report_to,
            project_config=accelerator_project_config,
            project_dir=logging_dir
        )

        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(self.accelerator.state, main_process_only=False)
        if self.accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        # If passed along, set the training seed now.
        if args.seed is not None:
            set_seed(args.seed)
            logger.info(f"🎲 Setting random seed to {args.seed} for reproducibility")

        if self.accelerator.is_main_process:
            if args.output_dir is not None and not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir, exist_ok=True)
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            args.ipadapter_path,
            subfolder="models/image_encoder",
            
        )
        # Load diffusion model without specifying torch_dtype for unet
        # Let accelerator handle mixed precision training automatically
        if self.accelerator.mixed_precision == "fp16":
            vae = AutoencoderKL.from_pretrained(args.xl_vae_path, torch_dtype=torch.float16)
            self.diffusion_model = DiffusionPipeline.from_pretrained(
                self.pretrained_model_name_or_path,
                vae=vae, image_encoder=image_encoder, use_safetensors=True)
        else:
            self.diffusion_model = DiffusionPipeline.from_pretrained(
                self.pretrained_model_name_or_path,
                image_encoder=image_encoder, use_safetensors=True)
        self.diffusion_model.load_ip_adapter(
                args.ipadapter_path,
                subfolder="sdxl_models",
                weight_name=["ip-adapter-plus_sdxl_vit-h.safetensors"]
        )
        self.diffusion_model.set_ip_adapter_scale([0.7])
        # Load the tokenizers
        
        tokenizer_one = self.diffusion_model.tokenizer  
        tokenizer_two = self.diffusion_model.tokenizer_2
        text_encoder_one = self.diffusion_model.text_encoder
        text_encoder_two = self.diffusion_model.text_encoder_2
        # Load scheduler and models
        
        self.noise_scheduler = DDPMScheduler.from_pretrained(self.pretrained_model_name_or_path, 
                                                             subfolder="scheduler",
                                                             cache_dir=args.cache_dir
        )

       
        self.vae = self.diffusion_model.vae
        self.unet = self.diffusion_model.unet

        self.image_features = data_utils.get_image_feature(args).to(self.accelerator.device)
        # Load uncertain model
        print(f"Loading uncertain model from {args.uncertain_model_path}...")
        feat_dim = self.image_features.size(1)
        if args.uncertain_model_type == 'SASRec':
            uncertain_model = SASRec(
                item_image_feature_dim=feat_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.SASRec_num_layers,
                num_heads=args.SASRec_num_heads,
                dropout=args.SASRec_dropout,
                max_seq_len=args.max_seq_len
            ).to(args.device)
        elif args.uncertain_model_type == 'DIN':
            uncertain_model = DIN(
                item_image_feature_dim=feat_dim,
                hidden_dim=args.hidden_dim
            ).to(args.device)
        else:
            raise ValueError(f"Unsupported uncertain_model_type: {args.uncertain_model_type}")
        checkpoint = torch.load(args.uncertain_model_path, map_location=args.device)
        uncertain_model.load_state_dict(checkpoint['model_state_dict'])
        uncertain_model.requires_grad_(False)
        uncertain_model.eval()
        print("✅ uncertain model loaded successfully!")
        self.set_retrival_attr(
            data_path=args.data_path,
            image_features=self.image_features,
            uncertain_model=uncertain_model,
            top_k=args.top_k
        )
        # Load reward model
        if args.reward_model_type == 'SASRec':
            logger.info("Loading SASRec reward model")
            self.reward_model = SASRec(
                item_image_feature_dim=feat_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.SASRec_num_layers,
                num_heads=args.SASRec_num_heads,
                dropout=args.SASRec_dropout,
                max_seq_len=args.max_seq_len
            ).to(self.accelerator.device)
        elif args.reward_model_type == 'DIN':
            logger.info("Loading DIN reward model")
            self.reward_model = DIN(
                item_image_feature_dim=feat_dim,
                hidden_dim=args.hidden_dim
            ).to(self.accelerator.device)
        else:
            raise ValueError(f"Unsupported reward_model_type: {args.reward_model_type}")
        
        checkpoint = torch.load(args.reward_model_path, map_location=self.accelerator.device)
        self.reward_model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded pretrained reward model from: {args.reward_model_path}")
        self.reward_model.requires_grad_(False)
        self.reward_model.eval()  # Set to eval mode to handle batch_size=1

        

        # Freeze vae and text encoders.
        self.vae.requires_grad_(False)
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)
        

        # Freeze the entire UNet first, then we'll unfreeze only IP-Adapter parameters
        self.unet.requires_grad_(False)

        # For mixed precision training we cast all non-trainable weights to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        # Move models to device and cast to weight_dtype
        # Note: reward_model stays in float32 for numerical stability (it's frozen anyway)
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        text_encoder_one.to(self.accelerator.device, dtype=self.weight_dtype)
        text_encoder_two.to(self.accelerator.device, dtype=self.weight_dtype)
        # Keep reward_model in float32 for CTR-based models
        self.reward_model.to(self.accelerator.device, dtype=torch.float32)
        
        # Move image_encoder for IP-Adapter to device
        if hasattr(self.diffusion_model, 'image_encoder') and self.diffusion_model.image_encoder is not None:
            self.diffusion_model.image_encoder.to(self.accelerator.device, dtype=self.weight_dtype)

        # Create EMA for the self.unet.
        # IMPORTANT: Create EMA from the actual unet (which includes IP-Adapter modifications)
        # not from a freshly loaded pretrained model
        if args.use_ema:
            self.ema_unet = EMAModel(self.unet.parameters(), model_cls=UNet2DConditionModel, model_config=self.unet.config)
        
        # Set up trainable parameters (only IP-Adapter components)
        self._set_train_parameters()

        if args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                import xformers

                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    logger.warning(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                self.unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

        # `accelerate` 0.16.0 will have better support for customized saving
        if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
            # create custom saving & loading hooks so that `self.accelerator.save_state(...)` serializes in a nice format
            def save_model_hook(models, weights, output_dir):
                if self.accelerator.is_main_process:
                    if args.use_ema:
                        self.ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                    for i, model in enumerate(models):
                        model.save_pretrained(os.path.join(output_dir, "unet"))

                        # make sure to pop weight so that corresponding model is not saved again
                        if weights:
                            weights.pop()

            def load_model_hook(models, input_dir):
                if args.use_ema:
                    load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                    self.ema_unet.load_state_dict(load_model.state_dict())
                    self.ema_unet.to(self.accelerator.device)
                    del load_model

                for _ in range(len(models)):
                    # pop models so that they are not loaded again
                    model = models.pop()

                    # load diffusers style into model
                    load_model = UNet2DConditionModel.from_pretrained(input_dir, 
                                                                      subfolder="unet",
                                                                      cache_dir=args.cache_dir
                    )
                    
                    model.register_to_config(**load_model.config)

                    model.load_state_dict(load_model.state_dict())
                    del load_model

            self.accelerator.register_save_state_pre_hook(save_model_hook)
            self.accelerator.register_load_state_pre_hook(load_model_hook)

        if args.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        if args.scale_lr:
            args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * self.accelerator.num_processes
            )

        # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        # Optimizer creation
        # Only optimize IP-Adapter parameters instead of all UNet parameters
        params_to_optimize = self.trainable_params
        logger.info(f"Training only IP-Adapter parameters: {len(list(params_to_optimize))} parameter groups")
        self.optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

        # Get the datasets: you can either provide your own training and evaluation files (see below)
        # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

       
        

        self.process_data(args,tokenizer_one,tokenizer_two,text_encoder_one,text_encoder_two)
        del tokenizer_one,tokenizer_two,text_encoder_one,text_encoder_two
        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        self.num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * self.num_update_steps_per_epoch
            overrode_max_train_steps = True

        self.lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        )

        # Prepare everything with our `self.accelerator`.
        self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler
        )

        if args.use_ema:
            self.ema_unet.to(self.accelerator.device)

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        self.num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / args.gradient_accumulation_steps)
        if overrode_max_train_steps:
            args.max_train_steps = args.num_train_epochs * self.num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        args.num_train_epochs = math.ceil(args.max_train_steps / self.num_update_steps_per_epoch)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if self.accelerator.is_main_process:
            # Filter args to only include serializable types for TensorBoard
            tracker_config = {}
            for key, value in vars(args).items():
                if isinstance(value, (int, float, str, bool)) or value is None:
                    tracker_config[key] = value
                elif isinstance(value, (list, tuple)) and all(isinstance(v, (int, float, str, bool)) for v in value):
                    tracker_config[key] = str(value)  # Convert list/tuple to string
                else:
                    # Skip complex objects (like torch.device, etc.)
                    continue
            self.accelerator.init_trackers("text2image-refl", config=tracker_config)

    def process_data(self,args,tokenizer_one,tokenizer_two,text_encoder_one,text_encoder_two):
        
        train_resize = transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR)
        train_crop = transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution)
        train_flip = transforms.RandomHorizontalFlip(p=1.0)
        train_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        
        # Load custom dataset
        train_data, valid_data, test_data = data_utils.load_data(args)
        train_dataset = data_utils.IPAdapterDataset(args, train_data)
        
        # Define image preprocessing function
        def preprocess_image(image):
            """Preprocess single PIL image"""
            # Check input image
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            original_size = (image.height, image.width)
            image = train_resize(image)
            if args.random_flip and random.random() < 0.5:
                image = train_flip(image)
            if args.center_crop:
                y1 = max(0, int(round((image.height - args.resolution) / 2.0)))
                x1 = max(0, int(round((image.width - args.resolution) / 2.0)))
                image = train_crop(image)
            else:
                y1, x1, h, w = train_crop.get_params(image, (args.resolution, args.resolution))
                image = crop(image, y1, x1, h, w)
            crop_top_left = (y1, x1)
            
            # Convert to tensor and normalize
            image_tensor = train_transforms(image)
            
            # Check for NaN after transforms
            if torch.isnan(image_tensor).any() or torch.isinf(image_tensor).any():
                image_tensor = torch.nan_to_num(image_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return image_tensor, original_size, crop_top_left

        # Let's first compute all the embeddings so that we can free up the text encoders
        # from memory. We will pre-compute the VAE encodings too.
        # text_encoders = [text_encoder_one, text_encoder_two]
        # tokenizers = [tokenizer_one, tokenizer_two]


            
        gc.collect()
        torch.cuda.empty_cache()

        def collate_fn(batch):
            """Process batch data returned by IPAdapterDataset"""
            processed_batch = []
            result = {}
            from torch.utils.data.dataloader import default_collate

            # Define which fields should be skipped (don't use default_collate)
            # These fields are either PIL images, variable-length tensors, or not needed in subsequent processing
            skip_keys = ['target_image', 'his_images', 'his_iids', 'sparse_iids']
            
            for key in batch[0].keys():
                if key in skip_keys:
                    # PIL images or variable-length tensors return list directly
                    result[key] = [item[key] for item in batch]
                else:
                    # Let official default_collate handle other fields (scalar fields like iids, uids, prompt, etc.)
                    result[key] = default_collate([item[key] for item in batch])
            
            for sample in batch:
                # Process target images
                target_image = sample['target_image']
                image_tensor, original_size, crop_top_left = preprocess_image(target_image)
                
                # Encode text prompts
                prompt = sample['prompt']
                with torch.no_grad():
                    # Encode text
                    prompt_embeds_list = []
                    for tokenizer, text_encoder in zip([tokenizer_one, tokenizer_two], [text_encoder_one, text_encoder_two]):
                        text_inputs = tokenizer(
                            [prompt],
                            padding="max_length",
                            max_length=tokenizer.model_max_length,
                            truncation=True,
                            return_tensors="pt",
                        )
                        text_input_ids = text_inputs.input_ids
                        prompt_embeds = text_encoder(
                            text_input_ids.to(text_encoder.device),
                            output_hidden_states=True,
                        )
                        pooled_prompt_embeds = prompt_embeds[0]
                        prompt_embeds = prompt_embeds.hidden_states[-2]
                        prompt_embeds_list.append(prompt_embeds)
                    
                    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
                    prompt_embeds = prompt_embeds.squeeze(0)
                    pooled_prompt_embeds = pooled_prompt_embeds.squeeze(0)
                
                # VAE encoding
                with torch.no_grad():
                    

                    image_tensor_clamped = torch.clamp(image_tensor, -1.0, 1.0)
                    pixel_values = image_tensor_clamped.unsqueeze(0).to(self.vae.device, dtype=self.vae.dtype)
                    
                    # Use VAE encoding
                    latent_dist = self.vae.encode(pixel_values).latent_dist
                    model_input = latent_dist.sample()
                    model_input = model_input * self.vae.config.scaling_factor
                    
                    # Check and fix NaN (silent)
                    if torch.isnan(model_input).any() or torch.isinf(model_input).any():
                        # If sampling has issues, use mean instead
                        model_input = latent_dist.mean * self.vae.config.scaling_factor
                        if torch.isnan(model_input).any() or torch.isinf(model_input).any():
                            # If still has issues, fill with zeros
                            model_input = torch.zeros_like(model_input)
                    
                    model_input = model_input.squeeze(0).cpu()
                
                # CTR-based reward model doesn't need text encoding
                # The reward is calculated using image features only
                
                processed_batch.append({
                    "model_input": model_input,
                    "prompt_embeds": prompt_embeds.cpu(),
                    "pooled_prompt_embeds": pooled_prompt_embeds.cpu(),
                    "original_sizes": original_size,
                    "crop_top_lefts": crop_top_left,
                })
            
            # Stack all samples
            model_input = torch.stack([item["model_input"] for item in processed_batch])
            prompt_embeds = torch.stack([item["prompt_embeds"] for item in processed_batch])
            pooled_prompt_embeds = torch.stack([item["pooled_prompt_embeds"] for item in processed_batch])
            original_sizes = [item["original_sizes"] for item in processed_batch]
            crop_top_lefts = [item["crop_top_lefts"] for item in processed_batch]
            
            result.update({
                "model_input": model_input,
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds,
                "original_sizes": original_sizes,
                "crop_top_lefts": crop_top_lefts,
            })
            return result
            

        # DataLoaders creation:
        self.train_dataset = train_dataset  # Save dataset for later use
        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=args.train_batch_size,
            num_workers=args.dataloader_num_workers,
        )

       
        
    def _set_train_parameters(self):
        """Set up trainable parameters for IP-Adapter only training"""
        ip_params = []

        # 1) All IP-Attention injection layers (to_k_ip / to_v_ip)
        for name, module in self.unet.named_modules():
            if name.endswith("to_k_ip") or name.endswith("to_v_ip"):
                for param in module.parameters():
                    param.requires_grad_(True)
                    ip_params.append(param)
        
        adapter_module_params = sum(p.numel() for p in ip_params)
        logger.info(f"to_k_ip/to_v_ip trainable params: {adapter_module_params:,}")
        
        # 2) Image projection module: Diffusers attaches it to encoder_hid_proj
        if hasattr(self.unet, "encoder_hid_proj") and self.unet.encoder_hid_proj is not None:
            for param in self.unet.encoder_hid_proj.parameters():
                param.requires_grad_(True)
                ip_params.append(param)
            image_proj_model_params = sum(p.numel() for p in self.unet.encoder_hid_proj.parameters())
            logger.info(f"encoder_hid_proj trainable params: {image_proj_model_params:,}")
        else:
            image_proj_model_params = 0

        total_params = sum(p.numel() for p in ip_params)
        logger.info(f"Total trainable parameters: {total_params:,}")
        
        # Calculate total parameter count and ratio
        all_param = sum(p.numel() for p in self.unet.parameters())
        logger.info(f"Total UNet parameters: {all_param:,}")
        logger.info(f"Adapter %: {adapter_module_params/all_param*100:.2f}%")
        logger.info(f"Image projection %: {image_proj_model_params/all_param*100:.2f}%")
        logger.info(f"Total trainable %: {total_params/all_param*100:.2f}%")
        
        self.trainable_params = ip_params
    
    def save_trainable_params(self, path: str):
        """
        Only save trainable parameters (IP-Adapter related)
        Format consistent with Image_reward_trainer for easy load_model loading
        """
        trainable_state_dict = {}
        
        # Save encoder_hid_proj parameters
        if hasattr(self.unet, "encoder_hid_proj") and self.unet.encoder_hid_proj is not None:
            trainable_state_dict['image_proj_state_dict'] = self.unet.encoder_hid_proj.state_dict()
        
        # Save IP-Adapter related parameters (to_k_ip, to_v_ip)
        ip_adapter_state_dict = {}
        for name, module in self.unet.named_modules():
            if name.endswith("to_k_ip") or name.endswith("to_v_ip"):
                ip_adapter_state_dict[name] = module.state_dict()
        
        trainable_state_dict['ip_adapter_state_dict'] = ip_adapter_state_dict
        
        # Verify saved parameter count
        saved_params = 0
        if 'image_proj_state_dict' in trainable_state_dict:
            saved_params += sum(p.numel() for p in trainable_state_dict['image_proj_state_dict'].values())
        saved_params += sum(p.numel() for module_dict in trainable_state_dict['ip_adapter_state_dict'].values() 
                          for p in module_dict.values())
        trainable_params_count = sum(p.numel() for p in self.trainable_params if p.requires_grad)
        
        if saved_params != trainable_params_count:
            raise RuntimeError(f"Saved parameter count ({saved_params:,}) vs training parameter count ({trainable_params_count:,}) mismatch!")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(trainable_state_dict, path)
        logger.info(f"Save only trainable parameters to: {path} (parameter count: {saved_params:,})")
        
        return saved_params
    
    def _save_trainable_checkpoint(self, args, path: str):
        """
        Save the trainable (IP-Adapter) parameters, using the EMA weights when --use_ema is set.
        
        save_trainable_params() always reads from self.unet, so the EMA weights have to be copied
        in first. The live training weights are stashed with store() and put back with restore(),
        because checkpointing happens mid-training (every --checkpointing_steps).
        """
        if args.use_ema:
            self.ema_unet.store(self.unet.parameters())
            self.ema_unet.copy_to(self.unet.parameters())
        try:
            self.save_trainable_params(path)
        finally:
            if args.use_ema:
                self.ema_unet.restore(self.unet.parameters())
    
    
    def _compute_time_ids(self, resolution, original_size, crops_coords_top_left):
        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        target_size = (resolution, resolution)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = add_time_ids.to(self.accelerator.device, dtype=self.weight_dtype)
        return add_time_ids
        
    # Function for unwrapping if torch.compile() was used in accelerate.
    def _unwrap_model(self, model):
        model = self.accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def _extract_image_features_from_tensor(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extract CLIP features from image tensor for CTR-based reward model
        
        Args:
            image_tensor: RGB image tensor (batch_size, 3, H, W), range [0, 1]
            
        Returns:
            features: Image feature vector (batch_size, feature_dim)
        """
        # Initialize CLIP model if not already done
        if not hasattr(self, 'clip_vision_model'):
            from transformers import CLIPVisionModelWithProjection, CLIPProcessor
            
            clip_path = self.diffusion_clip_path #getattr(self.'diffusion_clip_path', 'openai/clip-vit-large-patch14')
            
            self.clip_vision_model = CLIPVisionModelWithProjection.from_pretrained(clip_path).to(self.accelerator.device)
            self.clip_processor = CLIPProcessor.from_pretrained(clip_path)
            self.clip_vision_model.eval()
            
            logger.info(f"Initialized CLIP model: {clip_path}")
        
        # Ensure image is in correct range [0, 1]
        image_tensor = torch.clamp(image_tensor, 0.0, 1.0)
        
        # Preprocess for CLIP
        def _transform():
            return transforms.Compose([
                transforms.Resize(224, interpolation=BICUBIC),
                transforms.CenterCrop(224),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        
        rm_preprocess = _transform()
        image_tensor_normalized = rm_preprocess(image_tensor)
        
        # Extract features (preserve gradients)
        image_features = self.clip_vision_model(image_tensor_normalized).image_embeds  # (batch_size, feature_dim)
        
        return image_features

    
    
    def set_retrival_attr(self, data_path, image_features, uncertain_model, near_images=None, near_images_sim=None, image_paths=None, top_k=3):
        """
        Set attributes required for uncertain retrieval
        
        Args:
            data_path: Data path
            image_features: Image features tensor
            uncertain_model: Model for computing uncertainty (reward model)
            near_images: Semantic neighbours of each image (loaded from near_images.npy if None)
            near_images_sim: Semantic similarity of those neighbours (loaded from near_images_sim.npy if None)
            image_paths: Image path list (loaded from image_path.pkl if None)
            top_k: Number of reference images retrieved per target image
        """

        
        self.data_path = data_path
        self.image_features = image_features
        self.uncertain_model = uncertain_model
        self.top_k = top_k
        self.device = self.accelerator.device
        
        # If not provided, load preprocessed data from file
        if near_images is None:
            self.near_images = np.load(os.path.join(self.data_path, 'near_images.npy'))
            logger.info("✓ Loaded near_images from file")
        else:
            self.near_images = near_images
        
            
        if near_images_sim is None:
            self.near_images_sim = np.load(os.path.join(self.data_path, 'near_images_sim.npy'))
            logger.info("✓ Loaded near_images_sim from file")
        else:
            self.near_images_sim = near_images_sim
        
            
        
        if image_paths is None:
            with open(os.path.join(self.data_path, "image_path.pkl"), "rb") as file:
                self.image_paths = pickle.load(file)
            logger.info("✓ Loaded image_paths from file")
        else:
            self.image_paths = image_paths
        
        
        # Set uncertain_model to eval mode
        if self.uncertain_model is not None:
            self.uncertain_model.eval()
        
        logger.info(f"✓ Retrieval attributes set: top_k={top_k}")
    
    def uncertain_retrival(self, his_iids, target_iid, process_image=True):
        """
        Semantic-aware Low-uncertainty Retrieval module.
        
        Thin wrapper around the shared implementation in data_utils, so that training and
        inference (models/gen_model.py) always rank candidates identically.
        
        Returns:
            process_image=True : (batch_retrival_images, batch_retrival_iids, batch_ctr, batch_sim)
            process_image=False: (batch_retrival_iids, batch_ctr, batch_sim)
        """
        return data_utils.uncertainty_aware_retrieval(
            self.near_images, self.near_images_sim, self.image_features, self.uncertain_model,
            self.image_paths, self.data_path, his_iids, target_iid, self.top_k, self.device,
            process_image=process_image,
        )
    
    def get_image_embed(self, input_images, drop_image_embeds=None):

        his_images = input_images 
        bsz = len(his_images)
        
        # Pre-allocate tensors to avoid dynamic growth
        batch_his_embeds = []
        
        # Use torch.no_grad() to reduce memory usage
        with torch.no_grad():
            for i in range(bsz):
                # Reference images come from uncertainty-aware retrieval (see uncertain_retrival),
                # not from the user's history images.
                ip_adapter_images = [his_images[i]]
            
                ip_embeds = self.diffusion_model.prepare_ip_adapter_image_embeds(
                    ip_adapter_image=ip_adapter_images,
                    ip_adapter_image_embeds=None,
                    device=self.accelerator.device,
                    #device=self.device,
                    num_images_per_prompt=1,
                    # Must stay False. The UNet is called without CFG here: prompt_embeds has
                    # batch `bsz`, so image_embeds must have batch `bsz` too. With True,
                    # diffusers prepends the negative (all-zero) image embeddings and returns
                    # batch `2*bsz` — and that does NOT raise. IPAdapterAttnProcessor2_0 reshapes
                    # with `ip_key.view(batch_size, -1, heads, head_dim)` where `batch_size` comes
                    # from *prompt_embeds*, so the extra `2*bsz` rows are silently folded into
                    # `2*num_image` tokens: the first half of the batch is then conditioned on
                    # pure zeros (IP-Adapter effectively disabled) and the second half on another
                    # sample's reference images.
                    do_classifier_free_guidance=False,
                )
                
                # Apply drop mechanism - refer to tutorial_train_sdxl.py
                if drop_image_embeds is not None and i < len(drop_image_embeds):
                    if drop_image_embeds[i] == 1:
                        # Set image embeddings to zero
                        for j in range(len(ip_embeds)):
                            ip_embeds[j] = torch.zeros_like(ip_embeds[j])
                
                
                tmp,num_image,num_token,embed_dim = ip_embeds[0].shape

                # Guard against silently re-introducing the CFG batch doubling above: the
                # downstream reshape assumes `tmp == 1`, and the mistake is invisible because
                # nothing downstream ever checks the batch size.
                if tmp != 1:
                    raise ValueError(
                        f"image_embeds batch dimension is {tmp}, expected 1. This happens when "
                        f"do_classifier_free_guidance=True in prepare_ip_adapter_image_embeds: "
                        f"the uncond embeddings are prepended, and the IP-Adapter attention "
                        f"processor then silently folds the doubled batch into extra tokens "
                        f"instead of failing. image_embeds must match prompt_embeds' batch size."
                    )

                batch_his_embeds.append(ip_embeds[0].unsqueeze(1))
 
                torch.cuda.empty_cache()
 
        # Use more efficient memory operations
        batch_his_embeds = torch.cat(batch_his_embeds,dim=1).reshape(tmp*bsz,num_image,num_token,embed_dim)
        return [batch_his_embeds]


    def _extract_image_features(self, pil_image):
        """
        Extract feature vector from PIL image (using same CLIP method as data_utils.py)
        
        Args:
            pil_image: PIL image
            
        Returns:
            features: Image feature vector (1, feature_dim)
        """
        
        # Initialize CLIP model (if not already initialized)
        if not hasattr(self, 'clip_vision_model'):
            from transformers import CLIPVisionModelWithProjection, CLIPProcessor
            
            # Use same model path as data_utils.py
            clip_path = getattr(self.args, 'diffusion_clip_path', 'openai/clip-vit-large-patch14')
            
            self.clip_vision_model = CLIPVisionModelWithProjection.from_pretrained(clip_path).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(clip_path)
            self.clip_vision_model.eval()
            
            self.logger.info(f"Initialize CLIP model: {clip_path}")
        
        # Preprocess image (consistent with data_utils.py)
        inputs = self.clip_processor(images=[pil_image], return_tensors="pt", padding=True).to(self.device)
        
        # Extract features
        with torch.no_grad():
            image_features = self.clip_vision_model(inputs.pixel_values).image_embeds  # (1, feature_dim)
            
            
            return image_features  # (1, feature_dim)
    
    def _pretrain_loss(self, args, batch) -> torch.Tensor:
        # Sample noise that we'll add to the latents
        #print("batch",batch.keys())
        model_input = batch["model_input"].to(self.accelerator.device)
        
        # Check for NaN in model_input
        if torch.isnan(model_input).any():
            raise RuntimeError("NaN detected in model_input! Check your data and model for numerical stability issues.")
        
        noise = torch.randn_like(model_input)

        bsz = model_input.shape[0]
        timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
        )

        # Add noise to the model input according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_model_input = self.noise_scheduler.add_noise(model_input, noise, timesteps)

        add_time_ids = torch.cat(
            [self._compute_time_ids(args.resolution, s, c) for s, c in zip(batch["original_sizes"], batch["crop_top_lefts"])]
        )

        # Predict the noise residual
        unet_added_conditions = {"time_ids": add_time_ids}
        prompt_embeds = batch["prompt_embeds"].to(self.accelerator.device)
        pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(self.accelerator.device)
        unet_added_conditions.update({"text_embeds": pooled_prompt_embeds})
        
        
        unet_added_conditions.update({"image_embeds": self.image_embeds}) 
        model_pred = self.unet(
            noisy_model_input,
            timesteps,
            prompt_embeds,
            added_cond_kwargs=unet_added_conditions,
            return_dict=False,
        )[0]

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(model_input, noise, timesteps)
        elif self.noise_scheduler.config.prediction_type == "sample":
            # We set the target to latents here, but the model_pred will return the noise sample prediction.
            target = model_input
            # We will have to subtract the noise residual from the prediction to get the target sample.
            model_pred = model_pred - noise
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        
        # Check for NaN in loss
        if torch.isnan(loss):
            raise RuntimeError("NaN detected in pre_loss! Check your model, learning rate, and input data for numerical stability issues.")

        # Backpropagate
        self.accelerator.backward(loss)
        if self.accelerator.sync_gradients:
            # Only clip gradients for trainable parameters
            self.accelerator.clip_grad_norm_(self.trainable_params, args.max_grad_norm)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        
        return loss
    def _reward_loss(self, args, batch) -> torch.Tensor:
        add_time_ids = torch.cat(
            [self._compute_time_ids(args.resolution, s, c) for s, c in zip(batch["original_sizes"], batch["crop_top_lefts"])]
        )

        # Predict the noise residual
        unet_added_conditions = {"time_ids": add_time_ids}
        prompt_embeds = batch["prompt_embeds"].to(self.accelerator.device)
        pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(self.accelerator.device)
        unet_added_conditions.update({"text_embeds": pooled_prompt_embeds})
        
        # Use uncertain_retrieval to get retrieved images
        retrival_images, _, _, _ = self.uncertain_retrival(
                batch["sparse_iids"], batch["iids"], process_image=True
        )
        self.image_embeds = self.get_image_embed(retrival_images)        
        unet_added_conditions.update({"image_embeds": self.image_embeds})

        # Latent spatial size must follow the training resolution (the VAE downsamples by 8x).
        # A hard-coded 128x128 would decode to a 1024x1024 image and silently disagree with
        # --resolution (e.g. 512 -> 64x64), wasting 4x memory and mismatching the fine-tuning
        # resolution actually used by _pretrain_loss.
        bs = prompt_embeds.shape[0]
        latent_size = args.resolution // 8
        latents = torch.randn((bs, 4, latent_size, latent_size), device=self.accelerator.device)
        
        timesteps = self.noise_scheduler.timesteps

        # set_timesteps(num_inference_steps=40) is called immediately before this method, so
        # timesteps has 40 entries and indices 30-39 land in the last quarter of the denoising
        # trajectory: late enough for the predicted x0 to be nearly clean, early enough to keep
        # a useful gradient signal. The timestep is sampled (as in ImageReward/ReFL) rather than
        # fixed, so the reward model is not overfitted to a single noise level.
        mid_timestep = random.randint(30, 39)  # requires len(timesteps) > mid_timestep

        for i, t in enumerate(timesteps[:mid_timestep]):
            with torch.no_grad():
                latent_model_input = latents
                latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, t)
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    prompt_embeds,
                    added_cond_kwargs=unet_added_conditions,
                ).sample
                latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample
        
        latent_model_input = latents
        latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, timesteps[mid_timestep])
        noise_pred = self.unet(
            latent_model_input,
            timesteps[mid_timestep],
            prompt_embeds,
            added_cond_kwargs=unet_added_conditions,
        ).sample
        pred_original_sample = self.noise_scheduler.step(noise_pred, timesteps[mid_timestep], latents).pred_original_sample.to(self.weight_dtype)
        
        pred_original_sample = 1 / self.vae.config.scaling_factor * pred_original_sample

        
        image = self.vae.decode(pred_original_sample).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        
        
        # Calculate CTR-based reward
        # Extract CLIP features from generated image
        generated_features = self._extract_image_features_from_tensor(image)  # (batch_size, 768)
        
        batch_size = image.shape[0]
        rewards = []
        
        for i in range(batch_size):
            # Get user history features (up to args.max_seq_len items)
            his_iids = batch['his_iids'][i]
            if len(his_iids.shape) == 0:  # scalar
                his_iids = his_iids.unsqueeze(0)
            # Truncate to the same sequence length used by the Stage 1 preference model.
            his_iids = his_iids[:args.max_seq_len]
            
            user_hist_features = self.image_features[his_iids].unsqueeze(0)  # (1, seq_len, 768)
            gen_features = generated_features[i:i+1]  # (1, 768)
            
            # Get CTR prediction
            click_prob, _ = self.reward_model(user_hist_features, gen_features)
            
            # Optional: compare with original image CTR
            if 'iids' in batch:
                with torch.no_grad():
                    target_features = self.image_features[batch['iids'][i]].unsqueeze(0)
                    ori_ctr, _ = self.reward_model(user_hist_features, target_features)
                reward = click_prob - ori_ctr  # Relative reward
            else:
                reward = click_prob  # Absolute reward
            
            #reward = click_prob

            rewards.append(reward)
        
        rewards = torch.cat(rewards)
        # Maximize CTR: minimize negative reward
        loss = (-rewards + 1.0).mean() #[]

        # Backpropagate. L = alpha * L_r + L_p (paper Eq. 11): alpha scales the reward term
        # only; L_p is back-propagated in its own optimizer step (see train()).
        self.accelerator.backward(args.alpha * loss)
        if self.accelerator.sync_gradients:
            # Only clip gradients for trainable parameters
            self.accelerator.clip_grad_norm_(self.trainable_params, args.max_grad_norm)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()

        return loss
    

    def train(self, args):
        # Train!
        total_batch_size = args.train_batch_size * self.accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        logger.info(f" Apply pre-trained loss? {args.apply_pre_loss}")
        logger.info(f" Apply reward loss? {args.apply_reward_loss}")
        
        global_step = 0
        first_epoch = 0



        progress_bar = tqdm(
            range(0, args.max_train_steps),
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not self.accelerator.is_local_main_process,
        )
       

        for epoch in range(first_epoch, args.num_train_epochs):
            self.unet.train()

            # Total loss per process
            loss = 0.0

            # Pre-trained loss among all processes
            pre_avg_loss = 0.0

            # Reward loss among all processes
            reward_avg_loss = 0.0
            
            # Total loss among all processes
            train_loss = 0.0
            
            # Accumulated losses for logging (over gradient_accumulation_steps)
            
            for step, batch in enumerate(self.train_dataloader):
                logs = {}

                
                if args.resume_from_checkpoint and epoch == first_epoch:
                    if step % args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue
                
                
                        
                # Reward loss
                if args.apply_reward_loss:
                    self.noise_scheduler.set_timesteps(num_inference_steps=40, 
                                                        device=self.accelerator.device)

                    with self.accelerator.accumulate(self.unet):
                        reward_loss = self._reward_loss(args, batch)
                        loss += reward_loss.detach().item()

                        # Gather the losses across all processes for logging (if we use distributed training).
                        reward_avg_loss = self.accelerator.gather(reward_loss.repeat(args.train_batch_size)).mean()
                        train_loss += reward_avg_loss.item() / args.gradient_accumulation_steps

                    logs.update({"reward_loss": reward_loss.detach().item()})
                # Pre-train loss (NOTE: It uses same dataset as reward loss. Fix it later)
                if args.apply_pre_loss:
                    # Use the scheduler with default timesteps
                    assert args.apply_reward_loss, "apply_reward_loss must be True when apply_pre_loss is True due to the self.image_embeds"
                    with self.accelerator.accumulate(self.unet):
                        # Sample noise that we'll add to the latents
                        pre_loss = self._pretrain_loss(args, batch)
                        loss += pre_loss.detach().item()
                        
                        # Gather the losses across all processes for logging (if we use distributed training).
                        pre_avg_loss = self.accelerator.gather(pre_loss.repeat(args.train_batch_size)).mean()
                        train_loss += pre_avg_loss.item() / args.gradient_accumulation_steps
                                                
                    logs.update({"pre_loss": pre_loss.detach().item()})
                # Checks if the self.accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    if args.use_ema:
                        self.ema_unet.step(self.unet.parameters())
                    progress_bar.update(1)
                    global_step += 1
                    self.accelerator.log({"train_loss": train_loss, 
                                          "pre_loss": pre_avg_loss, 
                                          "reward_loss": reward_avg_loss}, step=global_step)

                    # Store the model weights
                    if self.accelerator.is_main_process:
                        if global_step % args.checkpointing_steps == 0:
                            save_path = os.path.join(args.output_dir,str(global_step))
                            
                            
                            # Save only trainable parameters by default (lightweight)
                            os.makedirs(save_path, exist_ok=True)
                            checkpoint_path = os.path.join(save_path, "checkpoint.pt")
                            
                            self._save_trainable_checkpoint(args, checkpoint_path)
                            
                            
  
                    
                    train_loss = 0.0

                logs.update({"step_loss": loss, "lr": self.lr_scheduler.get_last_lr()[0]})
                progress_bar.set_postfix(**logs)

                loss = 0.0

                if global_step >= args.max_train_steps:
                    # Make sure the final weights reach disk even when max_train_steps is not a
                    # multiple of checkpointing_steps (otherwise no checkpoint is ever written).
                    if self.accelerator.is_main_process and global_step % args.checkpointing_steps != 0:
                        save_path = os.path.join(args.output_dir, str(global_step))
                        os.makedirs(save_path, exist_ok=True)
                        self._save_trainable_checkpoint(args, os.path.join(save_path, "checkpoint.pt"))
                    break

        self.accelerator.wait_for_everyone()
        # Skip saving full pipeline (VAE/UNet) at end: VAE is unchanged from pretrained,
        # and inference loads pretrained base + checkpoint.pt only. Saves disk and time.
        self.accelerator.end_training()
   