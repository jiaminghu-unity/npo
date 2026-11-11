#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import io
import logging
import math
import os
import random
import shutil
import sys
from pathlib import Path

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
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel,     StableDiffusionXLPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.import_utils import is_xformers_available
if is_wandb_available():
    import wandb

    
    
## SDXL
import functools
import gc
from torchvision.transforms.functional import crop
from transformers import AutoTokenizer, PretrainedConfig



# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.20.0")

logger = get_logger(__name__, log_level="INFO")

DATASET_NAME_MAPPING = {
    "yuvalkirstain/pickapic_v1": ("jpg_0", "jpg_1", "label_0", "caption"),
    "yuvalkirstain/pickapic_v2": ("jpg_0", "jpg_1", "label_0", "caption"),
}

        
def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
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
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="caption",
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
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None,
                        # was random for submission, need to test that not distributing same noise etc across devices
                        help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help=(
            "The resolution for input images, all the images in the dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--random_crop",
        default=False,
        action="store_true",
        help=(
            "If set the images will be randomly"
            " cropped (instead of center). The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--no_hflip",
        action="store_true",
        help="whether to supress horizontal flipping",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=2000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
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
        default=1e-8,
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
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_adafactor", action="store_true", help="Whether or not to use adafactor (should save mem)"
    )
    # Bram Note: Haven't looked @ this yet
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
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=50.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
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
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
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
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default='latest',
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="tuning",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    ## SDXL
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument("--sdxl", action='store_true', help="Train sdxl")
    
    ## DPO
    parser.add_argument("--sft", action='store_true', help="Run Supervised Fine-Tuning instead of Direct Preference Optimization")
    parser.add_argument("--beta_dpo", type=float, default=5000, help="The beta DPO temperature controlling strength of KL penalty")
    parser.add_argument(
        "--hard_skip_resume", action="store_true", help="Load weights etc. but don't iter through loader for loader resume, useful b/c resume takes forever"
    )
    parser.add_argument(
        "--unet_init", type=str, default='', help="Initialize start of run from unet (not compatible w/ checkpoint load)"
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0.2,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--split", type=str, default='train', help="Datasplit"
    )
    parser.add_argument(
        "--choice_model", type=str, default='', help="Model to use for ranking (override dataset PS label_0/1). choices: aes, clip, hps, pickscore"
    )
    parser.add_argument(
        "--dreamlike_pairs_only", action="store_true", help="Only train on pairs where both generations are from dreamlike"
    )


    ## Train method selector (dpo/sft/nash/nash_sft/online_dpo)
    parser.add_argument(
    "--train_method",
    type=str,
    default=None,
    choices=["dpo", "sft", "nash", "nash_sft", "online_dpo"],
    help="Training objective to use. If omitted, defaults to 'sft' when --sft is set, else 'dpo'.",
    )
    
    ## INPO/Nash (online) hyperparams # TODO SD1.5 DPO best beta is 2000
    parser.add_argument("--nash_tau", type=float, default=0.1, help="INPO τ (KL weight toward π_ref) used in h_t.")
    parser.add_argument("--nash_eta", type=float, default=1.0, help="INPO η (OMD stepsize) used in h_t and the 1/(2η) target.")
    parser.add_argument("--nash_sft_lambda", type=float, default=0.1, help="Weight for SFT term when --train_method=nash_sft.")
    # TODO
    parser.add_argument(
    "--nash_prev_update_steps",
    type=int,
    default=1,
    help="How often (in optimizer steps) to refresh π_t (prev_unet) from the current unet. This approximates outer-iteration updates.",
    )
    parser.add_argument(
        "--nash_prev_hard_update",
        action="store_true",
        help=(
            "If set, refresh π_t (prev_unet) by copying weights from the current unet each refresh step. "
            "Default (unset) uses soft update: theta_prev <- mu*theta_prev + (1-mu)*theta_current."
        ),
    )
    parser.add_argument("--nash_num_inference_steps", type=int, default=10, help="Inference steps when sampling online images for Nash.")
    parser.add_argument("--nash_guidance_scale", type=float, default=7.5, help="CFG guidance scale for online sampling (non-SDXL path).")
    parser.add_argument("--nash_ddim_eta", type=float, default=1.0, help="DDIM eta; >0 introduces stochasticity so shared init noise can still yield diverse samples.")
    parser.add_argument(
    "--nash_oracles",
    type=str,
    default="pickscore,hps,clip,aes,im",
    help="Comma-separated list of preference oracles to ensemble for online labels. Supported: pickscore,hps,clip,aes",
    )
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    ## SDXL
    if args.sdxl:
        print("Running SDXL")
    if args.resolution is None:
        if args.sdxl:
            args.resolution = 1024
        else:
            args.resolution = 512
            
    # Resolve train method
    if args.train_method is None:
        args.train_method = "sft" if args.sft else "dpo"
    else:
        # Backward-compat: --sft implies train_method=sft unless explicitly overridden
        if args.sft and args.train_method != "sft":
            print(f"[WARN] Both --sft and --train_method={args.train_method} set; honoring --train_method.")
    if args.train_method in ("nash", "nash_sft", "online_dpo") and args.choice_model:
        raise ValueError("--choice_model (AIF flipping) is not used for Nash/nash_sft/online_dpo; labels are produced by --nash_oracles.")
    return args



# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt_sdxl(batch, text_encoders, tokenizers, proportion_empty_prompts, caption_column, is_train=True):
    prompt_embeds_list = []
    prompt_batch = batch[caption_column]

    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to('cuda'),
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return {"prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled_prompt_embeds}




def main():
    
    args = parse_args()
    
    #### START ACCELERATOR BOILERPLATE ###
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index) # added in + term, untested

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    ### END ACCELERATOR BOILERPLATE
    
    
    ### START DIFFUSION BOILERPLATE ###
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, 
                                                    subfolder="scheduler")
    def enforce_zero_terminal_snr(scheduler):
        # Modified from https://arxiv.org/pdf/2305.08891.pdf
        # Turbo needs zero terminal SNR to truly learn from noise
        # Turbo: https://static1.squarespace.com/static/6213c340453c3f502425776e/t/65663480a92fba51d0e1023f/1701197769659/adversarial_diffusion_distillation.pdf
        # Convert betas to alphas_bar_sqrt
        alphas = 1 - scheduler.betas
        alphas_bar = alphas.cumprod(0)
        alphas_bar_sqrt = alphas_bar.sqrt()

        # Store old values.
        alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
        alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
        # Shift so last timestep is zero.
        alphas_bar_sqrt -= alphas_bar_sqrt_T
        # Scale so first timestep is back to old value.
        alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

        alphas_bar = alphas_bar_sqrt ** 2
        alphas = alphas_bar[1:] / alphas_bar[:-1]
        alphas = torch.cat([alphas_bar[0:1], alphas])
    
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        scheduler.alphas_cumprod = alphas_cumprod
        return 
    if 'turbo' in args.pretrained_model_name_or_path:
        enforce_zero_terminal_snr(noise_scheduler)
    
    # SDXL has two text encoders
    if args.sdxl:
        # Load the tokenizers
        if args.pretrained_model_name_or_path=="stabilityai/stable-diffusion-xl-refiner-1.0":
            tokenizer_and_encoder_name = "stabilityai/stable-diffusion-xl-base-1.0"
        else:
            tokenizer_and_encoder_name = args.pretrained_model_name_or_path
        tokenizer_one = AutoTokenizer.from_pretrained(
            tokenizer_and_encoder_name, subfolder="tokenizer", revision=args.revision, use_fast=False
        )
        tokenizer_two = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision, use_fast=False
        )
    else:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
        )

    # Not sure if we're hitting this at all
    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    
    # BRAM NOTE: We're not using deepspeed currently so not sure it'll work. Could be good to add though!
    # 
    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        # SDXL has two text encoders
        if args.sdxl:
            # import correct text encoder classes
            text_encoder_cls_one = import_model_class_from_model_name_or_path(
               tokenizer_and_encoder_name, args.revision
            )
            text_encoder_cls_two = import_model_class_from_model_name_or_path(
                tokenizer_and_encoder_name, args.revision, subfolder="text_encoder_2"
            )
            text_encoder_one = text_encoder_cls_one.from_pretrained(
                tokenizer_and_encoder_name, subfolder="text_encoder", revision=args.revision
            )
            text_encoder_two = text_encoder_cls_two.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision
            )
            if args.pretrained_model_name_or_path=="stabilityai/stable-diffusion-xl-refiner-1.0":
                text_encoders = [text_encoder_two]
                tokenizers = [tokenizer_two]
            else:
                text_encoders = [text_encoder_one, text_encoder_two]
                tokenizers = [tokenizer_one, tokenizer_two]
        else:
            text_encoder = CLIPTextModel.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
            )
        # Can custom-select VAE (used in original SDXL tuning)
        vae_path = (
            args.pretrained_model_name_or_path
            if args.pretrained_vae_model_name_or_path is None
            else args.pretrained_vae_model_name_or_path
        )
        vae = AutoencoderKL.from_pretrained(
            vae_path, subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None, revision=args.revision
        )
        # clone of model
        ref_unet = UNet2DConditionModel.from_pretrained(
            args.unet_init if args.unet_init else args.pretrained_model_name_or_path,
            subfolder="unet", revision=args.revision
        )
        # newly added: nash's previous model
        if args.train_method in ('nash', 'nash_sft', 'online_dpo'):
            prev_unet = UNet2DConditionModel.from_pretrained(
            args.unet_init if args.unet_init else args.pretrained_model_name_or_path,
            subfolder="unet", revision=args.revision
        )
    if args.unet_init:
        print("Initializing unet from", args.unet_init)
    unet = UNet2DConditionModel.from_pretrained(
        args.unet_init if args.unet_init else args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    # Freeze vae, text_encoder(s), reference unet
    vae.requires_grad_(False)
    if args.sdxl:
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)
    else:
        text_encoder.requires_grad_(False)
    if args.train_method in ('dpo', 'nash', 'nash_sft', 'online_dpo'): ref_unet.requires_grad_(False)
    if args.train_method in ('nash', 'nash_sft', 'online_dpo'): prev_unet.requires_grad_(False)

    # xformers efficient attention
    if is_xformers_available():
        import xformers

        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            logger.warn(
                "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
            )
        unet.enable_xformers_memory_efficient_attention()
    else:
        raise ValueError("xformers is not available. Make sure it is installed correctly")

    # BRAM NOTE: We're using >=0.16.0. Below was a bit of a bug hive. I hacked around it, but ideally ref_unet wouldn't
    # be getting passed here
    # 
    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            
            if len(models) > 1:
                assert args.train_method in ('dpo', 'nash', 'nash_sft', 'online_dpo') # 2nd model is just ref_unet in DPO case
            models_to_save = models[:1]
            for i, model in enumerate(models_to_save):
                model.save_pretrained(os.path.join(output_dir, "unet"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):

            if len(models) > 1:
                assert args.train_method in ('dpo', 'nash', 'nash_sft', 'online_dpo') # 2nd model is just ref_unet in DPO case
            models_to_load = models[:1]
            for i in range(len(models_to_load)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing or args.sdxl: #  (args.sdxl and ('turbo' not in args.pretrained_model_name_or_path) ):
        print("Enabling gradient checkpointing, either because you asked for this or because you're using SDXL")
        unet.enable_gradient_checkpointing()

    # Bram Note: haven't touched
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    if args.use_adafactor or args.sdxl:
        print("Using Adafactor either because you asked for it or you're using SDXL")
        optimizer = transformers.Adafactor(unet.parameters(),
                                           lr=args.learning_rate,
                                           weight_decay=args.adam_weight_decay,
                                           clip_threshold=1.0,
                                           scale_parameter=False,
                                          relative_step=False)
    else:
        optimizer = torch.optim.AdamW(
            unet.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

        
        
        
    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            data_dir=args.train_data_dir,
            token="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",
        )
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files[args.split] = os.path.join(args.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset[args.split].column_names

    # 6. Get the column names for input/target.
    dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
    if 'pickapic' in args.dataset_name:
        pass
    elif args.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if random.random() < args.proportion_empty_prompts:
                captions.append("")
            elif isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(args.resolution) if args.random_crop else transforms.CenterCrop(args.resolution),
            transforms.Lambda(lambda x: x) if args.no_hflip else transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    
    ##### START BIG OLD DATASET BLOCK #####
    
    #### START PREPROCESSING/COLLATION ####
    if args.train_method == 'dpo':
        print("Ignoring image_column variable, reading from jpg_0 and jpg_1")
        def preprocess_train(examples):
            all_pixel_values = []
            for col_name in ['jpg_0', 'jpg_1']:
                images = [Image.open(io.BytesIO(im_bytes)).convert("RGB")
                            for im_bytes in examples[col_name]]
                pixel_values = [train_transforms(image) for image in images]
                all_pixel_values.append(pixel_values)
            # Double on channel dim, jpg_y then jpg_w
            im_tup_iterator = zip(*all_pixel_values)
            combined_pixel_values = []
            for im_tup, label_0 in zip(im_tup_iterator, examples['label_0']):
                if label_0==0 and (not args.choice_model): # don't want to flip things if using choice_model for AI feedback
                    im_tup = im_tup[::-1]
                combined_im = torch.cat(im_tup, dim=0) # no batch dim
                combined_pixel_values.append(combined_im)
            examples["pixel_values"] = combined_pixel_values
            # SDXL takes raw prompts
            if not args.sdxl: examples["input_ids"] = tokenize_captions(examples)
            return examples

        def collate_fn(examples):
            pixel_values = torch.stack([example["pixel_values"] for example in examples])
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
            return_d =  {"pixel_values": pixel_values}
            # SDXL takes raw prompts
            if args.sdxl:
                return_d["caption"] = [example["caption"] for example in examples]
            else:
                return_d["input_ids"] = torch.stack([example["input_ids"] for example in examples])
                
            if args.choice_model:
                # If using AIF then deliver image data for choice model to determine if should flip pixel values
                for k in ['jpg_0', 'jpg_1']:
                    return_d[k] = [Image.open(io.BytesIO( example[k])).convert("RGB")
                                   for example in examples]
                return_d["caption"] = [example["caption"] for example in examples] 
            return return_d
         
        if args.choice_model:
            # TODO: Fancy way of doing this?
            if args.choice_model == 'hps':
                from utils.hps_utils import Selector
            elif args.choice_model == 'clip':
                from utils.clip_utils import Selector
            elif args.choice_model == 'pickscore':
                from utils.pickscore_utils import Selector
            elif args.choice_model == 'aes':
                from utils.aes_utils import Selector
            selector = Selector('cpu' if args.sdxl else accelerator.device)

            def do_flip(jpg0, jpg1, prompt):
                scores = selector.score([jpg0, jpg1], prompt)
                return scores[1] > scores[0]
            def choice_model_says_flip(batch):
                assert len(batch['caption'])==1 # Can switch to iteration but not needed for nwo
                return do_flip(batch['jpg_0'][0], batch['jpg_1'][0], batch['caption'][0])
    elif args.train_method == 'sft':
        def preprocess_train(examples):
            if 'pickapic' in args.dataset_name:
                images = []
                # Probably cleaner way to do this iteration
                for im_0_bytes, im_1_bytes, label_0 in zip(examples['jpg_0'], examples['jpg_1'], examples['label_0']):
                    assert label_0 in (0, 1)
                    im_bytes = im_0_bytes if label_0==1 else im_1_bytes
                    images.append(Image.open(io.BytesIO(im_bytes)).convert("RGB"))
            else:
                images = [image.convert("RGB") for image in examples[image_column]]
            examples["pixel_values"] = [train_transforms(image) for image in images]
            if not args.sdxl: examples["input_ids"] = tokenize_captions(examples)
            return examples

        def collate_fn(examples):
            pixel_values = torch.stack([example["pixel_values"] for example in examples])
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
            return_d =  {"pixel_values": pixel_values}
            if args.sdxl:
                return_d["caption"] = [example["caption"] for example in examples]
            else:
                return_d["input_ids"] = torch.stack([example["input_ids"] for example in examples])
            return return_d
    elif args.train_method in ('nash', 'nash_sft', 'online_dpo'):
        # Online setting: dataset only needs prompts. We'll still return input_ids for SD1.5 conditioning.
        def preprocess_train(examples):
            # Keep raw caption (prompt) for online sampling
            if not args.sdxl:
                examples["input_ids"] = tokenize_captions(examples)
            return examples

        def collate_fn(examples):
            # We only need captions (and input_ids for SD1.5). No pixel_values from dataset.
            return_d = {"caption": [example["caption"] for example in examples]}
            if not args.sdxl:
                return_d["input_ids"] = torch.stack([example["input_ids"] for example in examples])
            return return_d

    #### END PREPROCESSING/COLLATION ####

    #### FOR NASH ONLY ####
    # Preference oracle(s) for online Nash: ensemble multiple scorers and take majority vote.
    nash_pick_winner = None
    if args.train_method in ('nash', 'nash_sft', 'online_dpo'):
        def nash_pick_winner_loser(candidate_latents, prompt: str, unet_judge, num_rank_trials: int = 10):
            """
            Return (winner_index, loser_index, score_dict) for candidate images represented by final latents.
            For each trial, sample one timestep t and noise epsilon, compute MSE((eps_pred - epsilon)^2) per image,
            rank by loss (smaller is better), and average ranks across trials.
            """
            if isinstance(candidate_latents, list):
                candidate_latents = torch.stack(candidate_latents, dim=0)
            if candidate_latents.ndim != 4:
                raise ValueError(f"Expected candidate_latents [K,C,H,W], got shape {tuple(candidate_latents.shape)}")

            latents = candidate_latents.to(device=accelerator.device, dtype=weight_dtype)
            num_candidates = latents.shape[0]
            if num_candidates < 2:
                raise ValueError("Need at least 2 candidates to pick winner/loser.")

            with torch.no_grad():
                text_in = tokenizer(
                    [prompt],
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.to(accelerator.device)
                encoder_hidden_states = text_encoder(text_in)[0].to(dtype=weight_dtype).repeat(num_candidates, 1, 1)

                ranks_accum = torch.zeros(num_candidates, device=accelerator.device, dtype=torch.float32)
                losses_per_trial = []

                with torch.no_grad(), torch.cuda.amp.autocast(dtype=weight_dtype):
                    for _ in range(num_rank_trials):
                        timesteps = torch.randint(
                            0,
                            noise_scheduler.config.num_train_timesteps,
                            (num_candidates,),
                            device=latents.device,
                        ).long()
                        if 'refiner' in args.pretrained_model_name_or_path:
                            timesteps = timesteps % 200
                        elif 'turbo' in args.pretrained_model_name_or_path:
                            timesteps_0_to_3 = timesteps % 4
                            timesteps = 250 * timesteps_0_to_3 + 249

                        target = torch.randn_like(latents)
                        noisy_latents = noise_scheduler.add_noise(latents, target, timesteps)
                        model_pred = unet_judge(
                            noisy_latents,
                            timesteps,
                            encoder_hidden_states=encoder_hidden_states,
                        ).sample
                        model_losses = (model_pred - target).pow(2).mean(dim=[1, 2, 3])
                        losses_per_trial.append(model_losses.detach().float())

                    order = torch.argsort(model_losses, descending=False)
                    ranks = torch.empty(num_candidates, device=latents.device, dtype=torch.float32)
                    ranks[order] = torch.arange(1, num_candidates + 1, device=latents.device, dtype=torch.float32)
                    ranks_accum += ranks

                avg_ranks = ranks_accum / float(num_rank_trials)
                winner_idx = int(torch.argmin(avg_ranks).item())
                loser_idx = int(torch.argmax(avg_ranks).item())

                mean_losses = torch.stack(losses_per_trial, dim=0).mean(dim=0)
                score_dict = {
                    "unet_model_loss_mean": mean_losses.detach().cpu().tolist(),
                    "unet_avg_rank": avg_ranks.detach().cpu().tolist(),
                }
            return winner_idx, loser_idx, score_dict
        
        def move_nash_oracles_device(device):
            # UNet-based judge shares the same model/device lifecycle as training/sampling.
            return None
 
    
    ### DATASET #####
    with accelerator.main_process_first():
        if 'pickapic' in args.dataset_name:
            # eliminate no-decisions (0.5-0.5 labels)
            orig_len = dataset[args.split].num_rows
            not_split_idx = [i for i,label_0 in enumerate(dataset[args.split]['label_0'])
                             if label_0 in (0,1) ]
            dataset[args.split] = dataset[args.split].select(not_split_idx)
            new_len = dataset[args.split].num_rows
            print(f"Eliminated {orig_len - new_len}/{orig_len} split decisions for Pick-a-pic")
            
            # Below if if want to train on just the Dreamlike vs dreamlike pairs
            if args.dreamlike_pairs_only:
                orig_len = dataset[args.split].num_rows
                dream_like_idx = [i for i,(m0,m1) in enumerate(zip(dataset[args.split]['model_0'],
                                                                   dataset[args.split]['model_1']))
                                  if ( ('dream' in m0) and ('dream' in m1) )]
                dataset[args.split] = dataset[args.split].select(dream_like_idx)
                new_len = dataset[args.split].num_rows
                print(f"Eliminated {orig_len - new_len}/{orig_len} non-dreamlike gens for Pick-a-pic")
                
        if args.max_train_samples is not None:
            dataset[args.split] = dataset[args.split].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset = dataset[args.split].with_transform(preprocess_train)

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=(args.split=='train'),
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True
    )
    ##### END BIG OLD DATASET BLOCK #####
    
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    
    #### START ACCELERATOR PREP ####
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

        
    # Move text_encode and vae to gpu and cast to weight_dtype
    # TODO: offload as necessary
    vae.to(accelerator.device, dtype=weight_dtype)
    if args.sdxl:
        text_encoder_one.to(accelerator.device, dtype=weight_dtype)
        text_encoder_two.to(accelerator.device, dtype=weight_dtype)
        print("offload vae (this actually stays as CPU)")
        vae = accelerate.cpu_offload(vae)
        print("Offloading text encoders to cpu")
        text_encoder_one = accelerate.cpu_offload(text_encoder_one)
        text_encoder_two = accelerate.cpu_offload(text_encoder_two)
        if args.train_method in ('dpo', 'nash', 'nash_sft', 'online_dpo'):
            ref_unet.to(accelerator.device, dtype=weight_dtype)
            print("offload ref_unet")
            ref_unet = accelerate.cpu_offload(ref_unet)
            if args.train_method in ('nash', 'nash_sft', 'online_dpo'):
                prev_unet.to(accelerator.device, dtype=weight_dtype)
                print("offload prev_unet")
                prev_unet = accelerate.cpu_offload(prev_unet)
    else:
        text_encoder.to(accelerator.device, dtype=weight_dtype)
        # text_encoder= accelerate.cpu_offload(text_encoder)
        # vae = accelerate.cpu_offload(vae)
        if args.train_method in ('dpo', 'nash', 'nash_sft', 'online_dpo'):
            ref_unet.to(accelerator.device, dtype=weight_dtype)
            # ref_unet = accelerate.cpu_offload(ref_unet)
            if args.train_method in ('nash', 'nash_sft', 'online_dpo'):
                prev_unet.to(accelerator.device, dtype=weight_dtype)
                print("offload prev_unet")
                # prev_unet = accelerate.cpu_offload(prev_unet)
    ### END ACCELERATOR PREP ###





    # ---- Nash (online) helpers ----
    sample_two_images_sd15 = None
    infer_scheduler = None
    if args.train_method in ('nash', 'nash_sft', 'online_dpo'):
        if args.sdxl:
            raise NotImplementedError(
                "Nash online sampling is currently implemented only for SD1.x (non-SDXL) in this script."
            )
        from diffusers import DDIMScheduler
        infer_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        infer_scheduler.set_timesteps(args.nash_num_inference_steps, device=accelerator.device)

        def sample_k_images_sd15_batched(unet_infer, prompts, seed_bases, num_candidates: int):
            """Sample K PIL images per prompt (batched) and return their final latents."""
            if num_candidates < 2:
                raise ValueError("num_candidates must be >= 2 for Nash sampling.")

            bs = len(prompts)
            assert bs == len(seed_bases)
            # text embeddings (cond + uncond) for classifier-free guidance
            with torch.no_grad():
                text_in = tokenizer(
                    prompts,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.to(accelerator.device)
                uncond_in = tokenizer(
                    [""] * bs,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.to(accelerator.device)
                cond_emb = text_encoder(text_in)[0]
                uncond_emb = text_encoder(uncond_in)[0]
                # [2B, seq, dim]
                prompt_emb = torch.cat([uncond_emb, cond_emb], dim=0).to(dtype=weight_dtype)

            # shared init noise per-sample
            latents_list = []
            for s in seed_bases:
                g_init = torch.Generator(device=accelerator.device).manual_seed(s)
                latents_one = torch.randn(
                    (1, unet_infer.in_channels, args.resolution // 8, args.resolution // 8),
                    generator=g_init,
                    device=accelerator.device,
                    dtype=weight_dtype,
                )
                latents_list.append(latents_one)
            latents0 = torch.cat(latents_list, dim=0)  # [B, C, H/8, W/8]

            out_per_k = []
            latents_per_k = []
            for k in range(num_candidates):
                # per-sample generators (affect stochastic part of DDIM when eta>0)
                g_steps = [
                    torch.Generator(device=accelerator.device).manual_seed(sb + 1000 * (k + 1))
                    for sb in seed_bases
                ]
                latents = latents0.clone()
                for t in infer_scheduler.timesteps:
                    # CFG
                    latent_in = torch.cat([latents, latents], dim=0)  # [2B, ...]
                    latent_in = infer_scheduler.scale_model_input(latent_in, t)
                    with torch.no_grad():
                        noise_pred = unet_infer(
                            latent_in,
                            t,
                            encoder_hidden_states=prompt_emb,
                            added_cond_kwargs={},
                        ).sample
                    noise_uncond, noise_text = noise_pred.chunk(2)
                    noise_pred = noise_uncond + args.nash_guidance_scale * (noise_text - noise_uncond)
                    step = infer_scheduler.step(noise_pred, t, latents, eta=args.nash_ddim_eta, generator=g_steps)
                    latents = step.prev_sample

                with torch.no_grad():
                    imgs = vae.decode(latents / vae.config.scaling_factor).sample
                imgs = (imgs / 2 + 0.5).clamp(0, 1).detach().cpu().permute(0, 2, 3, 1).numpy()
                from PIL import Image as _PILImage  # avoid shadowing
                pils = [_PILImage.fromarray((imgs[i] * 255).round().astype("uint8")) for i in range(bs)]
                out_per_k.append(pils)
                latents_per_k.append(latents.detach().clone())

            # Transpose: [K][B] -> [B][K]
            per_prompt = []
            per_prompt_latents = []
            for i in range(bs):
                per_prompt.append([out_per_k[k][i] for k in range(num_candidates)])
                per_prompt_latents.append([latents_per_k[k][i] for k in range(num_candidates)])
            return per_prompt, per_prompt_latents
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Training initialization
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0


    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)
        

    # Bram Note: This was pretty janky to wrangle to look proper but works to my liking now
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

        
    #### START MAIN TRAINING LOOP #####
    oracle_score_sums = None
    oracle_score_counts = None
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        implicit_acc_accumulated = 0.0
        grad_norm_accumulated = float("nan")
        grad_norm_post_clip_accumulated = float("nan")
        nash_sft_w_loss_accumulated = 0.0
        nash_only_loss_accumulated = 0.0
        nash_metric_accumulated = {
            "model_diff": 0.0,
            "ref_diff": 0.0,
            "prev_diff": 0.0,
            "model_losses_w": 0.0,
            "model_losses_l": 0.0,
            "ref_losses_w": 0.0,
            "ref_losses_l": 0.0,
            "prev_losses_w": 0.0,
            "prev_losses_l": 0.0,
        }
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step and (not args.hard_skip_resume):
                if step % args.gradient_accumulation_steps == 0:
                    print(f"Dummy processing step {step}, will start training at {resume_step}")
                continue
            with accelerator.accumulate(unet):
                # Convert images to latent space
                if args.train_method == 'dpo':
                    # y_w and y_l were concatenated along channel dimension
                    feed_pixel_values = torch.cat(batch["pixel_values"].chunk(2, dim=1))
                    # If using AIF then we haven't ranked yet so do so now
                    # Only implemented for BS=1 (assert-protected)
                    if args.choice_model:
                        if choice_model_says_flip(batch):
                            feed_pixel_values = feed_pixel_values.flip(0)
                
                elif args.train_method == 'sft':
                    feed_pixel_values = batch["pixel_values"]
                elif args.train_method in ('nash', 'nash_sft', 'online_dpo'):
                    # Online: sample multiple images per prompt from current unet, then use UNet loss ranking to pick winner.
                    assert nash_pick_winner_loser is not None, "Nash UNet ranker is not initialized."
                    assert (not args.sdxl), "Online Nash sampling path currently supports only non-SDXL"
                    assert sample_k_images_sd15_batched is not None, "Nash sampling helper not initialized"

                    if oracle_score_sums is None:
                        oracle_score_sums = {"unet_model_loss_mean": 0.0, "unet_avg_rank": 0.0}
                        oracle_score_counts = {"unet_model_loss_mean": 0, "unet_avg_rank": 0}
                
                    unet_infer = accelerator.unwrap_model(prev_unet)
                    unet_infer.eval()
                
                    # Move annotators to GPU just-in-time
                    # TODO: assign device as necessary
                    move_nash_oracles_device(accelerator.device)
                
                    win_tensors = []
                    lose_tensors = []
                    prompts = batch["caption"]
                    seed_bases = [(args.seed or 0) + global_step * 10000 + i for i in range(len(prompts))]
                    # batched sampling (8 candidates per prompt)
                    num_candidates = 8
                    pil_candidates, latent_candidates = sample_k_images_sd15_batched(unet_infer, prompts, seed_bases, num_candidates)
                    del unet_infer
                    torch.cuda.empty_cache()
                    unet_judge = accelerator.unwrap_model(unet)
                    unet_judge.eval()
                    for i, prompt in enumerate(prompts):
                        candidates = pil_candidates[i]
                        candidate_latents = latent_candidates[i]
                        w_idx, l_idx, oracle_scores = nash_pick_winner_loser(candidate_latents, prompt, unet_judge)
                        for name, scores in oracle_scores.items():
                            oracle_score_sums[name] += float(np.mean(scores))
                            oracle_score_counts[name] += 1
                        pil_w, pil_l = candidates[w_idx], candidates[l_idx]
                        win_tensors.append(train_transforms(pil_w))
                        lose_tensors.append(train_transforms(pil_l))
                
                    # Stack as [bs, 3, H, W] then concat into [2*bs, 3, H, W] with winner first.
                    y_w = torch.stack(win_tensors, dim=0).to(memory_format=torch.contiguous_format).float().to(accelerator.device)
                    y_l = torch.stack(lose_tensors, dim=0).to(memory_format=torch.contiguous_format).float().to(accelerator.device)
                    feed_pixel_values = torch.cat([y_w, y_l], dim=0)
                    
                    # Offload annotators back to CPU to free VRAM
                    # TODO: assign device as necessary
                    #move_nash_oracles_device('cpu')
                    if accelerator.device.type == 'cuda':
                        torch.cuda.empty_cache()
                
                
                #### Diffusion Stuff ####
                # encode pixels --> latents
                with torch.no_grad():
                    latents = vae.encode(feed_pixel_values.to(weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                # variants of noising
                if args.noise_offset: # haven't tried yet
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )
                if args.input_perturbation: # haven't tried yet
                    new_noise = noise + args.input_perturbation * torch.randn_like(noise)
                    
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                # only first 20% timesteps for SDXL refiner
                if 'refiner' in args.pretrained_model_name_or_path:
                    timesteps = timesteps % 200
                elif 'turbo' in args.pretrained_model_name_or_path:
                    timesteps_0_to_3 = timesteps % 4
                    timesteps = 250 * timesteps_0_to_3 + 249
                
                if args.train_method in ('dpo', 'nash', 'nash_sft', 'online_dpo'): # make timesteps and noise same for pairs in DPO
                    timesteps = timesteps.chunk(2)[0].repeat(2)
                    noise = noise.chunk(2)[0].repeat(2, 1, 1, 1)

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                
                noisy_latents = noise_scheduler.add_noise(latents,
                                                          new_noise if args.input_perturbation else noise,
                                                          timesteps)
                ### START PREP BATCH ###
                if args.sdxl:
                    # Get the text embedding for conditioning
                    with torch.no_grad():
                        # Need to compute "time_ids" https://github.com/huggingface/diffusers/blob/v0.20.0-release/examples/text_to_image/train_text_to_image_sdxl.py#L969
                        # for SDXL-base these are torch.tensor([args.resolution, args.resolution, *crop_coords_top_left, *target_size))
                        if 'refiner' in args.pretrained_model_name_or_path:
                            add_time_ids = torch.tensor([args.resolution, 
                                                         args.resolution,
                                                         0,
                                                         0,
                                                          6.0], # aesthetics conditioning https://github.com/huggingface/diffusers/blob/v0.20.0/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_img2img.py#L691C9-L691C24
                                                         dtype=weight_dtype,
                                                         device=accelerator.device)[None, :].repeat(timesteps.size(0), 1)
                        else: # SDXL-base
                            add_time_ids = torch.tensor([args.resolution, 
                                                         args.resolution,
                                                         0,
                                                         0,
                                                          args.resolution, 
                                                         args.resolution],
                                                         dtype=weight_dtype,
                                                         device=accelerator.device)[None, :].repeat(timesteps.size(0), 1)
                        prompt_batch = encode_prompt_sdxl(batch, 
                                                          text_encoders,
                                                           tokenizers,
                                                           args.proportion_empty_prompts, 
                                                          caption_column='caption',
                                                           is_train=True,
                                                          )
                    if args.train_method in ('dpo', 'nash', 'nash_sft', 'online_dpo'):
                        prompt_batch["prompt_embeds"] = prompt_batch["prompt_embeds"].repeat(2, 1, 1)
                        prompt_batch["pooled_prompt_embeds"] = prompt_batch["pooled_prompt_embeds"].repeat(2, 1)
                    unet_added_conditions = {"time_ids": add_time_ids,
                                            "text_embeds": prompt_batch["pooled_prompt_embeds"]}
                else: # sd1.5
                    # Get the text embedding for conditioning
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                    if args.train_method in ('dpo', 'nash', 'nash_sft', 'online_dpo'):
                        encoder_hidden_states = encoder_hidden_states.repeat(2, 1, 1)
                #### END PREP BATCH ####
                        
                assert noise_scheduler.config.prediction_type == "epsilon"
                target = noise
               
                # Make the prediction from the model we're learning
                model_batch_args = (noisy_latents,
                                    timesteps, 
                                    prompt_batch["prompt_embeds"] if args.sdxl else encoder_hidden_states)
                added_cond_kwargs = unet_added_conditions if args.sdxl else {}
                
                model_pred = unet(
                                *model_batch_args,
                                  added_cond_kwargs = added_cond_kwargs
                                 ).sample
                #### START LOSS COMPUTATION ####
                if args.train_method == 'sft': # SFT, casting for F.mse_loss
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                elif args.train_method in ('dpo', 'online_dpo'):
                    # model_pred and ref_pred will be (2 * LBS) x 4 x latent_spatial_dim x latent_spatial_dim
                    # losses are both 2 * LBS
                    # 1st half of tensors is preferred (y_w), second half is unpreferred
                    model_losses = (model_pred - target).pow(2).mean(dim=[1,2,3])
                    model_losses_w, model_losses_l = model_losses.chunk(2)
                    # below for logging purposes
                    raw_model_loss = 0.5 * (model_losses_w.mean() + model_losses_l.mean())
                    
                    model_diff = model_losses_w - model_losses_l # These are both LBS (as is t)
                    
                    with torch.no_grad(): # Get the reference policy (unet) prediction
                        ref_pred = ref_unet(
                                    *model_batch_args,
                                      added_cond_kwargs = added_cond_kwargs
                                     ).sample.detach()
                        ref_losses = (ref_pred - target).pow(2).mean(dim=[1,2,3])
                        ref_losses_w, ref_losses_l = ref_losses.chunk(2)
                        ref_diff = ref_losses_w - ref_losses_l
                        raw_ref_loss = ref_losses.mean()    
                        
                    scale_term = -0.5 * args.beta_dpo
                    inside_term = scale_term * (model_diff - ref_diff)
                    implicit_acc = (inside_term > 0).sum().float() / inside_term.size(0)
                    loss = -1 * F.logsigmoid(inside_term).mean()
                
                elif args.train_method in ('nash', 'nash_sft'):
                    # INPO / Nash objective (Eq. 8 in the paper): minimize ( h_t(π, y_w, y_l) - 1/(2η) )^2
                    # h_t(π, y_w, y_l) = log π(y_w)/π(y_l)
                    #                 - (τ/η) log π_ref(y_w)/π_ref(y_l)
                    #                 - ((η-τ)/η) log π_t(y_w)/π_t(y_l)
                    # For diffusion, we use the common approximation log π(y|x) ∝ -0.5 * ||ε - ε_θ||^2 at matched (t, ε).
                    model_losses = (model_pred - target).pow(2).mean(dim=[1,2,3])
                    model_losses_w, model_losses_l = model_losses.chunk(2)
                    model_diff = model_losses_w - model_losses_l
                
                    with torch.no_grad():
                        # reference π_ref
                        ref_pred = ref_unet(*model_batch_args, added_cond_kwargs=added_cond_kwargs).sample.detach()
                        ref_losses = (ref_pred - target).pow(2).mean(dim=[1,2,3])
                        ref_losses_w, ref_losses_l = ref_losses.chunk(2)
                        ref_diff = ref_losses_w - ref_losses_l
                
                        # previous iterate π_t
                        prev_pred = prev_unet(*model_batch_args, added_cond_kwargs=added_cond_kwargs).sample.detach()
                        prev_losses = (prev_pred - target).pow(2).mean(dim=[1,2,3])
                        prev_losses_w, prev_losses_l = prev_losses.chunk(2)
                        prev_diff = prev_losses_w - prev_losses_l
                
                    # Convert MSE diffs into log-ratio approximations
                    # log π(y_w)/π(y_l) ≈ -0.5 * (mse_w - mse_l)
                    log_ratio = -0.5 * model_diff
                    log_ratio_ref = -0.5 * ref_diff
                    log_ratio_prev = -0.5 * prev_diff
                
                    tau = args.nash_tau
                    eta = args.nash_eta
                    h_t = log_ratio - (tau / eta) * log_ratio_ref - ((eta - tau) / eta) * log_ratio_prev
                    target_val = 1.0 / (2.0 * eta)
                    nash_loss = (h_t - target_val).pow(2).mean()
                    if args.train_method == 'nash_sft':
                        # winner-only SFT term
                        sft_w_loss = model_losses_w.mean()
                        nash_sft_w_loss_accumulated += (
                            accelerator.gather(sft_w_loss.detach().reshape(1)).mean().item()
                            / args.gradient_accumulation_steps
                        )
                        nash_only_loss_accumulated += (
                            accelerator.gather(nash_loss.detach().reshape(1)).mean().item()
                            / args.gradient_accumulation_steps
                        )
                        loss = nash_loss + args.nash_sft_lambda * sft_w_loss
                    else:
                        loss = nash_loss

                    # Gather Nash internals for per-optimizer-step logging.
                    nash_metric_tensors = {
                        "model_diff": model_diff.mean(),
                        "ref_diff": ref_diff.mean(),
                        "prev_diff": prev_diff.mean(),
                        "model_losses_w": model_losses_w.mean(),
                        "model_losses_l": model_losses_l.mean(),
                        "ref_losses_w": ref_losses_w.mean(),
                        "ref_losses_l": ref_losses_l.mean(),
                        "prev_losses_w": prev_losses_w.mean(),
                        "prev_losses_l": prev_losses_l.mean(),
                    }
                    for metric_name, metric_tensor in nash_metric_tensors.items():
                        metric_avg = accelerator.gather(metric_tensor.detach().reshape(1)).mean().item()
                        nash_metric_accumulated[metric_name] += metric_avg / args.gradient_accumulation_steps
                
                #### END LOSS COMPUTATION ###
                    
                # Gather the losses across all processes for logging 
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                # Also gather:
                # - model MSE vs reference MSE (useful to observe divergent behavior)
                # - Implicit accuracy
                if args.train_method in ('dpo', 'online_dpo'):
                    avg_model_mse = accelerator.gather(raw_model_loss.repeat(args.train_batch_size)).mean().item()
                    avg_ref_mse = accelerator.gather(raw_ref_loss.repeat(args.train_batch_size)).mean().item()
                    avg_acc = accelerator.gather(implicit_acc).mean().item()
                    implicit_acc_accumulated += avg_acc / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if not args.use_adafactor: # Adafactor does itself, maybe could do here to cut down on code
                        grad_norm = accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                        if isinstance(grad_norm, torch.Tensor):
                            grad_norm_accumulated = accelerator.gather(grad_norm.detach().reshape(1)).mean().item()
                        else:
                            grad_norm_accumulated = float(grad_norm)
                        # Compute post-clip grad norm for observability.
                        grad_sq_sum = None
                        for p in unet.parameters():
                            if p.grad is None:
                                continue
                            g = p.grad.detach().float()
                            term = (g * g).sum()
                            grad_sq_sum = term if grad_sq_sum is None else (grad_sq_sum + term)
                        if grad_sq_sum is None:
                            grad_norm_post = torch.tensor(0.0, device=accelerator.device)
                        else:
                            grad_norm_post = torch.sqrt(grad_sq_sum)
                        grad_norm_post_clip_accumulated = accelerator.gather(grad_norm_post.reshape(1)).mean().item()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has just performed an optimization step, if so do "end of batch" logging
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if args.train_method in ('nash', 'nash_sft', 'online_dpo') and oracle_score_sums is not None:
                    oracle_avgs = {}
                    for name in oracle_score_sums.keys():
                        sum_tensor = torch.tensor(oracle_score_sums[name], device=accelerator.device)
                        count_tensor = torch.tensor(oracle_score_counts[name], device=accelerator.device)
                        sum_all = accelerator.gather(sum_tensor).sum().item()
                        count_all = accelerator.gather(count_tensor).sum().item()
                        oracle_avgs[name] = sum_all / max(count_all, 1.0)
                    if accelerator.is_main_process:
                        avg_loss_val = train_loss
                        logger.info(f"[step {global_step}] loss={avg_loss_val:.6f} oracle_avgs={oracle_avgs}")
                    oracle_score_sums = {name: 0.0 for name in oracle_score_sums.keys()}
                    oracle_score_counts = {name: 0 for name in oracle_score_counts.keys()}
                if args.train_method in ('nash', 'nash_sft', 'online_dpo'):
                    # Refresh π_t from current π every few optimizer steps.
                    if global_step > 0 and (global_step % args.nash_prev_update_steps == 0):
                        with torch.no_grad():
                            src = accelerator.unwrap_model(unet)
                            src_sd = src.state_dict()
                            prev_sd = prev_unet.state_dict()
                            if args.nash_prev_hard_update:
                                for k in prev_sd.keys():
                                    prev_sd[k].copy_(src_sd[k])
                            else:
                                mu = min(0.001 * global_step, 0.5)
                                for k in prev_sd.keys():
                                    if torch.is_floating_point(prev_sd[k]):
                                        # soft update: theta_prev <- mu * theta_prev + (1-mu) * theta_current
                                        prev_sd[k].mul_(mu).add_(src_sd[k], alpha=(1.0 - mu))
                                    else:
                                        prev_sd[k].copy_(src_sd[k])
                        prev_unet.eval()
                accelerator.log({"train_loss": train_loss}, step=global_step)
                accelerator.log({"grad_norm": grad_norm_accumulated}, step=global_step)
                accelerator.log({"grad_norm_post_clip": grad_norm_post_clip_accumulated}, step=global_step)
                if args.train_method in ('dpo', 'online_dpo'):
                    accelerator.log({"model_mse_unaccumulated": avg_model_mse}, step=global_step)
                    accelerator.log({"ref_mse_unaccumulated": avg_ref_mse}, step=global_step)
                    accelerator.log({"implicit_acc_accumulated": implicit_acc_accumulated}, step=global_step)
                if args.train_method in ('nash', 'nash_sft'):
                    accelerator.log({f"nash_{k}": v for k, v in nash_metric_accumulated.items()}, step=global_step)
                    if args.train_method == 'nash_sft':
                        accelerator.log(
                            {
                                "nash_sft_w_loss": nash_sft_w_loss_accumulated,
                                "nash_only_loss": nash_only_loss_accumulated,
                            },
                            step=global_step,
                        )
                    if accelerator.is_main_process:
                        logger.info(
                            f"[step {global_step}] "
                            f"model_diff={nash_metric_accumulated['model_diff']:.6f} "
                            f"ref_diff={nash_metric_accumulated['ref_diff']:.6f} "
                            f"prev_diff={nash_metric_accumulated['prev_diff']:.6f} "
                            f"model_w={nash_metric_accumulated['model_losses_w']:.6f} "
                            f"model_l={nash_metric_accumulated['model_losses_l']:.6f} "
                            f"ref_w={nash_metric_accumulated['ref_losses_w']:.6f} "
                            f"ref_l={nash_metric_accumulated['ref_losses_l']:.6f} "
                            f"prev_w={nash_metric_accumulated['prev_losses_w']:.6f} "
                            f"prev_l={nash_metric_accumulated['prev_losses_l']:.6f} "
                            f"grad_norm={grad_norm_accumulated:.6f} "
                            f"grad_norm_post_clip={grad_norm_post_clip_accumulated:.6f} "
                            f"nash_only_loss={nash_only_loss_accumulated:.6f} "
                            f"sft_w_loss={nash_sft_w_loss_accumulated:.6f}"
                        )
                train_loss = 0.0
                implicit_acc_accumulated = 0.0
                grad_norm_accumulated = float("nan")
                grad_norm_post_clip_accumulated = float("nan")
                nash_sft_w_loss_accumulated = 0.0
                nash_only_loss_accumulated = 0.0
                for k in nash_metric_accumulated.keys():
                    nash_metric_accumulated[k] = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        tau_val = getattr(args, "nash_tau", None)
                        eta_val = getattr(args, "nash_eta", None)
                        if tau_val is not None and eta_val is not None:
                            tau_tag = str(tau_val).replace(".", "p")
                            eta_tag = str(eta_val).replace(".", "p")
                            ckpt_name = f"checkpoint-{global_step}-loop-tau{tau_tag}-eta{eta_tag}"
                        else:
                            ckpt_name = f"checkpoint-{global_step}-loop"
                        save_path = os.path.join(args.output_dir, ckpt_name)
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                        logger.info("Pretty sure saving/loading is fixed but proceed cautiously")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            if args.train_method in ('dpo', 'online_dpo'):
                logs["implicit_acc"] = avg_acc
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break


    # Create the pipeline using the trained modules and save it.
    # This will save to top level of output_dir instead of a checkpoint directory
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if args.sdxl:
            # Serialize pipeline.
            vae = AutoencoderKL.from_pretrained(
                vae_path,
                subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
                revision=args.revision,
                torch_dtype=weight_dtype,
            )
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                args.pretrained_model_name_or_path, unet=unet, vae=vae, revision=args.revision, torch_dtype=weight_dtype
            )
            pipeline.save_pretrained(args.output_dir)
        else:
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                text_encoder=text_encoder,
                vae=vae,
                unet=unet,
                revision=args.revision,
            )
        pipeline.save_pretrained(args.output_dir)


    accelerator.end_training()


if __name__ == "__main__":
    main()
