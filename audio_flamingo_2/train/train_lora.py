# Copyright (c) 2025 NVIDIA CORPORATION. 
#   Licensed under the MIT license.

# Adapted from https://github.com/mlfoundations/open_flamingo under the MIT license.
#   LICENSE is in incl_licenses directory.

""" Main LoRA training script """

import argparse
import functools
import glob
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
import random
import shutil
import sys 
sys.path.append('../')
import yaml
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointWrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from torch.distributed.fsdp._init_utils import _init_intra_and_inter_node_groups
from torch.distributed.distributed_c10d import _get_default_group
torch.cuda.empty_cache() 

# LoRA imports
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from peft.utils import _get_submodules

from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from data.data import get_audiotext_dataloader  
from distributed import init_distributed_device, world_info_from_env
from train_utils import (
    train_one_epoch,
    get_mp_policy_dtype,
    save_checkpoint,
    Dict2Class,
    get_autocast, 
    get_cast_dtype
)
from valid_utils import validation_losses
from src.factory import create_model_and_transforms


def load_pretrained_from_hf(model, repo_id="nvidia/audio-flamingo-2-1.5B", hf_token=None):
    """Load pretrained AudioFlamingo weights from HuggingFace"""
    try:
        from huggingface_hub import snapshot_download
        import torch
        
        print(f"Loading pretrained Audio Flamingo 2 from {repo_id}")
        
        # Download the model files
        repo_path = snapshot_download(
            repo_id=repo_id,
            token=hf_token,
            allow_patterns=["*.pt", "*.safetensors", "*.bin"],
        )
        
        # Find the model checkpoint
        checkpoint_files = glob.glob(os.path.join(repo_path, "*.pt"))
        if not checkpoint_files:
            checkpoint_files = glob.glob(os.path.join(repo_path, "*.safetensors"))
        if not checkpoint_files:
            checkpoint_files = glob.glob(os.path.join(repo_path, "*.bin"))
            
        if not checkpoint_files:
            raise FileNotFoundError("No model checkpoint found in the repository")
            
        checkpoint_path = checkpoint_files[0]
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            model_state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            model_state_dict = checkpoint["state_dict"]
        else:
            model_state_dict = checkpoint
            
        # Remove module prefix if present
        model_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}
        
        # Load weights with strict=False to allow for missing keys
        missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
        
        if missing_keys:
            print(f"Warning: Missing keys when loading pretrained model: {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"Warning: Unexpected keys when loading pretrained model: {unexpected_keys[:5]}...")
        
        print("Successfully loaded pretrained Audio Flamingo 2 weights")
        
    except Exception as e:
        print(f"Failed to load pretrained model from HuggingFace: {e}")
        print("Continuing with randomly initialized weights...")


def setup_lora_model(model, lora_config):
    """Setup LoRA adapters for the language model"""
    print("Setting up LoRA adapters...")
    
    # First, identify all trainable parameters before LoRA
    original_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Apply LoRA to language model
    # Find the language model component
    lang_encoder = model.lang_encoder
    
    # Configure LoRA for the language model
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_config['r'],
        lora_alpha=lora_config['lora_alpha'],
        lora_dropout=lora_config['lora_dropout'],
        target_modules=lora_config['target_modules'],
        bias=lora_config.get('bias', 'none'),
    )
    
    # Apply LoRA to language encoder
    model.lang_encoder = get_peft_model(lang_encoder, peft_config)
    
    # Keep audio components trainable
    model.audio_transformer_clap.requires_grad_(True)
    model.lang_encoder.gated_cross_attn_layers_sound.requires_grad_(True)
    
    # Embeddings handling
    if not lora_config.get('freeze_lm_embeddings', True):
        if hasattr(model.lang_encoder, 'base_model'):
            model.lang_encoder.base_model.get_input_embeddings().requires_grad_(True)
        else:
            model.lang_encoder.get_input_embeddings().requires_grad_(True)
    
    # Calculate final trainable parameters
    final_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"LoRA setup complete!")
    print(f"Original trainable parameters: {original_trainable:,}")
    print(f"Final trainable parameters: {final_trainable:,}")
    print(f"Parameter reduction: {(1 - final_trainable/max(original_trainable, 1)):.2%}")
    
    # Print LoRA specific info
    if hasattr(model.lang_encoder, 'print_trainable_parameters'):
        model.lang_encoder.print_trainable_parameters()
    
    return model


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='../configs/lora_config.yaml', help='yaml config path')
    parsed_args = parser.parse_args()

    config = yaml.load(open(parsed_args.config), Loader=yaml.FullLoader)
    data_config = config['data_config']
    model_config = config['model_config']
    clap_config = config["clap_config"]
    lora_config = config['lora_config']  # New LoRA configuration
    args = Dict2Class(config['train_config'])

    if 'sft_config' in config:
        sft_config = config['sft_config']
        unfreeze_full_lm = sft_config['unfreeze_full_lm']
    else:
        sft_config = None
        unfreeze_full_lm = False

    # get paths done 
    exp_path = os.path.join(args.expdir, args.run_name)
    os.makedirs(exp_path, exist_ok=True)
    print('exp_path:', exp_path)
    shutil.copy(parsed_args.config, os.path.join(exp_path, 'config.yaml'))
    data_config["dataset_blending_output"] = os.path.join(exp_path, data_config["dataset_blending_output"])

    # Initialize distributed training
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    device_id = init_distributed_device(args)

    # Random seed
    random_seed(args.seed, args.rank)

    # Create model and transforms
    print("Creating model...")
    model, tokenizer = create_model_and_transforms(
        **model_config,
        clap_config=clap_config,
        use_local_files=args.offline,
        gradient_checkpointing=args.gradient_checkpointing,
        freeze_lm_embeddings=lora_config.get('freeze_lm_embeddings', True),
        unfreeze_full_lm=False,  # We'll handle this with LoRA
    )

    # Load pretrained weights if specified
    if sft_config and sft_config.get('pretrained_path'):
        if sft_config['pretrained_path'].startswith('nvidia/'):
            hf_token = "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"  # Replace with your token or set to None for public models
            load_pretrained_from_hf(model, repo_id="nvidia/audio-flamingo-2-1.5B", hf_token=hf_token)
        else:
            # Load local checkpoint
            checkpoint_path = os.path.join(sft_config['pretrained_path'], sft_config['pretrained_ckpt'])
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                model_state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
                model_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}
                model.load_state_dict(model_state_dict, strict=False)
                print(f"Loaded weights from {checkpoint_path}")

    # Setup LoRA adapters
    model = setup_lora_model(model, lora_config)
    
    # Move model to device
    model = model.to(device_id)

    # Clear cache before setting up distributed training
    torch.cuda.empty_cache()

    # Setup distributed training
    if args.world_size > 1:
        if args.fsdp:
            print("Using FSDP for distributed training")
            # FSDP setup (simplified for LoRA)
            model = FSDP(
                model,
                auto_wrap_policy=None,
                sharding_strategy=ShardingStrategy.FULL_SHARD if args.fsdp_sharding_strategy == "full" else ShardingStrategy.HYBRID_SHARD,
                device_id=device_id,
                use_orig_params=args.fsdp_use_orig_params,
            )
        else:
            print("Using DDP for distributed training")
            model = DDP(model, device_ids=[device_id], find_unused_parameters=True)
    
    ddp_model = model

    # Print memory usage
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    # Resume from checkpoint if exists
    resume_from_checkpoint = None
    if os.path.exists(f"{exp_path}/checkpoint_last.pt") and not args.delete_previous_checkpoint:
        resume_from_checkpoint = f"{exp_path}/checkpoint_last.pt"
    elif args.resume_from_checkpoint is not None:
        resume_from_checkpoint = args.resume_from_checkpoint

    if resume_from_checkpoint is not None:
        print(f"Loading checkpoint from {resume_from_checkpoint}")
        checkpoint = torch.load(resume_from_checkpoint, map_location="cpu")
        
        # For LoRA, we need to load the state dict carefully
        model_state_dict = checkpoint["model_state_dict"]
        if args.fsdp:
            model_state_dict = FSDP.optim_state_dict_to_load(model_state_dict, ddp_model)
            
        missing_keys, unexpected_keys = ddp_model.load_state_dict(model_state_dict, strict=False)
        if missing_keys:
            print(f"Missing keys: {missing_keys[:10]}...")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys[:10]}...")
            
        print(f"Resumed from checkpoint {resume_from_checkpoint}")

    # Setup optimizer - only for trainable parameters
    params_to_optimize = [(n, p) for n, p in ddp_model.named_parameters() if p.requires_grad]
    total_trainable = sum(p.numel() for _, p in params_to_optimize)
    
    print(f"Optimizing {len(params_to_optimize)} parameter groups with {total_trainable:,} total parameters")
    
    # Use different learning rates for different components if specified
    if lora_config.get('use_different_lr', False):
        lora_params = []
        other_params = []
        
        for n, p in params_to_optimize:
            if 'lora_' in n:
                lora_params.append(p)
            else:
                other_params.append(p)
        
        optimizer = torch.optim.AdamW([
            {'params': lora_params, 'lr': lora_config.get('lora_lr', args.learning_rate)},
            {'params': other_params, 'lr': args.learning_rate}
        ], weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(
            (p for _, p in params_to_optimize),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

    # Load optimizer checkpoint
    if resume_from_checkpoint is not None:
        osd = checkpoint["optimizer_state_dict"]
        if args.fsdp:
            osd = FSDP.optim_state_dict_to_load(osd, ddp_model, optimizer)
        optimizer.load_state_dict(osd)
        del checkpoint["optimizer_state_dict"]
        del osd

    # Initialize data loaders
    AudioTextDataInfo = get_audiotext_dataloader(
        data_config, clap_config, tokenizer, args.batch_size, split='train',
        epoch=0, force_reblend=True
    )

    total_training_steps = (
        len(AudioTextDataInfo.dataset) // (args.batch_size * args.world_size)
    ) * args.num_epochs

    if args.rank == 0:
        print(f"Total training steps: {total_training_steps}")
        print(f"Dataset size: {len(AudioTextDataInfo.dataset)}")
        print(f"Batch size: {args.batch_size}")
        print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
        print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps * args.world_size}")
        tb = SummaryWriter(os.path.join(exp_path, 'tensorboard'))
    else:
        tb = None

    # Initialize lr scheduler
    if args.lr_scheduler == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_training_steps,
        )
    elif args.lr_scheduler == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_training_steps,
        )
    else:
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps
        )

    # Load lr scheduler checkpoint
    if resume_from_checkpoint is not None:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        del checkpoint["lr_scheduler_state_dict"]

    # Validation data loader
    if args.rank == 0:
        valid_dataloader_info = get_audiotext_dataloader(
            data_config, clap_config, tokenizer, args.batch_size, split='val',
            epoch=0, force_reblend=False
        )
    else:
        valid_dataloader_info = None

    # Start training!
    print("Starting LoRA training...")
    
    for epoch in range(args.num_epochs):
        # Set epoch for distributed sampling
        AudioTextDataInfo.sampler.set_epoch(epoch)
        
        # Training epoch
        ddp_model.train()
        
        train_metrics = train_one_epoch(
            args=args,
            model=ddp_model,
            epoch=epoch,
            tokenizer=tokenizer,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            AudioTextDataInfo=AudioTextDataInfo,
            device_id=device_id,
            tb=tb,
        )
        
        # Validation
        if args.rank == 0 and valid_dataloader_info is not None:
            ddp_model.eval()
            with torch.no_grad():
                valid_metrics = validation_losses(
                    args=args,
                    model=ddp_model,
                    epoch=epoch,
                    tokenizer=tokenizer,
                    valid_dataloader_info=valid_dataloader_info,
                    device_id=device_id,
                    tb=tb,
                )
            
            # Log validation metrics
            if tb is not None:
                for key, value in valid_metrics.items():
                    tb.add_scalar(f"valid/{key}", value, epoch)
        
        # Save checkpoint
        if args.rank == 0:
            save_checkpoint(
                ddp_model=ddp_model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                epoch=epoch,
                args=args,
            )
            
            # Save best model based on validation loss
            if epoch == 0 or (valid_dataloader_info and 'loss' in valid_metrics and 
                              valid_metrics['loss'] < getattr(save_checkpoint, 'best_valid_loss', float('inf'))):
                save_checkpoint.best_valid_loss = valid_metrics.get('loss', float('inf'))
                # Save as best model
                best_path = os.path.join(exp_path, 'checkpoint_best.pt')
                checkpoint_path = os.path.join(exp_path, f'checkpoint_{epoch}.pt')
                if os.path.exists(checkpoint_path):
                    shutil.copy(checkpoint_path, best_path)
                    print(f"Saved best model at epoch {epoch}")

        # Clear cache periodically
        if epoch % 5 == 0:
            torch.cuda.empty_cache()
    
    if args.rank == 0:
        print("LoRA training completed!")
        if tb is not None:
            tb.close()


if __name__ == "__main__":
    main()