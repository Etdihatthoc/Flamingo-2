# Copyright (c) 2025 NVIDIA CORPORATION. 
#   Licensed under the MIT license.

# Adapted from https://github.com/mlfoundations/open_flamingo under the MIT license.
#   LICENSE is in incl_licenses directory.

""" LoRA training script for Audio Flamingo - Memory efficient fine-tuning """
from pyexpat import model
import wandb
import re
import numpy as np
from sklearn.metrics import mean_absolute_error
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
from torch.distributed.fsdp._init_utils import _init_intra_and_inter_node_groups
from torch.distributed.distributed_c10d import _get_default_group
torch.cuda.empty_cache() 

from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from metrics import mae_test_epoch, mae_val_epoch
# LoRA imports
from peft import (
    LoraConfig, 
    get_peft_model, 
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    TaskType
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
#from src.factory import create_model_and_transforms
from src.factory_inference import create_model_and_transforms


def load_pretrained_from_hf(model, repo_id="nvidia/audio-flamingo-2-1.5B", hf_token=None):
    """
    Load pretrained Audio Flamingo 2 model from HuggingFace
    Similar to inference_HF_pretrained/inference.py
    """
    print(f"Loading pretrained model from HuggingFace: {repo_id}")
    
    try:
        from huggingface_hub import snapshot_download
        from safetensors import safe_open
        import json
        
        # Download model files
        if hf_token:
            snapshot_download(repo_id=repo_id, local_dir="./hf_model", token=hf_token)
        else:
            snapshot_download(repo_id=repo_id, local_dir="./hf_model")
        
        # Load metadata
        with open("./hf_model/safe_ckpt/metadata.json", "r") as f:
            metadata = json.load(f)
        
        # Reconstruct the full state_dict
        state_dict = {}
        
        # Load each SafeTensors chunk
        for chunk_name in metadata:
            chunk_path = f"./hf_model/safe_ckpt/{chunk_name}.safetensors"
            with safe_open(chunk_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
        
        # Load state dict into model
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"Missing keys when loading pretrained model: {missing_keys[:20]}...")  # Show first 20
        if unexpected_keys:
            print(f"Unexpected keys when loading pretrained model: {unexpected_keys[:20]}...")  # Show first 20

        print("Successfully loaded pretrained Audio Flamingo 2 weights")
        
    except Exception as e:
        print(f"Failed to load pretrained model from HuggingFace: {e}")
        print("Continuing with randomly initialized weights...")


# def apply_lora_to_model(model, lora_config):
#     """Apply LoRA adapters to the language encoder"""
#     # Only apply LoRA to the language encoder, not the entire model
#     lang_encoder = model.lang_encoder
    
#     # Apply LoRA to language encoder
#     lang_encoder_lora = get_peft_model(lang_encoder, lora_config)
#     model.lang_encoder = lang_encoder_lora
    
#     return model
def apply_lora_to_model(model, lora_config):
    """Apply LoRA adapters to the language encoder"""
    # Only apply LoRA to the language encoder, not the entire model
    lang_encoder = model.lang_encoder
    
    # Apply LoRA to language encoder
    lang_encoder_lora = get_peft_model(lang_encoder, lora_config)
    
    # Verify LoRA was applied correctly
    if not hasattr(lang_encoder_lora, 'peft_config'):
        raise ValueError("LoRA not properly applied to language encoder")
    
    model.lang_encoder = lang_encoder_lora
    
    # Count LoRA parameters
    lora_param_count = sum(p.numel() for n, p in model.named_parameters() 
                          if 'lora_' in n and p.requires_grad)
    print(f"LoRA adapter parameters: {lora_param_count:,}")
    
    return model

def setup_lora_trainable_params(model, lora_config):
    """Setup which parameters should be trainable with LoRA"""
    # First, freeze everything
    for param in model.parameters():
        param.requires_grad = False
    
    # Enable LoRA adapters (they will be set to trainable by PEFT)
    if hasattr(model.lang_encoder, 'peft_config'):
        # LoRA adapters are automatically trainable
        pass
    
    # Keep essential components trainable
    model.audio_transformer_clap.requires_grad_(True)
    model.lang_encoder.get_input_embeddings().requires_grad_(True)
    
    # Enable gated cross attention layers if they exist
    if hasattr(model.lang_encoder, 'gated_cross_attn_layers_sound'):
        model.lang_encoder.gated_cross_attn_layers_sound.requires_grad_(True)
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    return model


# def save_lora_checkpoint(model, optimizer, lr_scheduler, epoch, args):
#     """Save LoRA checkpoint efficiently"""
#     if args.rank != 0:
#         return
        
#     exp_path = os.path.join(args.expdir, args.run_name)
#     checkpoint_dict = {
#         "epoch": epoch,
#         "model_state_dict": get_peft_model_state_dict(model.lang_encoder),  # Only save LoRA weights
#         "optimizer_state_dict": optimizer.state_dict(),
#         "lr_scheduler_state_dict": lr_scheduler.state_dict(),
#     }
    
#     # Save current checkpoint
#     checkpoint_path = f"{exp_path}/lora_checkpoint_{epoch}.pt"
#     torch.save(checkpoint_dict, checkpoint_path)
#     print(f"Saved LoRA checkpoint: {checkpoint_path}")
    
#     # Also save the full audio_transformer_clap state (it's small)
#     audio_transformer_dict = {
#         "audio_transformer_clap": model.audio_transformer_clap.state_dict(),
#         "epoch": epoch
#     }
#     audio_checkpoint_path = f"{exp_path}/audio_transformer_checkpoint_{epoch}.pt"
#     torch.save(audio_transformer_dict, audio_checkpoint_path)
    
#     # Keep only the last 5 checkpoints
#     checkpoint_list = glob.glob(f"{exp_path}/lora_checkpoint_*.pt")
#     if len(checkpoint_list) > 5:
#         checkpoint_list.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
#         for old_ckpt in checkpoint_list[:-3]:
#             os.remove(old_ckpt)
#             audio_ckpt = old_ckpt.replace("lora_checkpoint_", "audio_transformer_checkpoint_")
#             if os.path.exists(audio_ckpt):
#                 os.remove(audio_ckpt)

def save_lora_checkpoint(model, optimizer, lr_scheduler, epoch, args):
    """Save LoRA checkpoint với ĐẦY ĐỦ trainable components"""
    if args.rank != 0:
        return
        
    exp_path = os.path.join(args.expdir, args.run_name)
    
    # ✅ CÁCH 1: Lưu TOÀN BỘ lang_encoder state (khuyến nghị)
    checkpoint_dict = {
        "epoch": epoch,
        "model_state_dict": model.lang_encoder.state_dict(),  # ← ĐỔI DÒNG NÀY!
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
    }
    
    # Save checkpoint
    checkpoint_path = f"{exp_path}/lora_checkpoint_{epoch}.pt"
    torch.save(checkpoint_dict, checkpoint_path)
    print(f"Saved FULL LoRA checkpoint: {checkpoint_path}")
    
    # Also save audio_transformer_clap separately
    audio_transformer_dict = {
        "audio_transformer_clap": model.audio_transformer_clap.state_dict(),
        "epoch": epoch
    }
    audio_checkpoint_path = f"{exp_path}/audio_transformer_checkpoint_{epoch}.pt"
    torch.save(audio_transformer_dict, audio_checkpoint_path)
    
    # Keep only last 5 checkpoints
    checkpoint_list = glob.glob(f"{exp_path}/lora_checkpoint_*.pt")
    if len(checkpoint_list) > 5:
        checkpoint_list.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        for old_ckpt in checkpoint_list[:-5]:
            os.remove(old_ckpt)
            old_audio = old_ckpt.replace("lora_checkpoint_", "audio_transformer_checkpoint_")
            if os.path.exists(old_audio):
                os.remove(old_audio)
 

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='../configs/vstep_lora.yaml', help='yaml config path')
    parsed_args = parser.parse_args()

    config = yaml.load(open(parsed_args.config), Loader=yaml.FullLoader)
    data_config = config['data_config']
    model_config = config['model_config']
    clap_config = config["clap_config"]
    lora_config = config['lora_config']
    args = Dict2Class(config['train_config'])

    if 'sft_config' in config:
        sft_config = config['sft_config']
        unfreeze_full_lm = False  # Always False for LoRA
    else:
        sft_config = None
        unfreeze_full_lm = False

    # get paths done 
    exp_path = os.path.join(args.expdir, args.run_name)
    os.makedirs(exp_path, exist_ok=True)
    print('exp_path:', exp_path)
    shutil.copy(parsed_args.config, os.path.join(exp_path, 'config.yaml'))
    data_config["dataset_blending_output"] = os.path.join(exp_path, data_config["dataset_blending_output"])

    # Set up distributed training (required even for single GPU due to DistributedSampler)
    print('initializing distributed environment')
    if args.offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    # Always initialize distributed training (even for single GPU)
    if 'MASTER_ADDR' not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if 'MASTER_PORT' not in os.environ:
        os.environ["MASTER_PORT"] = "12356"
    if 'RANK' not in os.environ:
        os.environ["RANK"] = "0"
    if 'WORLD_SIZE' not in os.environ:
        os.environ["WORLD_SIZE"] = "1"
    
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    device_id = init_distributed_device(args)
    
    random_seed(args.seed)

    if args.rank == 0:  # Only on main process
        wandb.login(key=config.get('wandb_key'), relogin=True)
        wandb.init(
            project=config.get('wandb_project', 'Flamingo-2-finetune'),
            name=f"{args.run_name}-test",
            #dir='/home/user06/sondinh/Flamingo/wandb',
            config={
                **config['train_config'],
                **config['lora_config'],
                **config['model_config'],
                'dataset': config['data_config']['dataset_blending_config']
            },
            tags=['lora', 'flamingo2', 'vstep']
        )
        print("✅ Wandb initialized")

    # Initialize model
    print('creating model')
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model, tokenizer = create_model_and_transforms(
        **model_config,
        clap_config=clap_config, 
        use_local_files=args.offline,
        gradient_checkpointing=args.gradient_checkpointing,
        freeze_lm_embeddings=args.freeze_lm_embeddings,
        unfreeze_full_lm=unfreeze_full_lm
    )
    print("✅ Model and tokenizer created")
    
    # Load pretrained weights BEFORE applying LoRA
    if sft_config is not None and sft_config.get('pretrained_ckpt') is None:
        # Load from HuggingFace instead of local checkpoint
        hf_token = "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"  # Replace with your token or set to None for public models
        load_pretrained_from_hf(model, repo_id="nvidia/audio-flamingo-2-1.5B", hf_token=hf_token)
        print("Loaded pretrained model from HuggingFace for SFT.")
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Freeze model
    with torch.no_grad():
        for param in model.parameters():
            param.requires_grad = False

    # Setup LoRA configuration
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_config['r'],
        lora_alpha=lora_config['lora_alpha'],
        lora_dropout=lora_config['lora_dropout'],
        target_modules=lora_config['target_modules'],
        bias=lora_config.get('bias', 'none'),
        modules_to_save=lora_config.get('modules_to_save', None)
    )
    
    # Apply LoRA to the model
    model = apply_lora_to_model(model, peft_config)
    model = setup_lora_trainable_params(model, peft_config)
    
    print(f"Model created with LoRA adapters")
    random_seed(args.seed, args.rank)

    # Move model to GPU
    model = model.to(device_id)
    
    # Wrap with DDP for multi-GPU (single GPU also uses distributed for data loading compatibility)
    if args.world_size > 1:
        ddp_model = DDP(model, device_ids=[device_id], find_unused_parameters=True)
    else:
        # Single GPU still needs DDP wrapper for compatibility with distributed data loading
        ddp_model = DDP(model, device_ids=[device_id], find_unused_parameters=True)

    # Initialize logging
    print(f"Start running LoRA training on rank {args.rank}.")

    # Look for existing LoRA checkpoints
    checkpoint_list = glob.glob(f"{args.expdir}/{args.run_name}/lora_checkpoint_*.pt")
    if len(checkpoint_list) == 0:
        print(f"Found no LoRA checkpoints for run {args.run_name}.")
        resume_from_checkpoint = None
        resume_from_epoch = 0
    else:
        resume_from_checkpoint = sorted(
            checkpoint_list, key=lambda x: int(x.split("_")[-1].split(".")[0])
        )[-1]
        print(f"Found LoRA checkpoint {resume_from_checkpoint} for run {args.run_name}.")
        
        # Load LoRA checkpoint
        checkpoint = torch.load(resume_from_checkpoint, map_location="cpu")
        resume_from_epoch = checkpoint["epoch"] + 1

        # Load LoRA weights
        model.lang_encoder.load_state_dict(checkpoint["model_state_dict"], strict=False)
        
        # Load audio transformer weights
        audio_checkpoint = resume_from_checkpoint.replace("lora_checkpoint_", "audio_transformer_checkpoint_")
        if os.path.exists(audio_checkpoint):
            audio_dict = torch.load(audio_checkpoint, map_location="cpu")
            model.audio_transformer_clap.load_state_dict(audio_dict["audio_transformer_clap"], strict=False)
        
        print(f"Resumed from LoRA checkpoint, starting from epoch {resume_from_epoch}")

    # Identify trainable parameters for optimizer
    trainable_params = [p for p in ddp_model.parameters() if p.requires_grad]
    
    # Initialize optimizer - only for trainable parameters
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Load optimizer checkpoint if resuming
    if resume_from_checkpoint is not None:
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

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
    if resume_from_checkpoint is not None and "lr_scheduler_state_dict" in checkpoint:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

    # Start training!
    ddp_model.train()

    print('start LoRA training from epoch {}'.format(resume_from_epoch))
    for epoch in range(resume_from_epoch, args.num_epochs):
        #Force reblending dataset for every epoch
        if epoch > 0:
            AudioTextDataInfo = get_audiotext_dataloader(
                data_config, clap_config, tokenizer, args.batch_size, split='train',
                epoch=epoch, force_reblend=True
            )
        AudioTextDataInfo.set_epoch(epoch)
        trainloader = AudioTextDataInfo.dataloader
        
        # Train one epoch
        # train_one_epoch(
        #     args=args,
        #     model=ddp_model,
        #     epoch=epoch,
        #     tokenizer=tokenizer,
        #     optimizer=optimizer,
        #     lr_scheduler=lr_scheduler,
        #     trainloader=trainloader,
        #     device_id=device_id,
        #     tb=tb
        # )

        # Save LoRA checkpoint
        # save_lora_checkpoint(ddp_model.module, optimizer, lr_scheduler, epoch, args)
        # time.sleep(1.0)

        # Validation 
        print("Validation.......")
        if epoch % 1 == 0:
            if args.world_size > 1:
                torch.distributed.barrier()
                
            try:
                # Existing validation code
                with torch.no_grad():
                    valid_losses = validation_losses(
                        model=ddp_model, 
                        data_config=data_config, 
                        clap_config=clap_config, 
                        tokenizer=tokenizer, 
                        batch_size=args.batch_size, 
                        autocast=get_autocast(args.precision, cache_enabled=True), 
                        cast_dtype=get_cast_dtype(args.precision),
                        device_id=device_id
                    )
                
                # NEW: MAE test - Truyền đúng parameters như validation_losses
                if args.rank == 0:
                    # validation 
                    mae_metrics, sample_logs = mae_val_epoch(
                        model=ddp_model.module,  # Unwrap DDP
                        data_config=data_config,
                        clap_config=clap_config,
                        tokenizer=tokenizer,
                        batch_size=args.batch_size,  # Same batch size
                        autocast=get_autocast(args.precision, cache_enabled=True),  # Pass autocast
                        cast_dtype=get_cast_dtype(args.precision),  # Pass cast_dtype
                        device_id=device_id,
                        max_samples=5438
                    )
                    
                    # Log to wandb
                    if wandb.run is not None:
                        wandb_log = {"epoch": epoch + 1}
                        
                        # Add validation losses
                        for key in valid_losses:
                            wandb_log[f"valid/{key}"] = valid_losses[key]
                        
                        # Add MAE metrics
                        for key, value in mae_metrics.items():
                            wandb_log[f"mae/{key}"] = value
                        
                        # Log metrics
                        wandb.log(wandb_log)
                        
                        # Log sample predictions as table
                        if len(sample_logs) > 0:
                            sample_table = wandb.Table(columns=[
                                "Audio File", "Prompt", "Ground Truth", "Prediction", 
                                "GT Total", "GT Vocab", "GT Grammar", "GT Pronunciation", "GT Fluency", "GT Discourse", 
                                 "Pred Total", "Pred Vocab", "Pred Grammar", "Pred Pronunciation", "Pred Fluency", "Pred Discourse"
                            ])
                            
                            for sample in sample_logs:
                                sample_table.add_data(
                                    sample['audio_file'],
                                    sample['prompt'],
                                    sample['ground_truth'],
                                    sample['prediction'],
                                    sample['gt_scores'][0], sample['gt_scores'][1], sample['gt_scores'][2], 
                                    sample['gt_scores'][3], sample['gt_scores'][4], sample['gt_scores'][5],
                                    sample['pred_scores'][0], sample['pred_scores'][1], sample['pred_scores'][2],
                                    sample['pred_scores'][3], sample['pred_scores'][4], sample['pred_scores'][5]
                                )
                            
                            wandb.log({f"samples/epoch_{epoch+1}": sample_table})
                    
                    # Print MAE results
                    print(f"\n{'='*50}")
                    print(f"MAE Results - Epoch {epoch+1}")
                    print(f"{'='*50}")
                    print(f"Total MAE:          {mae_metrics['mae_total']:.4f}")
                    print(f"Vocab MAE:        {mae_metrics['mae_vocabulary']:.4f}")
                    print(f"Grammar MAE:     {mae_metrics['mae_grammar']:.4f}")
                    print(f"Pronunciation MAE:  {mae_metrics['mae_pronunciation']:.4f}")
                    print(f"Fluency MAE:        {mae_metrics['mae_fluency']:.4f}")
                    print(f"Discourse MAE:      {mae_metrics['mae_discourse']:.4f}")
                    print(f"Average MAE:        {mae_metrics['mae_average']:.4f}")
                    print(f"Samples tested:     {mae_metrics['num_samples']}")
                    print(f"{'='*50}\n")

                    
            except Exception as error:
                print("An exception occurred during validation/MAE:", error)
                import traceback
                traceback.print_exc()
                
            if args.world_size > 1:
                torch.distributed.barrier()

    print("Testing LoRA training (disabled full training for quick test)...")
    epoch = args.num_epochs - 1  # Set epoch to last for testing
    mae_metrics, sample_logs = mae_test_epoch(
                            model=ddp_model.module,  # Unwrap DDP
                            data_config=data_config,
                            clap_config=clap_config,
                            tokenizer=tokenizer,
                            batch_size=args.batch_size,  # Same batch size
                            autocast=get_autocast(args.precision, cache_enabled=True),  # Pass autocast
                            cast_dtype=get_cast_dtype(args.precision),  # Pass cast_dtype
                            device_id=device_id,
                            max_samples=12325  # Test on specified number of samples
                        )
                        
    # Log to wandb
    if wandb.run is not None:
        wandb_log = {"epoch": epoch + 1}
        
        # Add MAE metrics
        for key, value in mae_metrics.items():
            wandb_log[f"mae_test/{key}"] = value
        
        # Log metrics
        wandb.log(wandb_log)
        
        # Log sample predictions as table
        if len(sample_logs) > 0:
            sample_table = wandb.Table(columns=[
                "Audio File", "Prompt", "Ground Truth", "Prediction", 
                "GT Total", "GT Vocabulary", "GT Grammar", "GT Pronunciation", "GT Fluency", "GT Discourse",
                "Pred Total", "Pred Vocabulary", "Pred Grammar", "Pred Pronunciation", "Pred Fluency", "Pred Discourse"
            ])
            
            for sample in sample_logs:
                sample_table.add_data(
                    sample['audio_file'],
                    sample['prompt'],
                    sample['ground_truth'],
                    sample['prediction'],
                    sample['gt_scores'][0], sample['gt_scores'][1], sample['gt_scores'][2], 
                    sample['gt_scores'][3], sample['gt_scores'][4], sample['gt_scores'][5],
                    sample['pred_scores'][0], sample['pred_scores'][1], sample['pred_scores'][2],
                    sample['pred_scores'][3], sample['pred_scores'][4], sample['pred_scores'][5]
                )
            
            wandb.log({f"samples/epoch_{epoch+1}": sample_table})
    
    # Print MAE results
    print(f"\n{'='*50}")
    print(f"MAE Test Results - Epoch {epoch+1}")
    print(f"{'='*50}")
    print(f"Total MAE:      {mae_metrics['mae_total']:.4f}")
    print(f"Vocabulary MAE: {mae_metrics['mae_vocabulary']:.4f}")
    print(f"Grammar MAE:    {mae_metrics['mae_grammar']:.4f}")
    print(f"Discourse MAE:  {mae_metrics['mae_discourse']:.4f}")
    print(f"Pronunciation MAE: {mae_metrics['mae_pronunciation']:.4f}")
    print(f"Fluency MAE:    {mae_metrics['mae_fluency']:.4f}")
    print(f"Average MAE:    {mae_metrics['mae_average']:.4f}")

    # Save final LoRA checkpoint
    #save_lora_checkpoint(ddp_model.module, optimizer, lr_scheduler, epoch, args)
    
    if args.rank == 0:
        tb.close()
        print("LoRA training completed successfully!")


if __name__ == "__main__":
    main()
    #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
    #17611