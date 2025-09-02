#!/usr/bin/env python3
"""
LoRA fine-tuning script for Audio Flamingo 2
Custom LoRA implementation to avoid PEFT compatibility issues
"""

import argparse
import glob
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
import random
import shutil
import sys
sys.path.append('../')
import yaml
import time
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from data.data import get_audiotext_dataloader, AudioTextData, DataCollator
from train_utils import Dict2Class, get_cast_dtype
from valid_utils import validation_losses
from src.factory import create_model_and_transforms


def get_single_gpu_dataloader(data_config, clap_config, tokenizer, batch_size, split='train', epoch=0):
    """Custom data loader for single GPU training without DistributedSampler"""
    from torch.utils.data import DataLoader
    
    data_collator = DataCollator(tokenizer, clap_config)
    
    if split == 'train':
        dataset = AudioTextData(
            **data_config,
            clap_config=clap_config,
            tokenizer=tokenizer,
            split=split,
            epoch=epoch,
            force_reblend=True
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,  # Regular shuffle instead of distributed
            collate_fn=data_collator,
            num_workers=data_config["num_workers"],
            pin_memory=True,
            drop_last=True
        )
        
        return dataloader
        
    elif split in ['val', 'test']:
        dataloaders = {}
        for valid_dataset_name in data_config.get("valid_dataset_config", {}).keys():
            valid_dataset_name = valid_dataset_name.strip()
            dataset = AudioTextData(
                **data_config,
                clap_config=clap_config,
                tokenizer=tokenizer,
                split=split,
                valid_dataset_name=valid_dataset_name
            )
            
            dataloaders[valid_dataset_name] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=data_collator,
                num_workers=data_config["num_workers"],
                pin_memory=True
            )
        
        return dataloaders


class LoRALinear(nn.Module):
    """Custom LoRA linear layer implementation"""
    def __init__(self, original_layer, rank=16, alpha=32, dropout=0.1):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        
        # Freeze original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False
            
        # LoRA matrices
        if hasattr(original_layer, 'weight'):
            in_features = original_layer.in_features
            out_features = original_layer.out_features
            
            self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
    def forward(self, x):
        original_out = self.original_layer(x)
        
        if hasattr(self, 'lora_A'):
            lora_out = (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
            return original_out + lora_out
        return original_out


def apply_lora_to_model(model, target_modules, rank=16, alpha=32):
    """Apply LoRA to specific modules in the model"""
    
    def apply_lora_recursive(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            # Apply LoRA to target linear layers
            if isinstance(child, nn.Linear) and any(target in full_name for target in target_modules):
                print(f"Applying LoRA to: {full_name}")
                lora_layer = LoRALinear(child, rank=rank, alpha=alpha)
                setattr(module, name, lora_layer)
            else:
                apply_lora_recursive(child, full_name)
    
    apply_lora_recursive(model)
    return model


def load_pretrained_from_hf(model, repo_id="nvidia/audio-flamingo-2", hf_token=None):
    """Load pretrained Audio Flamingo 2 model from HuggingFace"""
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
            print(f"Missing keys when loading pretrained model: {len(missing_keys)} keys")
        if unexpected_keys:
            print(f"Unexpected keys when loading pretrained model: {len(unexpected_keys)} keys")
            
        print("Successfully loaded pretrained Audio Flamingo 2 weights")
        
    except Exception as e:
        print(f"Failed to load pretrained model from HuggingFace: {e}")
        print("Continuing with randomly initialized weights...")


def count_trainable_parameters(model):
    """Count trainable parameters in the model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
    
    return trainable_params, total_params


def save_lora_checkpoint(model, optimizer, lr_scheduler, epoch, args):
    """Save LoRA checkpoint with only trainable parameters"""
    
    # Extract only LoRA parameters
    lora_state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state_dict[f"{name}.lora_A"] = module.lora_A
            lora_state_dict[f"{name}.lora_B"] = module.lora_B
    
    checkpoint = {
        "epoch": epoch,
        "lora_state_dict": lora_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
    }
    
    checkpoint_path = os.path.join(args.expdir, args.run_name, f"lora_checkpoint_{epoch}.pt")
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved LoRA checkpoint: {checkpoint_path}")


def load_lora_checkpoint(model, checkpoint_path):
    """Load LoRA checkpoint"""
    print(f"Loading LoRA checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    lora_state_dict = checkpoint["lora_state_dict"]
    
    # Load LoRA parameters
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_a_key = f"{name}.lora_A"
            lora_b_key = f"{name}.lora_B"
            
            if lora_a_key in lora_state_dict:
                module.lora_A.data = lora_state_dict[lora_a_key]
            if lora_b_key in lora_state_dict:
                module.lora_B.data = lora_state_dict[lora_b_key]
    
    return checkpoint


def train_one_epoch(model, dataloader, optimizer, lr_scheduler, tokenizer, device, epoch, args, tb=None):
    """Train one epoch with LoRA"""
    model.train()
    
    # Use new autocast syntax
    autocast_context = torch.amp.autocast('cuda', dtype=torch.bfloat16) if args.precision == 'amp_bf16' else torch.amp.autocast('cuda')
    scaler = torch.amp.GradScaler('cuda') if args.precision in ['amp_bf16', 'fp16'] else None
    
    total_loss = 0
    num_batches = 0
    gradient_accumulation_count = 0
    
    print(f"Starting epoch {epoch+1}, total batches: {len(dataloader)}")
    
    for batch_idx, batch in enumerate(dataloader):
        try:
            # Move batch to device - using correct key names from original code
            audio_clips = batch["audio_clips"].to(device, dtype=torch.float32, non_blocking=True)
            audio_embed_mask = batch["audio_embed_mask"].to(device, dtype=torch.float32, non_blocking=True)
            input_ids = batch["input_ids"].to(device, dtype=torch.long, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, dtype=torch.float32, non_blocking=True)
            
            # Create labels from input_ids following original code pattern
            labels = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100
            labels[:, :1] = -100  # Mask first token
            
            # Get special token IDs
            try:
                audio_token_id = tokenizer.encode("<audio>")[-1]
                labels[labels == audio_token_id] = -100
            except:
                pass  # Skip if <audio> token doesn't exist
            
            # Forward pass
            with autocast_context:
                outputs = model(
                    audio_x=audio_clips,
                    audio_x_mask=audio_embed_mask,
                    lang_x=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss / args.gradient_accumulation_steps
            
            # Backward pass
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            gradient_accumulation_count += 1
            
            # Optimizer step
            if gradient_accumulation_count >= args.gradient_accumulation_steps:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                lr_scheduler.step()
                optimizer.zero_grad()
                gradient_accumulation_count = 0
            
            total_loss += outputs.loss.item()
            num_batches += 1
            
            # Logging
            if batch_idx % args.logging_steps == 0 and batch_idx > 0:
                avg_loss = total_loss / num_batches
                current_lr = lr_scheduler.get_last_lr()[0]
                print(f"Epoch {epoch+1}, Batch {batch_idx}/{len(dataloader)}, Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")
                
                if tb:
                    global_step = epoch * len(dataloader) + batch_idx
                    tb.add_scalar("Train/Loss", avg_loss, global_step)
                    tb.add_scalar("Train/LR", current_lr, global_step)
            
            # Memory cleanup
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            print(f"Batch keys: {batch.keys()}")
            # Print shapes for debugging
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")
            raise e
    
    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='yaml config path')
    parsed_args = parser.parse_args()

    config = yaml.load(open(parsed_args.config), Loader=yaml.FullLoader)
    data_config = config['data_config']
    model_config = config['model_config'] 
    clap_config = config["clap_config"]
    lora_config = config['lora_config']
    args = Dict2Class(config['train_config'])

    # Setup paths
    exp_path = os.path.join(args.expdir, args.run_name)
    os.makedirs(exp_path, exist_ok=True)
    print('exp_path:', exp_path)
    shutil.copy(parsed_args.config, os.path.join(exp_path, 'config.yaml'))
    
    # Update dataset blending output path
    data_config["dataset_blending_output"] = os.path.join(exp_path, os.path.basename(data_config["dataset_blending_output"]))

    # Set device (single GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Create model
    print('Creating model...')
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model, tokenizer = create_model_and_transforms(
        **model_config,
        clap_config=clap_config,
        use_local_files=args.offline,
        gradient_checkpointing=args.gradient_checkpointing,
        freeze_lm_embeddings=args.freeze_lm_embeddings,
    )

    # Load pretrained weights
    hf_token = "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"  # Replace with your token
    load_pretrained_from_hf(model, repo_id="nvidia/audio-flamingo-2", hf_token=hf_token)

    # Apply LoRA to target modules
    target_modules = lora_config['target_modules']
    model = apply_lora_to_model(
        model, 
        target_modules, 
        rank=lora_config['rank'], 
        alpha=lora_config['alpha']
    )

    # Move to device
    model = model.to(device)

    # Count parameters
    count_trainable_parameters(model)

    # Setup optimizer for LoRA parameters only
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        lora_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )

    # Setup data loader (single GPU version)
    print('Loading training data...')
    train_dataloader = get_single_gpu_dataloader(
        data_config, clap_config, tokenizer, args.batch_size, split='train', epoch=0
    )
    
    total_training_steps = len(train_dataloader.dataset) * args.num_epochs // args.batch_size
    print(f"Dataset size: {len(train_dataloader.dataset)}")
    print(f"Batches per epoch: {len(train_dataloader)}")
    print(f"Total training steps: {total_training_steps}")

    # Setup lr scheduler
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

    # Setup tensorboard
    tb = SummaryWriter(os.path.join(exp_path, 'tensorboard'))

    # Check for existing checkpoints
    checkpoint_list = glob.glob(f"{exp_path}/lora_checkpoint_*.pt")
    resume_from_epoch = 0
    
    if checkpoint_list:
        latest_checkpoint = sorted(checkpoint_list, key=lambda x: int(x.split("_")[-1].split(".")[0]))[-1]
        print(f"Resuming from checkpoint: {latest_checkpoint}")
        checkpoint = load_lora_checkpoint(model, latest_checkpoint)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        resume_from_epoch = checkpoint["epoch"] + 1

    print(f'Starting training from epoch {resume_from_epoch}')

    # Training loop
    for epoch in range(resume_from_epoch, args.num_epochs):
        # Reload dataset each epoch for data blending
        if epoch > 0:
            print(f'Reloading dataset for epoch {epoch+1}...')
            train_dataloader = get_single_gpu_dataloader(
                data_config, clap_config, tokenizer, args.batch_size, split='train', epoch=epoch
            )
        
        # Train one epoch
        avg_loss = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            tokenizer=tokenizer,
            device=device,
            epoch=epoch,
            args=args,
            tb=tb
        )
        
        print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
        
        # Save checkpoint
        save_lora_checkpoint(model, optimizer, lr_scheduler, epoch, args)
        
        # Validation (every 2 epochs for LoRA)
        if epoch % 2 == 0:
            try:
                print("Running validation...")
                with torch.no_grad():
                    # Create simple validation dataset
                    val_dataloaders = get_single_gpu_dataloader(
                        data_config, clap_config, tokenizer, 
                        batch_size=1, split='val'
                    )
                    
                    model.eval()
                    val_loss = 0
                    val_batches = 0
                    
                    # Take first validation dataset
                    first_val_name = list(val_dataloaders.keys())[0]
                    val_dataloader = val_dataloaders[first_val_name]
                    
                    for val_batch in val_dataloader:
                        # Move batch to device with correct key names
                        audio_clips = val_batch["audio_clips"].to(device, dtype=torch.float32, non_blocking=True)
                        audio_embed_mask = val_batch["audio_embed_mask"].to(device, dtype=torch.float32, non_blocking=True)
                        input_ids = val_batch["input_ids"].to(device, dtype=torch.long, non_blocking=True)
                        attention_mask = val_batch["attention_mask"].to(device, dtype=torch.float32, non_blocking=True)
                        
                        # Create labels from input_ids
                        labels = input_ids.clone()
                        labels[labels == tokenizer.pad_token_id] = -100
                        labels[:, :1] = -100
                        
                        # Get special token IDs
                        try:
                            audio_token_id = tokenizer.encode("<audio>")[-1]
                            labels[labels == audio_token_id] = -100
                        except:
                            pass
                        
                        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                            outputs = model(
                                audio_x=audio_clips,
                                audio_x_mask=audio_embed_mask,
                                lang_x=input_ids,
                                attention_mask=attention_mask,
                                labels=labels
                            )
                        
                        val_loss += outputs.loss.item()
                        val_batches += 1
                        
                        if val_batches >= 10:  # Limit validation batches
                            break
                    
                    avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
                    print(f"Validation loss: {avg_val_loss:.4f}")
                    tb.add_scalar("Valid/Loss", avg_val_loss, epoch)
                    
            except Exception as e:
                print(f"Validation failed: {e}")
                import traceback
                traceback.print_exc()
        
        torch.cuda.empty_cache()

    # Save final checkpoint
    save_lora_checkpoint(model, optimizer, lr_scheduler, args.num_epochs-1, args)
    tb.close()
    
    print("Training completed!")


if __name__ == "__main__":
    main()