#!/usr/bin/env python3
"""
Debug script to compare what happens during training vs testing
"""
import argparse
import glob
import os
import sys
sys.path.append('../')
import yaml
import torch
import numpy as np
import random
from torch.nn.parallel import DistributedDataParallel as DDP

from peft import LoraConfig, get_peft_model, TaskType

from data.data import get_audiotext_dataloader
from distributed import init_distributed_device, world_info_from_env
from train_utils import Dict2Class, get_autocast, get_cast_dtype
from src.factory_inference import create_model_and_transforms


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def apply_lora_to_model(model, lora_config):
    """Apply LoRA adapters to the language encoder"""
    lang_encoder = model.lang_encoder
    lang_encoder_lora = get_peft_model(lang_encoder, lora_config)
    
    if not hasattr(lang_encoder_lora, 'peft_config'):
        raise ValueError("LoRA not properly applied to language encoder")
    
    model.lang_encoder = lang_encoder_lora
    
    lora_param_count = sum(p.numel() for n, p in model.named_parameters() 
                          if 'lora_' in n and p.requires_grad)
    print(f"LoRA adapter parameters: {lora_param_count:,}")
    
    return model


def setup_lora_trainable_params(model, lora_config):
    """Setup which parameters should be trainable with LoRA - SAME AS TRAIN_LORA.PY"""
    # First, freeze everything
    for param in model.parameters():
        param.requires_grad = False
    
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


def load_lora_checkpoint(model, checkpoint_path, device_id):
    print(f"Loading LoRA checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    epoch = checkpoint["epoch"]
    
    model.lang_encoder.load_state_dict(checkpoint["model_state_dict"], strict=False)
    print(f"✓ Loaded LoRA weights from epoch {epoch}")
    
    audio_checkpoint_path = checkpoint_path.replace("lora_checkpoint_", "audio_transformer_checkpoint_")
    if os.path.exists(audio_checkpoint_path):
        audio_dict = torch.load(audio_checkpoint_path, map_location="cpu")
        model.audio_transformer_clap.load_state_dict(audio_dict["audio_transformer_clap"], strict=False)
        print(f"✓ Loaded audio transformer weights from epoch {audio_dict['epoch']}")
    
    return epoch


def debug_one_batch(model, tokenizer, batch, device_id, cast_dtype):
    """Debug a single batch to see what's happening"""
    print("\n" + "="*60)
    print("DEBUG: Analyzing one batch")
    print("="*60)
    
    # Load data
    audio_clips = batch["audio_clips"].to(device_id, dtype=cast_dtype, non_blocking=True)
    audio_embed_mask = batch["audio_embed_mask"].to(device_id, dtype=cast_dtype, non_blocking=True)
    input_ids = batch["input_ids"].to(device_id, dtype=torch.long, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device_id, dtype=cast_dtype, non_blocking=True)
    
    print(f"\n1. INPUT SHAPES:")
    print(f"   audio_clips: {audio_clips.shape}")
    print(f"   input_ids: {input_ids.shape}")
    
    # Decode input to see if it contains ground truth
    print(f"\n2. DECODED INPUT (first sample):")
    full_text = tokenizer.decode(input_ids[0])
    print(f"   Full length: {len(full_text)} chars")
    print(f"   First 200 chars: {full_text[:200]}")
    print(f"   Last 200 chars: {full_text[-200:]}")
    
    # Check for SEP token
    if tokenizer.sep_token in full_text:
        parts = full_text.split(tokenizer.sep_token)
        print(f"\n3. SEP TOKEN FOUND:")
        print(f"   Number of parts: {len(parts)}")
        print(f"   Part 0 (prompt): {parts[0][:100]}...")
        if len(parts) > 1:
            print(f"   Part 1 (target): {parts[-1][:100]}...")
    else:
        print(f"\n3. ⚠️ WARNING: NO SEP TOKEN FOUND!")
    
    # Setup labels
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    labels[:, :1] = -100
    labels[labels == tokenizer.encode("<audio>")[-1]] = -100
    
    sep_token_id = tokenizer.sep_token_id
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)["input_ids"][-1]
    
    sep_locations = labels == sep_token_id
    eoc_locations = labels == endofchunk_token_id
    
    print(f"\n4. LABEL MASKING:")
    print(f"   SEP tokens: {sep_locations.sum().item()}")
    print(f"   EOC tokens: {eoc_locations.sum().item()}")
    print(f"   Non-masked tokens: {(labels != -100).sum().item()}")
    
    # Forward pass
    print(f"\n5. FORWARD PASS:")
    autocast = get_autocast('amp_bf16', cache_enabled=True)
    
    with torch.no_grad():
        with autocast():
            output = model(
                audio_x=audio_clips,
                audio_x_mask=audio_embed_mask,
                lang_x=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
    
    print(f"   Output logits shape: {output.logits.shape}")
    print(f"   Loss: {output.loss.item() if output.loss is not None else 'None'}")
    
    # Get prediction
    sep_positions = (input_ids[0] == sep_token_id).nonzero()
    if len(sep_positions) > 0:
        sep_pos = sep_positions[-1].item()
        print(f"\n6. PREDICTION EXTRACTION:")
        print(f"   SEP position: {sep_pos}")
        
        pred_logits_after_sep = output.logits[0, sep_pos:-1, :]
        pred_tokens_after_sep = torch.argmax(pred_logits_after_sep, dim=-1)
        
        pred_text = tokenizer.decode(pred_tokens_after_sep)
        pred_text = pred_text.replace('<|endofchunk|>', '').replace(tokenizer.eos_token, '').replace(tokenizer.pad_token, '').strip()
        
        print(f"   Predicted tokens: {pred_tokens_after_sep.shape[0]}")
        print(f"   Predicted text length: {len(pred_text)} chars")
        print(f"   Predicted text:\n{pred_text}")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='../configs/vstep_lora.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parsed_args = parser.parse_args()
    
    print("="*60)
    print("DEBUG TEST SCRIPT - Comparing train vs test behavior")
    print("="*60)
    
    # Load config
    config = yaml.load(open(parsed_args.config), Loader=yaml.FullLoader)
    data_config = config['data_config']
    model_config = config['model_config']
    clap_config = config['clap_config']
    lora_config = config['lora_config']
    args = Dict2Class(config['train_config'])
    
    # Setup distributed
    if args.offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
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
    random_seed(args.seed, args.rank)
    
    # Create model
    print("\n1. CREATING MODEL (same as train_lora.py)")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    model, tokenizer = create_model_and_transforms(
        **model_config,
        clap_config=clap_config,
        use_local_files=args.offline,
        gradient_checkpointing=args.gradient_checkpointing,
        freeze_lm_embeddings=args.freeze_lm_embeddings,
        unfreeze_full_lm=False
    )
    
    # Freeze all params first
    with torch.no_grad():
        for param in model.parameters():
            param.requires_grad = False
    
    # Apply LoRA (SAME AS TRAIN_LORA.PY)
    print("\n2. APPLYING LoRA (same as train_lora.py)")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,  # SAME AS TRAINING
        r=lora_config['r'],
        lora_alpha=lora_config['lora_alpha'],
        lora_dropout=lora_config['lora_dropout'],
        target_modules=lora_config['target_modules'],
        bias=lora_config.get('bias', 'none'),
        modules_to_save=lora_config.get('modules_to_save', None)
    )
    
    model = apply_lora_to_model(model, peft_config)
    
    # CRITICAL: Call setup_lora_trainable_params (SAME AS TRAIN_LORA.PY)
    print("\n3. SETUP TRAINABLE PARAMS (same as train_lora.py)")
    model = setup_lora_trainable_params(model, peft_config)
    
    # Load checkpoint
    print("\n4. LOADING CHECKPOINT")
    epoch = load_lora_checkpoint(model, parsed_args.checkpoint, device_id)
    
    # Move to GPU
    model = model.to(device_id)
    
    # Set to eval mode
    print("\n5. SETTING EVAL MODE")
    model.eval()
    
    # Wrap with DDP (SAME AS TRAINING)
    print("\n6. WRAPPING WITH DDP (same as train_lora.py)")
    ddp_model = DDP(model, device_ids=[device_id], find_unused_parameters=True)
    ddp_model.eval()
    
    # Load test data
    print("\n7. LOADING TEST DATA")
    test_data = get_audiotext_dataloader(
        data_config, clap_config, tokenizer, args.batch_size, split='test'
    )
    
    dataset_name = list(test_data.keys())[0]
    testloader = test_data[dataset_name].dataloader
    
    print(f"   Dataset: {dataset_name}")
    print(f"   Batch size: {args.batch_size}")
    
    # Debug first batch
    print("\n8. DEBUGGING FIRST BATCH")
    for batch in testloader:
        debug_one_batch(ddp_model.module, tokenizer, batch, device_id, get_cast_dtype(args.precision))
        break  # Only first batch
    
    print("\n" + "="*60)
    print("DEBUG COMPLETE")
    print("="*60)
    print("\nNow compare this output with what happens during training!")


if __name__ == "__main__":
    main()