#!/usr/bin/env python3
"""
FIXED Test script - Properly loads embedding layer
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

from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType
)

from metrics import mae_test_epoch
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


def load_lora_checkpoint_FIXED(model, checkpoint_path, device_id):
    """
    Load LoRA checkpoint WITH EMBEDDINGS - FIXED VERSION
    """
    print(f"Loading LoRA checkpoint from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    epoch = checkpoint["epoch"]
    
    # Load LoRA weights
    model.lang_encoder.load_state_dict(checkpoint["model_state_dict"], strict=False)
    print(f"✓ Loaded LoRA weights from epoch {epoch}")
    
    # ✅ FIX: LOAD EMBEDDING LAYER!
    if "embedding_state_dict" in checkpoint:
        embedding_state = checkpoint["embedding_state_dict"]
        
        # Load embed_tokens
        if "embed_tokens.weight" in embedding_state:
            model.lang_encoder.get_input_embeddings().weight.data.copy_(
                embedding_state["embed_tokens.weight"]
            )
            print(f"✓ Loaded embedding layer (special tokens included)")
        
        # Load lm_head if exists
        if "lm_head.weight" in embedding_state and hasattr(model.lang_encoder, 'lm_head'):
            model.lang_encoder.lm_head.weight.data.copy_(
                embedding_state["lm_head.weight"]
            )
            print(f"✓ Loaded lm_head layer")
    else:
        print(f"⚠ WARNING: No embedding_state_dict found in checkpoint!")
        print(f"⚠ This checkpoint was saved with the OLD version - special tokens may not work correctly!")
    
    # Load audio transformer
    audio_checkpoint_path = checkpoint_path.replace("lora_checkpoint_", "audio_transformer_checkpoint_")
    if os.path.exists(audio_checkpoint_path):
        audio_dict = torch.load(audio_checkpoint_path, map_location="cpu")
        model.audio_transformer_clap.load_state_dict(audio_dict["audio_transformer_clap"], strict=False)
        print(f"✓ Loaded audio transformer from epoch {audio_dict['epoch']}")
    else:
        print(f"⚠ Warning: Audio transformer checkpoint not found")
    
    return epoch

def find_latest_checkpoint(checkpoint_dir):
    checkpoint_pattern = os.path.join(checkpoint_dir, "lora_checkpoint_*.pt")
    checkpoint_list = glob.glob(checkpoint_pattern)
    
    if len(checkpoint_list) == 0:
        raise FileNotFoundError(f"No LoRA checkpoints found in {checkpoint_dir}")
    
    latest_checkpoint = sorted(
        checkpoint_list, 
        key=lambda x: int(x.split("_")[-1].split(".")[0])
    )[-1]
    
    return latest_checkpoint


def main():
    parser = argparse.ArgumentParser(description='Test Audio Flamingo LoRA model (FIXED VERSION)')
    parser.add_argument('-c', '--config', type=str, 
                       default='../configs/vstep_lora.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Specific checkpoint path')
    parser.add_argument('--max_samples', type=int, default=50,
                       help='Maximum number of test samples')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for inference')
    parsed_args = parser.parse_args()
    
    # Load configuration
    print(f"Loading config from: {parsed_args.config}")
    config = yaml.load(open(parsed_args.config), Loader=yaml.FullLoader)
    
    data_config = config['data_config']
    model_config = config['model_config']
    clap_config = config['clap_config']
    lora_config = config['lora_config']
    args = Dict2Class(config['train_config'])
    
    if parsed_args.batch_size:
        args.batch_size = parsed_args.batch_size
    
    checkpoint_dir = os.path.join(args.expdir, args.run_name)
    print(f"Checkpoint directory: {checkpoint_dir}")
    
    # Initialize distributed environment
    print("Initializing distributed environment...")
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
    
    # Create model and tokenizer
    print("\n" + "="*60)
    print("Creating model and tokenizer...")
    print("="*60)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    model, tokenizer = create_model_and_transforms(
        **model_config,
        clap_config=clap_config,
        use_local_files=args.offline,
        gradient_checkpointing=args.gradient_checkpointing,
        freeze_lm_embeddings=args.freeze_lm_embeddings,
        unfreeze_full_lm=False
    )
    print("✓ Model and tokenizer created")
    print(f"✓ Tokenizer size: {len(tokenizer)} tokens")
    print(f"✓ Special tokens: <audio>={tokenizer.encode('<audio>')[-1]}, "
          f"<|endofchunk|>={tokenizer.encode('<|endofchunk|>')[-1]}, "
          f"<|PAD_TOKEN|>={tokenizer.pad_token_id}")
    
    # Freeze parameters
    with torch.no_grad():
        for param in model.parameters():
            param.requires_grad = False
    
    # Setup LoRA
    print("\nApplying LoRA configuration...")
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
    
    model = apply_lora_to_model(model, peft_config)
    print("✓ LoRA applied to model")
    
    # Find and load checkpoint
    if parsed_args.checkpoint:
        checkpoint_path = parsed_args.checkpoint
    else:
        checkpoint_path = find_latest_checkpoint(checkpoint_dir)
    
    epoch = load_lora_checkpoint_FIXED(model, checkpoint_path, device_id)
    
    # Move to GPU and wrap with DDP
    model = model.to(device_id)
    print(f"✓ Model moved to device {device_id}")
    
    ddp_model = DDP(model, device_ids=[device_id], find_unused_parameters=True)
    
    print(f"\n{'='*60}")
    print(f"Starting inference on test set")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Epoch: {epoch}")
    print(f"Max samples: {parsed_args.max_samples}")
    print(f"Batch size: {args.batch_size}")
    print(f"{'='*60}\n")
    
    # Run inference
    model.eval()
    with torch.no_grad():
        mae_metrics, sample_logs = mae_test_epoch(
            model=ddp_model.module,
            data_config=data_config,
            clap_config=clap_config,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            autocast=get_autocast(args.precision, cache_enabled=True),
            cast_dtype=get_cast_dtype(args.precision),
            device_id=device_id,
            max_samples=parsed_args.max_samples
        )
    
    # Print results
    print(f"\n{'='*60}")
    print(f"TEST RESULTS - Epoch {epoch}")
    print(f"{'='*60}")
    print(f"Total MAE:          {mae_metrics['mae_total']:.4f}")
    print(f"Vocabulary MAE:     {mae_metrics['mae_vocabulary']:.4f}")
    print(f"Grammar MAE:        {mae_metrics['mae_grammar']:.4f}")
    print(f"Pronunciation MAE:  {mae_metrics['mae_pronunciation']:.4f}")
    print(f"Fluency MAE:        {mae_metrics['mae_fluency']:.4f}")
    print(f"Discourse MAE:      {mae_metrics['mae_discourse']:.4f}")
    print(f"{'='*60}")
    print(f"Average MAE:        {mae_metrics['mae_average']:.4f}")
    print(f"Samples tested:     {mae_metrics['num_samples']}")
    print(f"{'='*60}\n")
    
    # Print sample predictions to verify format
    print("\n📝 Sample Predictions (to verify format):")
    print("="*60)
    for i, sample in enumerate(sample_logs[:3], 1):
        print(f"\nSample {i}:")
        print(f"Ground Truth: {sample['ground_truth'][:150]}")
        print(f"Prediction:   {sample['prediction'][:150]}")
        print(f"GT Scores:    {sample['gt_scores']}")
        print(f"Pred Scores:  {sample['pred_scores']}")
    
    # Save results
    results_file = os.path.join(checkpoint_dir, f"test_results_FIXED_epoch_{epoch}.txt")
    with open(results_file, 'w') as f:
        f.write(f"FIXED Test Results - Epoch {epoch}\n")
        f.write("="*60 + "\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Max samples: {parsed_args.max_samples}\n\n")
        f.write("MAE Metrics:\n")
        f.write(f"  Total:          {mae_metrics['mae_total']:.4f}\n")
        f.write(f"  Vocabulary:     {mae_metrics['mae_vocabulary']:.4f}\n")
        f.write(f"  Grammar:        {mae_metrics['mae_grammar']:.4f}\n")
        f.write(f"  Pronunciation:  {mae_metrics['mae_pronunciation']:.4f}\n")
        f.write(f"  Fluency:        {mae_metrics['mae_fluency']:.4f}\n")
        f.write(f"  Discourse:      {mae_metrics['mae_discourse']:.4f}\n")
        f.write(f"  Average:        {mae_metrics['mae_average']:.4f}\n\n")
        
        f.write("Sample Predictions:\n")
        f.write("="*60 + "\n")
        for i, sample in enumerate(sample_logs[:10], 1):
            f.write(f"\nSample {i}:\n")
            f.write(f"  Ground Truth: {sample['ground_truth'][:100]}...\n")
            f.write(f"  Prediction: {sample['prediction'][:100]}...\n")
            f.write(f"  GT Scores: {sample['gt_scores']}\n")
            f.write(f"  Pred Scores: {sample['pred_scores']}\n")
    
    print(f"\nResults saved to: {results_file}")
    print("\n✅ Test completed successfully!")


if __name__ == "__main__":
    main()