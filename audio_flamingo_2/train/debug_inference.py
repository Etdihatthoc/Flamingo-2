#!/usr/bin/env python3
"""
Debug script để tìm nguyên nhân prediction bị sai format
"""
import sys
sys.path.append('../')
import torch
import yaml
from train_utils import Dict2Class, get_autocast, get_cast_dtype
from src.factory_inference import create_model_and_transforms
from data.data import get_audiotext_dataloader
from peft import LoraConfig, get_peft_model, TaskType
from distributed import init_distributed_device, world_info_from_env
import os
import glob

def debug_single_sample(model, tokenizer, batch, device_id, cast_dtype, autocast):
    """Debug một sample để xem prediction như thế nào"""
    
    # Load data
    audio_clips = batch["audio_clips"].to(device_id, dtype=cast_dtype, non_blocking=True)
    audio_embed_mask = batch["audio_embed_mask"].to(device_id, dtype=cast_dtype, non_blocking=True)
    input_ids = batch["input_ids"].to(device_id, dtype=torch.long, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device_id, dtype=cast_dtype, non_blocking=True)
    
    print("\n" + "="*80)
    print("DEBUG INFORMATION")
    print("="*80)
    
    # 1. Kiểm tra input_ids
    print(f"\n1. INPUT IDS SHAPE: {input_ids.shape}")
    print(f"   Sample input_ids (first 50 tokens): {input_ids[0, :50]}")
    
    # 2. Kiểm tra special tokens
    print(f"\n2. SPECIAL TOKENS:")
    print(f"   <audio> token ID: {tokenizer.encode('<audio>')[-1]}")
    print(f"   <SEP> token ID: {tokenizer.sep_token_id}")
    print(f"   <|endofchunk|> token ID: {tokenizer.encode('<|endofchunk|>')[-1]}")
    print(f"   EOS token ID: {tokenizer.eos_token_id}")
    print(f"   PAD token ID: {tokenizer.pad_token_id}")
    
    # 3. Decode full sequence
    full_decoded = tokenizer.decode(input_ids[0])
    print(f"\n3. FULL DECODED SEQUENCE:")
    print(f"   {full_decoded[:500]}...")
    
    # 4. Tìm SEP position
    sep_positions = (input_ids[0] == tokenizer.sep_token_id).nonzero()
    print(f"\n4. SEP TOKEN POSITIONS:")
    print(f"   Found SEP at positions: {sep_positions.squeeze().tolist() if len(sep_positions) > 0 else 'NONE'}")
    
    if len(sep_positions) > 0:
        sep_pos = sep_positions[-1].item()
        print(f"   Using SEP position: {sep_pos}")
        
        # 5. Decode phần prompt và ground truth
        if tokenizer.sep_token in full_decoded:
            parts = full_decoded.split(tokenizer.sep_token)
            prompt_text = parts[0].replace('<audio>', '').strip()
            ground_truth_text = parts[-1].replace('<|endofchunk|>', '').replace(tokenizer.eos_token, '').strip()
            
            print(f"\n5. PARSED CONTENT:")
            print(f"   PROMPT: {prompt_text[:200]}...")
            print(f"   GROUND TRUTH: {ground_truth_text[:200]}...")
    
    # 6. Setup labels
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    labels[:, :1] = -100
    labels[labels == tokenizer.encode("<audio>")[-1]] = -100
    
    endofchunk_token_id = tokenizer.encode("<|endofchunk|>")[-1]
    sep_locations = labels == tokenizer.sep_token_id
    eoc_locations = labels == endofchunk_token_id
    
    # Mask labels
    for i in range(labels.shape[0]):
        shouldmask = True
        for j in range(labels.shape[1]):
            if shouldmask and (labels[i][j] != tokenizer.eos_token_id):
                masked_value = -100
            else:
                masked_value = labels[i][j]
            
            if labels[i][j] == tokenizer.sep_token_id:
                shouldmask = False
            elif labels[i][j] == endofchunk_token_id:
                shouldmask = True
            
            labels[i][j] = masked_value
    
    labels = labels.to(device_id)
    
    # 7. Forward pass
    print(f"\n6. RUNNING FORWARD PASS...")
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
    
    # 8. Get prediction
    if len(sep_positions) > 0:
        sep_pos = sep_positions[-1].item()
        
        print(f"\n7. EXTRACTING PREDICTION:")
        print(f"   Taking logits from position {sep_pos} to {output.logits.shape[1]-1}")
        
        pred_logits_after_sep = output.logits[0, sep_pos:-1, :]
        pred_tokens_after_sep = torch.argmax(pred_logits_after_sep, dim=-1)
        
        print(f"   Predicted tokens shape: {pred_tokens_after_sep.shape}")
        print(f"   First 30 predicted tokens: {pred_tokens_after_sep[:30]}")
        
        # Decode
        pred_text = tokenizer.decode(pred_tokens_after_sep)
        pred_text_clean = pred_text.replace('<|endofchunk|>', '').replace(tokenizer.eos_token, '').replace(tokenizer.pad_token, '').strip()
        
        print(f"\n8. FINAL PREDICTION:")
        print(f"   RAW: {pred_text[:300]}")
        print(f"   CLEAN: {pred_text_clean[:300]}")
        
        # So sánh với ground truth
        if tokenizer.sep_token in full_decoded:
            parts = full_decoded.split(tokenizer.sep_token)
            ground_truth_text = parts[-1].replace('<|endofchunk|>', '').replace(tokenizer.eos_token, '').strip()
            
            print(f"\n9. COMPARISON:")
            print(f"   GROUND TRUTH: {ground_truth_text[:200]}")
            print(f"   PREDICTION:   {pred_text_clean[:200]}")
            
            # Check if format is correct
            if "vocabulary:" in pred_text_clean and "grammar:" in pred_text_clean:
                print(f"\n   ✅ FORMAT: CORRECT")
            else:
                print(f"\n   ❌ FORMAT: WRONG!")
                print(f"   Missing keywords or format corrupted")
    
    print("\n" + "="*80)


def main():
    # Load config
    config_path = '../configs/vstep_lora.yaml'
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    
    data_config = config['data_config']
    model_config = config['model_config']
    clap_config = config['clap_config']
    lora_config = config['lora_config']
    args = Dict2Class(config['train_config'])
    
    # Setup distributed
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
    
    # Create model
    print("Creating model...")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model, tokenizer = create_model_and_transforms(
        **model_config,
        clap_config=clap_config,
        use_local_files=args.offline,
        gradient_checkpointing=False,
        freeze_lm_embeddings=args.freeze_lm_embeddings,
        unfreeze_full_lm=False
    )
    
    # Freeze and apply LoRA
    with torch.no_grad():
        for param in model.parameters():
            param.requires_grad = False
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_config['r'],
        lora_alpha=lora_config['lora_alpha'],
        lora_dropout=lora_config['lora_dropout'],
        target_modules=lora_config['target_modules'],
        bias=lora_config.get('bias', 'none'),
    )
    
    lang_encoder_lora = get_peft_model(model.lang_encoder, peft_config)
    model.lang_encoder = lang_encoder_lora
    
    # Load checkpoint
    checkpoint_dir = os.path.join(args.expdir, args.run_name)
    checkpoint_pattern = os.path.join(checkpoint_dir, "lora_checkpoint_*.pt")
    checkpoint_list = glob.glob(checkpoint_pattern)
    
    if len(checkpoint_list) == 0:
        print(f"ERROR: No checkpoint found in {checkpoint_dir}")
        return
    
    latest_checkpoint = sorted(checkpoint_list, key=lambda x: int(x.split("_")[-1].split(".")[0]))[-1]
    print(f"Loading checkpoint: {latest_checkpoint}")
    
    checkpoint = torch.load(latest_checkpoint, map_location="cpu")
    model.lang_encoder.load_state_dict(checkpoint["model_state_dict"], strict=False)
    
    # Load audio transformer
    audio_checkpoint = latest_checkpoint.replace("lora_checkpoint_", "audio_transformer_checkpoint_")
    if os.path.exists(audio_checkpoint):
        audio_dict = torch.load(audio_checkpoint, map_location="cpu")
        model.audio_transformer_clap.load_state_dict(audio_dict["audio_transformer_clap"], strict=False)
    
    model = model.to(device_id)
    model.eval()
    
    # Get test data
    print("\nLoading test data...")
    all_test_data = get_audiotext_dataloader(
        data_config, clap_config, tokenizer, batch_size=1, split='test'
    )
    
    dataset_name = list(all_test_data.keys())[0]
    testloader = all_test_data[dataset_name].dataloader
    
    # Get first batch
    batch = next(iter(testloader))
    
    # Debug
    autocast = get_autocast(args.precision, cache_enabled=True)
    cast_dtype = get_cast_dtype(args.precision)
    
    print("\n" + "="*80)
    print("DEBUGGING FIRST TEST SAMPLE")
    print("="*80)
    
    debug_single_sample(model, tokenizer, batch, device_id, cast_dtype, autocast)


if __name__ == "__main__":
    main()