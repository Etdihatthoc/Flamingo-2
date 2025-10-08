# Copyright (c) 2025 NVIDIA CORPORATION. 
#   Licensed under the MIT license.

# Adapted from https://github.com/mlfoundations/open_flamingo under the MIT license.
#   LICENSE is in incl_licenses directory.


import argparse
import functools
import os
import random
from tqdm import tqdm
import sys 
sys.path.append('../')
import yaml
import time

import numpy as np
import torch
from data.data import get_audiotext_dataloader


# @torch.no_grad()
# def validation_losses(model, data_config, clap_config, tokenizer, batch_size, autocast, cast_dtype, device_id, verbose=True):

#     model.eval()

#     @torch.no_grad()
#     def get_val_loss(validloader):

#         loss_sum = 0.0
#         for idx, batch in tqdm(enumerate(validloader)):

#             audio_clips = batch["audio_clips"].to(device_id, dtype=cast_dtype, non_blocking=True)
#             audio_embed_mask = batch["audio_embed_mask"].to(device_id, dtype=cast_dtype, non_blocking=True)
#             input_ids = batch["input_ids"].to(device_id, dtype=cast_dtype, non_blocking=True)
#             attention_mask = batch["attention_mask"].to(device_id, dtype=cast_dtype, non_blocking=True)

#             labels = input_ids.clone()
#             labels[labels == tokenizer.pad_token_id] = -100
#             labels[:, :1] = -100
#             labels[labels == tokenizer.encode("<audio>")[-1]] = -100

#             sep_locations = labels == tokenizer.sep_token_id
#             eoc_locations = labels == endofchunk_token_id

#             for i in range(labels.shape[0]):
#                 shouldmask = True
#                 for j in range(labels.shape[1]):
#                     if shouldmask and (labels[i][j] != tokenizer.eos_token_id):
#                         masked_value = -100
#                     else:
#                         masked_value = labels[i][j]

#                     if labels[i][j] == tokenizer.sep_token_id:
#                         shouldmask = False
#                     elif labels[i][j] == endofchunk_token_id:
#                         shouldmask = True
                    
#                     labels[i][j] = masked_value
                
#                 if labels[i][-1] not in [-100, tokenizer.eos_token_id, tokenizer.pad_token_id, endofchunk_token_id]:
#                     for j in range(labels.shape[1]-1, -1, -1):
#                         if labels[i][j] not in [-100, tokenizer.eos_token_id, endofchunk_token_id]:
#                             labels[i][j] = -100
#                         else:
#                             break

#             labels = labels.to(device_id)

#             with autocast():
#                 output = model(
#                     audio_x=audio_clips,
#                     audio_x_mask=audio_embed_mask,
#                     lang_x=input_ids,
#                     attention_mask=attention_mask,
#                     labels=labels
#                 )
#                 valid_loss_no_multiplier = output.loss.item()
#                 loss_sum += valid_loss_no_multiplier

#         return loss_sum / ((idx+1) * batch_size)

#     media_token_id = tokenizer("<audio>", add_special_tokens=False)["input_ids"][-1]
#     assert media_token_id == tokenizer.encode("<audio>")[-1]
#     endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)["input_ids"][-1]

#     valid_losses = {}
#     all_valid_AudioTextDataInfo = get_audiotext_dataloader(data_config, clap_config, tokenizer, batch_size, split='val')
#     for valid_dataset_name in all_valid_AudioTextDataInfo:
#         if verbose:
#             print('computing validation loss on {}'.format(valid_dataset_name))

#         validloader = all_valid_AudioTextDataInfo[valid_dataset_name].dataloader 
#         valid_losses[valid_dataset_name] = get_val_loss(validloader)

#         if verbose:
#             print('validation loss on {} is {:.3f}'.format(valid_dataset_name, valid_losses[valid_dataset_name]))
    
#     model.train() 

#     return valid_losses

@torch.no_grad()
def validation_losses(model, data_config, clap_config, tokenizer, batch_size, autocast, cast_dtype, device_id, verbose=False):
    """
    Compute validation losses using AUTOREGRESSIVE GENERATION
    """
    
    def compute_loss_one_dataset(dataloader):
        """Compute loss for one validation dataset using generation"""
        loss_sum = 0
        
        for idx, batch in enumerate(dataloader):
            audio_clips = batch["audio_clips"].to(device_id, dtype=cast_dtype, non_blocking=True)
            audio_embed_mask = batch["audio_embed_mask"].to(device_id, dtype=cast_dtype, non_blocking=True)
            
            # Input: only prompt (no ground truth)
            input_ids = batch["input_ids"].to(device_id, dtype=torch.long, non_blocking=True)
            
            # Ground truth for comparison
            ground_truth_ids = batch["ground_truth_ids"].to(device_id, dtype=torch.long, non_blocking=True)
            ground_truth_texts = batch["ground_truth_texts"]
            
            # ===== AUTOREGRESSIVE GENERATION =====
            with autocast():
                generated_ids = model.generate(
                    audio_x=audio_clips,
                    audio_x_mask=audio_embed_mask,
                    lang_x=input_ids,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    max_new_tokens=256,
                    do_sample=False,
                )[0]
            
            # Decode generated output
            generated_text = tokenizer.decode(generated_ids).split(tokenizer.sep_token)[-1]
            generated_text = generated_text.replace('<|endofchunk|>', '').replace(
                tokenizer.eos_token, '').replace(tokenizer.pad_token, '').strip()
            
            # Decode ground truth
            gt_text = ground_truth_texts[0]  # First in batch
            
            # Compute loss by comparing tokens
            # Re-tokenize both to get comparable token sequences
            gen_tokens = tokenizer(generated_text, return_tensors="pt")["input_ids"][0]
            gt_tokens = tokenizer(gt_text, return_tensors="pt")["input_ids"][0]
            
            # Compute cross-entropy-like loss manually
            min_len = min(len(gen_tokens), len(gt_tokens))
            matches = (gen_tokens[:min_len] == gt_tokens[:min_len]).float().mean()
            loss = 1.0 - matches  # Simple token-level accuracy loss
            
            loss_sum += loss.item()
        
        return loss_sum / ((idx+1) * batch_size)

    # Setup
    media_token_id = tokenizer("<audio>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)["input_ids"][-1]

    valid_losses = {}
    all_valid_AudioTextDataInfo = get_audiotext_dataloader(data_config, clap_config, tokenizer, batch_size, split='val')
    
    for valid_dataset_name in all_valid_AudioTextDataInfo:
        if verbose:
            print(f'Computing validation loss on {valid_dataset_name}')

        validloader = all_valid_AudioTextDataInfo[valid_dataset_name].dataloader
        valid_losses[valid_dataset_name] = compute_loss_one_dataset(validloader)
        
        if verbose:
            print(f'{valid_dataset_name}: {valid_losses[valid_dataset_name]:.4f}')

    return valid_losses

if __name__ == "__main__":
    from src.factory import create_model_and_transforms
    from train_utils import Dict2Class, get_autocast, get_cast_dtype

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='../configs/config.yaml', help='yaml config path')
    parsed_args = parser.parse_args()

    config = yaml.load(open(parsed_args.config), Loader=yaml.FullLoader)
    data_config = config['data_config']
    model_config = config['model_config']
    clap_config = config['clap_config']
    args = Dict2Class(config['train_config'])

    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # disable the tokenizer parallelism warning
    model, tokenizer = create_model_and_transforms(
        **model_config,
        clap_config=clap_config,
        use_local_files=args.offline,
        gradient_checkpointing=args.gradient_checkpointing,
        freeze_lm_embeddings=args.freeze_lm_embeddings,
    )

    device_id = 0
    model = model.to(device_id)
    
    autocast = get_autocast(
        args.precision, cache_enabled=(not args.fsdp)
    )  # if fsdp, disable cache to save memory
    cast_dtype = get_cast_dtype(args.precision)

    valid_losses = validation_losses(
        model, 
        data_config, 
        clap_config,
        tokenizer, 
        args.batch_size, 
        autocast, 
        cast_dtype,
        device_id,
        verbose=True
    )

    print(valid_losses)