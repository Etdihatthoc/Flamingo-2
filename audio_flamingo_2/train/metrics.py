# train/metrics.py
import re
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

def extract_scores_from_response(response_text):
    """Extract scores from model response"""
    default_score = 5.0
    
    try:
        response_text = response_text.lower().strip()
        
        # Extract grammar score
        grammar_match = re.search(r'grammar:\s*(\d+\.?\d*)', response_text)
        grammar_score = float(grammar_match.group(1)) if grammar_match else default_score
        
        # Extract vocabulary score  
        vocab_match = re.search(r'vocabulary:\s*(\d+\.?\d*)', response_text)
        vocab_score = float(vocab_match.group(1)) if vocab_match else default_score
        
        # Extract discourse management score
        discourse_patterns = [
            r'discourse management:\s*(\d+\.?\d*)',
            r'discourse:\s*(\d+\.?\d*)',
            r'content:\s*(\d+\.?\d*)'
        ]
        discourse_score = default_score
        for pattern in discourse_patterns:
            discourse_match = re.search(pattern, response_text)
            if discourse_match:
                discourse_score = float(discourse_match.group(1))
                break
        
        # Calculate total
        total_score = (grammar_score + vocab_score + discourse_score) / 3.0
        
        # Clamp scores to valid range [0, 10]
        grammar_score = max(0, min(10, grammar_score))
        vocab_score = max(0, min(10, vocab_score))
        discourse_score = max(0, min(10, discourse_score))
        total_score = max(0, min(10, total_score))
        
        return grammar_score, vocab_score, discourse_score, total_score
        
    except Exception as e:
        print(f"Score extraction failed: {e}")
        return default_score, default_score, default_score, default_score


def calculate_mae_metrics(predictions, ground_truths):
    """Calculate MAE for each score component"""
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have same length")
    
    # Convert to arrays
    pred_array = np.array(predictions)  # Shape: (N, 4)
    gt_array = np.array(ground_truths)   # Shape: (N, 4)
    
    # Calculate MAE for each component
    mae_grammar = mean_absolute_error(gt_array[:, 0], pred_array[:, 0])
    mae_vocab = mean_absolute_error(gt_array[:, 1], pred_array[:, 1])
    mae_discourse = mean_absolute_error(gt_array[:, 2], pred_array[:, 2])
    mae_total = mean_absolute_error(gt_array[:, 3], pred_array[:, 3])
    
    return {
        'mae_grammar': mae_grammar,
        'mae_vocabulary': mae_vocab,
        'mae_discourse': mae_discourse,
        'mae_total': mae_total,
        'mae_average': (mae_grammar + mae_vocab + mae_discourse) / 3.0
    }


@torch.no_grad()
def mae_test_epoch(model, data_config, clap_config, tokenizer, batch_size, autocast, cast_dtype, device_id, max_samples=50):
    """
    Run MAE test on validation set after each epoch
    Based on validation_losses implementation
    
    Returns:
        - mae_metrics: Dict with MAE scores
        - sample_logs: List of sample input/output for logging
    """
    from data.data import get_audiotext_dataloader
    
    print(f"Running MAE test on up to {max_samples} samples...")
    
    model.eval()
    
    # Setup tokens (same as validation_losses)
    media_token_id = tokenizer("<audio>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)["input_ids"][-1]
    
    # Get validation data
    all_valid_AudioTextDataInfo = get_audiotext_dataloader(
        data_config, clap_config, tokenizer, batch_size, split='test'
    )
    
    predictions = []
    ground_truths = []
    sample_logs = []
    
    # Use the first validation dataset
    dataset_name = list(all_valid_AudioTextDataInfo.keys())[0]
    validloader = all_valid_AudioTextDataInfo[dataset_name].dataloader
    
    print(f"Running MAE test on {dataset_name}")
    
    for idx, batch in tqdm(enumerate(validloader), desc="MAE Testing"):
        if idx >= max_samples:
            break
            
        try:
            # Load data with correct dtypes (same as validation_losses)
            audio_clips = batch["audio_clips"].to(device_id, dtype=cast_dtype, non_blocking=True)
            audio_embed_mask = batch["audio_embed_mask"].to(device_id, dtype=cast_dtype, non_blocking=True)
            
            # IMPORTANT: input_ids should be torch.long, not cast_dtype
            input_ids = batch["input_ids"].to(device_id, dtype=torch.long, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device_id, dtype=cast_dtype, non_blocking=True)
            filenames = batch["filenames"]
            
            # Extract ground truth by decoding full sequence
            full_decoded = tokenizer.decode(input_ids[0])
            
            # Find SEP token to split prompt and target
            if tokenizer.sep_token in full_decoded:
                parts = full_decoded.split(tokenizer.sep_token)
                if len(parts) >= 2:
                    prompt_text = parts[0].replace('<audio>', '').strip()
                    ground_truth_text = parts[-1].replace('<|endofchunk|>', '').replace(tokenizer.eos_token, '').replace(tokenizer.pad_token, '').strip()
                else:
                    continue
            else:
                # Skip samples without SEP token
                continue
            
            # Setup labels (same as validation_losses)
            labels = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100
            labels[:, :1] = -100
            labels[labels == tokenizer.encode("<audio>")[-1]] = -100
            
            sep_locations = labels == tokenizer.sep_token_id
            eoc_locations = labels == endofchunk_token_id
            
            # Mask labels (same logic as validation_losses)
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
                
                if labels[i][-1] not in [-100, tokenizer.eos_token_id, tokenizer.pad_token_id, endofchunk_token_id]:
                    for j in range(labels.shape[1]-1, -1, -1):
                        if labels[i][j] not in [-100, tokenizer.eos_token_id, endofchunk_token_id]:
                            labels[i][j] = -100
                        else:
                            break
            
            labels = labels.to(device_id)
            
            # Forward pass with autocast (same as validation_losses)
            with autocast():
                output = model(
                    audio_x=audio_clips,
                    audio_x_mask=audio_embed_mask,
                    lang_x=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
            
            # Get prediction from logits
            logits = output.logits  # Shape: (batch_size, seq_len, vocab_size)
            
            # Get predicted tokens (argmax over vocab dimension)
            pred_tokens = torch.argmax(logits, dim=-1)  # Shape: (batch_size, seq_len)
            
            # Find SEP position to get only the generated part
            sep_positions = (input_ids[0] == tokenizer.sep_token_id).nonzero()
            if len(sep_positions) > 0:
                sep_pos = sep_positions[-1].item()
                # Get predicted tokens after SEP
                pred_tokens_after_sep = pred_tokens[0, sep_pos+1:]
                
                # Decode prediction
                pred_text = tokenizer.decode(pred_tokens_after_sep)
                pred_text = pred_text.replace('<|endofchunk|>', '').replace(tokenizer.eos_token, '').replace(tokenizer.pad_token, '').strip()
            else:
                continue
            
            # Extract scores from both prediction and ground truth
            pred_scores = extract_scores_from_response(pred_text)
            gt_scores = extract_scores_from_response(ground_truth_text)
            
            predictions.append(pred_scores)
            ground_truths.append(gt_scores)
            
            # Save samples for logging (first 5 samples)
            if len(sample_logs) < 12325:
                filename = filenames[0] if isinstance(filenames[0], str) else filenames[0][0]
                sample_logs.append({
                    'audio_file': filename,
                    'prompt': prompt_text[:500] + "..." if len(prompt_text) > 500 else prompt_text,
                    'ground_truth': ground_truth_text,
                    'prediction': pred_text,
                    'gt_scores': gt_scores,
                    'pred_scores': pred_scores
                })
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    model.train()
    
    # Calculate MAE metrics
    if len(predictions) > 0:
        mae_metrics = calculate_mae_metrics(predictions, ground_truths)
        mae_metrics['num_samples'] = len(predictions)
    else:
        mae_metrics = {
            'mae_grammar': 999.0,
            'mae_vocabulary': 999.0, 
            'mae_discourse': 999.0,
            'mae_total': 999.0,
            'mae_average': 999.0,
            'num_samples': 0
        }
    
    return mae_metrics, sample_logs