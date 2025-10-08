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
        
        # Extract vocabulary score  
        vocab_match = re.search(r'vocabulary:\s*(\d+\.?\d*)', response_text)
        vocab_score = float(vocab_match.group(1)) if vocab_match else default_score

        # Extract grammar score
        grammar_match = re.search(r'grammar:\s*(\d+\.?\d*)', response_text)
        grammar_score = float(grammar_match.group(1)) if grammar_match else default_score
        
        # Extract pronunciation score
        pronunciation_match = re.search(r'pronunciation:\s*(\d+\.?\d*)', response_text)
        pronunciation_score = float(pronunciation_match.group(1)) if pronunciation_match else default_score
        
        # Extract fluency score
        fluency_match = re.search(r'fluency:\s*(\d+\.?\d*)', response_text)
        fluency_score = float(fluency_match.group(1)) if fluency_match else default_score
        
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
        total_score = (grammar_score + vocab_score + pronunciation_score + 
                      fluency_score + discourse_score) / 5.0
        
        total_score = round(total_score * 2) / 2.0
        # Clamp scores to valid range [0, 10]
        
        total_score = max(0, min(10, total_score))
        vocab_score = max(0, min(10, vocab_score))
        grammar_score = max(0, min(10, grammar_score))
        pronunciation_score = max(0, min(10, pronunciation_score))
        fluency_score = max(0, min(10, fluency_score))
        discourse_score = max(0, min(10, discourse_score))
        
        
        return total_score, vocab_score , grammar_score, pronunciation_score, fluency_score, discourse_score
        
    except Exception as e:
        print(f"Score extraction failed: {e}")
        return default_score, default_score, default_score, default_score, default_score, default_score

def calculate_mae_metrics(predictions, ground_truths):
    """Calculate MAE for each score component"""
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have same length")
    
    # Convert to arrays
    pred_array = np.array(predictions)  # Shape: (N, 6)
    gt_array = np.array(ground_truths)   # Shape: (N, 6)
    
    # Calculate MAE for each component
    mae_total = mean_absolute_error(gt_array[:, 0], pred_array[:, 0])
    mae_vocab = mean_absolute_error(gt_array[:, 1], pred_array[:, 1])
    mae_grammar = mean_absolute_error(gt_array[:, 2], pred_array[:, 2])
    mae_pronunciation = mean_absolute_error(gt_array[:, 3], pred_array[:, 3])
    mae_fluency = mean_absolute_error(gt_array[:, 4], pred_array[:, 4])
    mae_discourse = mean_absolute_error(gt_array[:, 5], pred_array[:, 5])

    
    return {
        'mae_total': mae_total,
        'mae_vocabulary': mae_vocab,
        'mae_grammar': mae_grammar,
        'mae_pronunciation': mae_pronunciation,
        'mae_fluency': mae_fluency,
        'mae_discourse': mae_discourse,
        'mae_average': (mae_grammar + mae_vocab + mae_pronunciation + mae_fluency + mae_discourse) / 5.0
    }


@torch.no_grad()
def mae_val_epoch(model, data_config, clap_config, tokenizer, batch_size, autocast, cast_dtype, device_id, max_samples=50):
    """
    Run MAE validation - uses model.generate() like inference.py
    Input format: <audio>{prompt}{sep_token}
    
    Returns:
        - mae_metrics: Dict with MAE scores
        - sample_logs: List of sample input/output for logging
    """
    from data.data import get_audiotext_dataloader
    
    print(f"Running MAE val on up to {max_samples} samples...")
    
    model.eval()
    
    # Setup tokens
    media_token_id = tokenizer("<audio>", add_special_tokens=False)["input_ids"][-1]
    sep_token_id = tokenizer.sep_token_id
    
    # Get validation data
    all_valid_AudioTextDataInfo = get_audiotext_dataloader(
        data_config, clap_config, tokenizer, batch_size, split='val'
    )
    
    predictions = []
    ground_truths = []
    sample_logs = []
    
    # Use the first validation dataset
    dataset_name = list(all_valid_AudioTextDataInfo.keys())[0]
    validloader = all_valid_AudioTextDataInfo[dataset_name].dataloader
    
    print(f"Running MAE validation on {dataset_name}")
    
    for idx, batch in tqdm(enumerate(validloader), desc="MAE Val"):
        if idx >= max_samples:
            break
            
        try:
            # Load data with correct dtypes
            audio_clips = batch["audio_clips"].to(device_id, dtype=cast_dtype, non_blocking=True)
            audio_embed_mask = batch["audio_embed_mask"].to(device_id, dtype=cast_dtype, non_blocking=True)
            
            # Full input_ids from data (includes prompt + sep + output)
            input_ids_full = batch["input_ids"].to(device_id, dtype=torch.long, non_blocking=True)
            filenames = batch["filenames"]
            
            # Verify <audio> token is present
            has_audio_token = (input_ids_full[0] == media_token_id).any()
            if not has_audio_token:
                print(f"WARNING: Sample {idx} missing <audio> token! Skipping...")
                continue
            
            # Find SEP token position
            sep_positions = (input_ids_full[0] == sep_token_id).nonzero(as_tuple=True)[0]
            if len(sep_positions) == 0:
                print(f"WARNING: Sample {idx} missing SEP token! Skipping...")
                continue
            
            sep_pos = sep_positions[-1].item()
            
            # Extract ground truth (everything after SEP)
            ground_truth_ids = input_ids_full[0, sep_pos+1:]
            ground_truth_text = tokenizer.decode(ground_truth_ids)
            ground_truth_text = ground_truth_text.replace('<|endofchunk|>', '').replace(
                tokenizer.eos_token, '').replace(tokenizer.pad_token, '').strip()
            
            # Extract prompt (everything before SEP, excluding <audio>)
            prompt_ids = input_ids_full[0, :sep_pos]
            prompt_text = tokenizer.decode(prompt_ids).replace('<audio>', '').strip()
            
            # Truncate input to only prompt + SEP (for generation, like inference.py)
            # Format: <audio>{prompt}{sep_token}
            input_ids_truncated = input_ids_full[:, :sep_pos+1]
            
            # Verify <audio> token is still present after truncation
            has_audio_after_truncate = (input_ids_truncated[0] == media_token_id).any()
            if not has_audio_after_truncate:
                print(f"ERROR: Sample {idx} lost <audio> after truncate!")
                continue
            
            # Generate using model.generate() like inference.py
            with autocast():
                generated_ids = model.generate(
                    audio_x=audio_clips,
                    audio_x_mask=audio_embed_mask,
                    lang_x=input_ids_truncated,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    max_new_tokens=256,
                    do_sample=False,
                )[0]
            
            # Decode generated output (split after SEP token)
            output_decoded = tokenizer.decode(generated_ids).split(tokenizer.sep_token)[-1]
            output_decoded = output_decoded.replace('<|endofchunk|>', '').replace(
                tokenizer.eos_token, '').replace(tokenizer.pad_token, '').strip()
            
            pred_text = output_decoded.lower()

            # Extract scores from both prediction and ground truth
            pred_scores = extract_scores_from_response(pred_text)
            gt_scores = extract_scores_from_response(ground_truth_text)
            
            predictions.append(pred_scores)
            ground_truths.append(gt_scores)

            # Save samples for logging (first 5438 samples)
            if len(sample_logs) < 5438:
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
            'mae_total': 999.0,
            'mae_vocabulary': 999.0,
            'mae_grammar': 999.0,
            'mae_pronunciation': 999.0,
            'mae_fluency': 999.0,
            'mae_discourse': 999.0,
            'mae_average': 999.0,
            'num_samples': 0
        }
    
    return mae_metrics, sample_logs

@torch.no_grad()
def mae_test_epoch(model, data_config, clap_config, tokenizer, batch_size, autocast, cast_dtype, device_id, max_samples=50):
    """
    Run MAE test on test set - uses model.generate() like inference.py
    Input format: <audio>{prompt}{sep_token}
    
    Returns:
        - mae_metrics: Dict with MAE scores
        - sample_logs: List of sample input/output for logging
    """
    from data.data import get_audiotext_dataloader
    
    print(f"Running MAE test on up to {max_samples} samples...")
    
    model.eval()
    
    # Setup tokens
    media_token_id = tokenizer("<audio>", add_special_tokens=False)["input_ids"][-1]
    sep_token_id = tokenizer.sep_token_id
    
    # Get test data
    all_valid_AudioTextDataInfo = get_audiotext_dataloader(
        data_config, clap_config, tokenizer, batch_size, split='test'
    )
    
    predictions = []
    ground_truths = []
    sample_logs = []
    
    # Use the first test dataset
    dataset_name = list(all_valid_AudioTextDataInfo.keys())[0]
    validloader = all_valid_AudioTextDataInfo[dataset_name].dataloader
    
    print(f"Running MAE test on {dataset_name}")
    
    for idx, batch in tqdm(enumerate(validloader), desc="MAE Test"):
        if idx >= max_samples:
            break
            
        try:
            # Load data with correct dtypes
            audio_clips = batch["audio_clips"].to(device_id, dtype=cast_dtype, non_blocking=True)
            audio_embed_mask = batch["audio_embed_mask"].to(device_id, dtype=cast_dtype, non_blocking=True)
            
            # Full input_ids from data (includes prompt + sep + output)
            input_ids_full = batch["input_ids"].to(device_id, dtype=torch.long, non_blocking=True)
            filenames = batch["filenames"]
            
            # Verify <audio> token is present
            has_audio_token = (input_ids_full[0] == media_token_id).any()
            if not has_audio_token:
                print(f"WARNING: Sample {idx} missing <audio> token! Skipping...")
                continue
            
            # Find SEP token position
            sep_positions = (input_ids_full[0] == sep_token_id).nonzero(as_tuple=True)[0]
            if len(sep_positions) == 0:
                print(f"WARNING: Sample {idx} missing SEP token! Skipping...")
                continue
            
            sep_pos = sep_positions[-1].item()
            
            # Extract ground truth (everything after SEP)
            ground_truth_ids = input_ids_full[0, sep_pos+1:]
            ground_truth_text = tokenizer.decode(ground_truth_ids)
            ground_truth_text = ground_truth_text.replace('<|endofchunk|>', '').replace(
                tokenizer.eos_token, '').replace(tokenizer.pad_token, '').strip()
            
            # Extract prompt (everything before SEP, excluding <audio>)
            prompt_ids = input_ids_full[0, :sep_pos]
            prompt_text = tokenizer.decode(prompt_ids).replace('<audio>', '').strip()
            
            # Truncate input to only prompt + SEP (for generation, like inference.py)
            # Format: <audio>{prompt}{sep_token}
            input_ids_truncated = input_ids_full[:, :sep_pos+1]
            
            # Verify <audio> token is still present after truncation
            has_audio_after_truncate = (input_ids_truncated[0] == media_token_id).any()
            if not has_audio_after_truncate:
                print(f"ERROR: Sample {idx} lost <audio> after truncate!")
                continue
            
            # Generate using model.generate() like inference.py
            with autocast():
                generated_ids = model.generate(
                    audio_x=audio_clips,
                    audio_x_mask=audio_embed_mask,
                    lang_x=input_ids_truncated,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    max_new_tokens=256,
                    do_sample=False,
                )[0]
            
            # Decode generated output (split after SEP token)
            output_decoded = tokenizer.decode(generated_ids).split(tokenizer.sep_token)[-1]
            output_decoded = output_decoded.replace('<|endofchunk|>', '').replace(
                tokenizer.eos_token, '').replace(tokenizer.pad_token, '').strip()
            
            pred_text = output_decoded.lower()
            
            # Extract scores from both prediction and ground truth
            pred_scores = extract_scores_from_response(pred_text)
            gt_scores = extract_scores_from_response(ground_truth_text)
            
            predictions.append(pred_scores)
            ground_truths.append(gt_scores)
            
            # Save samples for logging
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
            'mae_total': 999.0,
            'mae_vocabulary': 999.0,
            'mae_grammar': 999.0,
            'mae_pronunciation': 999.0,
            'mae_fluency': 999.0,
            'mae_discourse': 999.0,
            'mae_average': 999.0,
            'num_samples': 0
        }
    
    return mae_metrics, sample_logs