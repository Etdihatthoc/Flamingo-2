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
    Run MAE test on validation set after each epoch using autoregressive generation
    
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
    
    print(f"Running MAE test on {dataset_name}")
    
    for idx, batch in tqdm(enumerate(validloader), desc="MAE Val Testing"):
        if idx >= max_samples:
            break
            
        try:
            # Load data
            audio_clips = batch["audio_clips"].to(device_id, dtype=cast_dtype, non_blocking=True)
            audio_embed_mask = batch["audio_embed_mask"].to(device_id, dtype=cast_dtype, non_blocking=True)
            input_ids = batch["input_ids"].to(device_id, dtype=torch.long, non_blocking=True)
            filenames = batch["filenames"]
            
            # Process each sample in the batch
            for sample_idx in range(input_ids.shape[0]):
                input_id = input_ids[sample_idx]
                filename = filenames[sample_idx]
                if type(filename) is list:
                    filename = filename[-1]
                
                # Find last SEP token to extract prompt
                sep_location = None
                for i in range(len(input_id)-1, -1, -1):
                    if input_id[i] == sep_token_id:
                        sep_location = i
                        break
                
                if sep_location is None:
                    continue
                
                # Extract prompt (everything up to and including SEP)
                prompt = input_id[:sep_location+1]
                
                # Decode ground truth
                full_decoded = tokenizer.decode(input_id)
                if tokenizer.sep_token in full_decoded:
                    parts = full_decoded.split(tokenizer.sep_token)
                    if len(parts) >= 2:
                        prompt_text = parts[0].replace('<audio>', '').strip()
                        ground_truth_text = parts[-1].replace('<|endofchunk|>', '').replace(tokenizer.eos_token, '').replace(tokenizer.pad_token, '').strip()
                    else:
                        continue
                else:
                    continue
                
                # Generate prediction using autoregressive generation
                with autocast():
                    output = model.generate(
                        audio_x=audio_clips[sample_idx].unsqueeze(0),
                        audio_x_mask=audio_embed_mask[sample_idx].unsqueeze(0),
                        lang_x=prompt.unsqueeze(0),
                        eos_token_id=tokenizer.eos_token_id,
                        max_new_tokens=256,
                        temperature=1.0,  # Use same default as inference
                        num_beams=1,  # Greedy decoding
                    )[0]
                
                # Decode prediction
                pred_text = tokenizer.decode(output).split(tokenizer.sep_token)[-1].replace('<|endofchunk|>', '').replace(tokenizer.eos_token, '').replace(tokenizer.pad_token, '').strip()
                
                # Extract scores from both prediction and ground truth
                pred_scores = extract_scores_from_response(pred_text)
                gt_scores = extract_scores_from_response(ground_truth_text)
                
                predictions.append(pred_scores)
                ground_truths.append(gt_scores)

                # Save samples for logging (first 5438 samples)
                if len(sample_logs) < 5438:
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
    Run MAE test on test set using autoregressive generation
    
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
    
    for idx, batch in tqdm(enumerate(validloader), desc="MAE Testing"):
        if idx >= max_samples:
            break
            
        try:
            # Load data
            audio_clips = batch["audio_clips"].to(device_id, dtype=cast_dtype, non_blocking=True)
            audio_embed_mask = batch["audio_embed_mask"].to(device_id, dtype=cast_dtype, non_blocking=True)
            input_ids = batch["input_ids"].to(device_id, dtype=torch.long, non_blocking=True)
            filenames = batch["filenames"]
            
            # Process each sample in the batch
            for sample_idx in range(input_ids.shape[0]):
                input_id = input_ids[sample_idx]
                filename = filenames[sample_idx]
                if type(filename) is list:
                    filename = filename[-1]
                
                # Find last SEP token to extract prompt
                sep_location = None
                for i in range(len(input_id)-1, -1, -1):
                    if input_id[i] == sep_token_id:
                        sep_location = i
                        break
                
                if sep_location is None:
                    continue
                
                # Extract prompt (everything up to and including SEP)
                prompt = input_id[:sep_location+1]
                
                # Decode ground truth
                full_decoded = tokenizer.decode(input_id)
                if tokenizer.sep_token in full_decoded:
                    parts = full_decoded.split(tokenizer.sep_token)
                    if len(parts) >= 2:
                        prompt_text = parts[0].replace('<audio>', '').strip()
                        ground_truth_text = parts[-1].replace('<|endofchunk|>', '').replace(tokenizer.eos_token, '').replace(tokenizer.pad_token, '').strip()
                    else:
                        continue
                else:
                    continue
                
                # Generate prediction using autoregressive generation
                with autocast():
                    output = model.generate(
                        audio_x=audio_clips[sample_idx].unsqueeze(0),
                        audio_x_mask=audio_embed_mask[sample_idx].unsqueeze(0),
                        lang_x=prompt.unsqueeze(0),
                        eos_token_id=tokenizer.eos_token_id,
                        max_new_tokens=256,
                        temperature=1.0,  # Use same default as inference
                        num_beams=1,  # Greedy decoding
                    )[0]
                
                # Decode prediction
                pred_text = tokenizer.decode(output).split(tokenizer.sep_token)[-1].replace('<|endofchunk|>', '').replace(tokenizer.eos_token, '').replace(tokenizer.pad_token, '').strip()
                
                # Extract scores from both prediction and ground truth
                pred_scores = extract_scores_from_response(pred_text)
                gt_scores = extract_scores_from_response(ground_truth_text)
                
                predictions.append(pred_scores)
                ground_truths.append(gt_scores)
                
                # Save samples for logging (first 12325 samples)
                if len(sample_logs) < 12325:
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