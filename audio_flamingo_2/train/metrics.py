# train/metrics.py
import re
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error

def extract_scores_from_response(response_text):
    """
    Extract scores from model response
    Expected format: Grammar: X.X/10, Vocabulary: X.X/10, Discourse management: X.X/10
    Returns: (grammar_score, vocabulary_score, discourse_score, total_score)
    """
    # Default scores if extraction fails
    default_score = 5.0
    
    try:
        # Clean the response
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
    """
    Calculate MAE for each score component
    
    Args:
        predictions: List of (grammar, vocab, discourse, total) tuples from model
        ground_truths: List of (grammar, vocab, discourse, total) tuples from dataset
    
    Returns:
        Dict with MAE for each component
    """
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


def mae_test_epoch(model, data_config, clap_config, tokenizer, device_id, max_samples=100):
    """
    Run MAE test on validation set after each epoch
    
    Returns:
        - mae_metrics: Dict with MAE scores
        - sample_logs: List of sample input/output for logging
    """
    from data.data import get_audiotext_dataloader
    from train_utils import get_autocast, get_cast_dtype
    
    print(f"Running MAE test on {max_samples} samples...")
    
    model.eval()
    
    # Get validation data
    valid_dataloaders = get_audiotext_dataloader(
        data_config, clap_config, tokenizer, batch_size=1, split='val'
    )
    
    predictions = []
    ground_truths = []
    sample_logs = []
    
    # Use the first validation dataset
    dataset_name = list(valid_dataloaders.keys())[0]
    validloader = valid_dataloaders[dataset_name].dataloader
    
    autocast = get_autocast('fp16', cache_enabled=True)
    
    with torch.no_grad():
        for idx, batch in enumerate(validloader):
            if idx >= max_samples:
                break
                
            try:
                audio_clips = batch["audio_clips"].to(device_id, non_blocking=True)
                audio_embed_mask = batch["audio_embed_mask"].to(device_id, non_blocking=True)
                input_ids = batch["input_ids"].to(device_id, non_blocking=True)
                filenames = batch["filenames"]
                
                # Extract prompt and ground truth
                input_id = input_ids[0]
                sep_token_id = tokenizer.sep_token_id
                
                # Find SEP token position
                sep_positions = (input_id == sep_token_id).nonzero()
                if len(sep_positions) == 0:
                    continue
                    
                sep_pos = sep_positions[-1].item()
                prompt = input_id[:sep_pos+1]
                
                # Decode ground truth (everything after SEP)
                full_decoded = tokenizer.decode(input_id)
                if tokenizer.sep_token in full_decoded:
                    ground_truth_text = full_decoded.split(tokenizer.sep_token)[-1]
                    ground_truth_text = ground_truth_text.replace('<|endofchunk|>', '').replace(tokenizer.eos_token, '').strip()
                else:
                    continue
                
                # Generate prediction
                with autocast():
                    output = model.generate(
                        audio_x=audio_clips,
                        audio_x_mask=audio_embed_mask,
                        lang_x=prompt.unsqueeze(0),
                        eos_token_id=tokenizer.eos_token_id,
                        max_new_tokens=200,
                        do_sample=True,
                        top_p=0.9,
                        temperature=0.7
                    )[0]
                
                # Decode prediction
                pred_text = tokenizer.decode(output)
                if tokenizer.sep_token in pred_text:
                    pred_text = pred_text.split(tokenizer.sep_token)[-1]
                    pred_text = pred_text.replace('<|endofchunk|>', '').replace(tokenizer.eos_token, '').strip()
                
                # Extract scores
                pred_scores = extract_scores_from_response(pred_text)
                gt_scores = extract_scores_from_response(ground_truth_text)
                
                predictions.append(pred_scores)
                ground_truths.append(gt_scores)
                
                # Save samples for logging (first 5 samples)
                if len(sample_logs) < 5:
                    prompt_text = tokenizer.decode(prompt).replace('<audio>', '').replace('<SEP>', '').strip()
                    sample_logs.append({
                        'audio_file': filenames[0] if isinstance(filenames[0], str) else filenames[0][0],
                        'prompt': prompt_text[:500] + "..." if len(prompt_text) > 500 else prompt_text,
                        'ground_truth': ground_truth_text,
                        'prediction': pred_text,
                        'gt_scores': gt_scores,
                        'pred_scores': pred_scores
                    })
                
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
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