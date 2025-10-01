#!/usr/bin/env python3
"""
Extract and calculate MAE metrics from test results CSV
Based on metrics.py and train_lora.py logic
"""

import pandas as pd
from sklearn.metrics import mean_absolute_error
import numpy as np
import re
import argparse
import sys
import os

def extract_scores_from_response(response_text):
    """Extract scores from model response - same logic as metrics.py"""
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
    """Calculate MAE for each score component - same logic as metrics.py"""
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


def extract_and_calculate_mae(csv_file_path, output_file=None, corrected_csv_name="test_result_corrected.csv"):
    """
    Extract scores from CSV file and calculate MAE metrics
    """
    print(f"Loading CSV file: {csv_file_path}")
    
    try:
        # Read CSV file
        df = pd.read_csv(csv_file_path)
        print(f"Loaded {len(df)} samples from CSV")
        
        # Check required columns
        required_cols = ["Ground Truth", "Prediction"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        predictions = []
        ground_truths = []
        failed_extractions = 0
        
        print("Extracting scores from text responses...")
        
        for idx, row in df.iterrows():
            # Extract scores from ground truth text
            gt_text = str(row["Ground Truth"]) if pd.notna(row["Ground Truth"]) else ""
            gt_scores = extract_scores_from_response(gt_text)
            
            # Extract scores from prediction text  
            pred_text = str(row["Prediction"]) if pd.notna(row["Prediction"]) else ""
            pred_scores = extract_scores_from_response(pred_text)
            
            # Check if extraction was successful (not all default scores)
            if (gt_scores == (5.0, 5.0, 5.0, 5.0) and "grammar:" not in gt_text.lower()) or \
               (pred_scores == (5.0, 5.0, 5.0, 5.0) and "grammar:" not in pred_text.lower()):
                failed_extractions += 1
                if idx < 5:  # Show first few failures for debugging
                    print(f"Warning: Default scores used for sample {idx}")
                    print(f"  GT: {gt_text[:100]}...")
                    print(f"  Pred: {pred_text[:100]}...")
            
            ground_truths.append(gt_scores)
            predictions.append(pred_scores)
        
        print(f"Extraction complete. Failed extractions: {failed_extractions}/{len(df)}")
        
        # Calculate MAE metrics
        print("Calculating MAE metrics...")
        mae_metrics = calculate_mae_metrics(predictions, ground_truths)
        
        # Print results in same format as terminal output
        print("\n" + "="*50)
        print("MAE METRICS RESULTS")
        print("="*50)
        print(f"Grammar MAE:    {mae_metrics['mae_grammar']:.4f}")
        print(f"Vocabulary MAE: {mae_metrics['mae_vocabulary']:.4f}")
        print(f"Discourse MAE:  {mae_metrics['mae_discourse']:.4f}")
        print(f"Total MAE:      {mae_metrics['mae_total']:.4f}")
        print(f"Average MAE:    {mae_metrics['mae_average']:.4f}")
        print(f"Samples tested: {len(predictions)}")
        print("="*50)
        
        # Create corrected CSV with same structure as original test_result.csv
        corrected_df = pd.DataFrame({
            'Audio File': df['Audio File'] if 'Audio File' in df.columns else [f"sample_{i}" for i in range(len(df))],
            'Prompt': df['Prompt'] if 'Prompt' in df.columns else ["" for _ in range(len(df))],
            'Ground Truth': df['Ground Truth'] if 'Ground Truth' in df.columns else ["" for _ in range(len(df))],
            'Prediction': df['Prediction'] if 'Prediction' in df.columns else ["" for _ in range(len(df))],
            'GT Grammar': [gt[0] for gt in ground_truths],
            'GT Vocab': [gt[1] for gt in ground_truths], 
            'GT Discourse': [gt[2] for gt in ground_truths],
            'GT Total': [gt[3] for gt in ground_truths],
            'Pred Grammar': [pred[0] for pred in predictions],
            'Pred Vocab': [pred[1] for pred in predictions],
            'Pred Discourse': [pred[2] for pred in predictions], 
            'Pred Total': [pred[3] for pred in predictions]
        })
        
        # Create detailed results dataframe for analysis
        results_df = pd.DataFrame({
            'Audio_File': df['Audio File'] if 'Audio File' in df.columns else range(len(df)),
            'GT_Grammar': [gt[0] for gt in ground_truths],
            'GT_Vocabulary': [gt[1] for gt in ground_truths], 
            'GT_Discourse': [gt[2] for gt in ground_truths],
            'GT_Total': [gt[3] for gt in ground_truths],
            'Pred_Grammar': [pred[0] for pred in predictions],
            'Pred_Vocabulary': [pred[1] for pred in predictions],
            'Pred_Discourse': [pred[2] for pred in predictions], 
            'Pred_Total': [pred[3] for pred in predictions],
            'Grammar_Error': [abs(gt[0] - pred[0]) for gt, pred in zip(ground_truths, predictions)],
            'Vocabulary_Error': [abs(gt[1] - pred[1]) for gt, pred in zip(ground_truths, predictions)],
            'Discourse_Error': [abs(gt[2] - pred[2]) for gt, pred in zip(ground_truths, predictions)],
            'Total_Error': [abs(gt[3] - pred[3]) for gt, pred in zip(ground_truths, predictions)]
        })
        
        # Save corrected CSV with same structure as original
        corrected_df.to_csv(corrected_csv_name, index=False)
        print(f"\nCorrected CSV saved to: {corrected_csv_name}")
        
        # Save detailed results if output file specified
        if output_file:
            results_df.to_csv(output_file, index=False)
            print(f"Detailed analysis saved to: {output_file}")
        
        # Save summary metrics
        summary_file = "mae_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("MAE METRICS SUMMARY\n")
            f.write("="*50 + "\n")
            f.write(f"Grammar MAE:    {mae_metrics['mae_grammar']:.4f}\n")
            f.write(f"Vocabulary MAE: {mae_metrics['mae_vocabulary']:.4f}\n")
            f.write(f"Discourse MAE:  {mae_metrics['mae_discourse']:.4f}\n")
            f.write(f"Total MAE:      {mae_metrics['mae_total']:.4f}\n")
            f.write(f"Average MAE:    {mae_metrics['mae_average']:.4f}\n")
            f.write(f"Samples tested: {len(predictions)}\n")
            f.write(f"Failed extractions: {failed_extractions}\n")
        
        print(f"Summary saved to: {summary_file}")
        
        return mae_metrics, results_df, corrected_df
        
    except Exception as e:
        print(f"Error processing CSV file: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    parser = argparse.ArgumentParser(description='Extract scores and calculate MAE metrics from test results CSV')
    parser.add_argument('csv_file', help='Path to the test results CSV file')
    parser.add_argument('--output', '-o', help='Output file for detailed analysis results (optional)')
    parser.add_argument('--corrected', '-c', default='test_result_corrected.csv', help='Output file for corrected CSV (default: test_result_corrected.csv)')
    parser.add_argument('--sample', '-s', type=int, help='Process only first N samples (for testing)')
    
    args = parser.parse_args()
    
    # Check if CSV file exists
    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file not found: {args.csv_file}")
        sys.exit(1)
    
    # Load and process data
    if args.sample:
        print(f"Processing first {args.sample} samples only")
        df = pd.read_csv(args.csv_file, nrows=args.sample)
        temp_file = "temp_sample.csv"
        df.to_csv(temp_file, index=False)
        mae_metrics, results_df, corrected_df = extract_and_calculate_mae(temp_file, args.output, args.corrected)
        os.remove(temp_file)
    else:
        mae_metrics, results_df, corrected_df = extract_and_calculate_mae(args.csv_file, args.output, args.corrected)
    
    if mae_metrics is None:
        print("Failed to calculate MAE metrics")
        sys.exit(1)
    
    print("\nExtraction and calculation completed successfully!")


if __name__ == "__main__":
    main()