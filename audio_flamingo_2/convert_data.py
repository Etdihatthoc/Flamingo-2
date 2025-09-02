import pandas as pd
import json
import os
import librosa
from prompt_3 import system_prompt_fewshot
from tqdm import tqdm
import pandas as pd

def convert_csv_to_manifest(csv_path, split_name, output_path, data_root, audio_subdir):
    """
    Convert CSV format to Audio Flamingo 2 manifest format
    
    Args:
        csv_path: Path to your CSV file (train_pro.csv, val_pro.csv, test_pro.csv)
        split_name: 'train', 'val', or 'test'
        output_path: Where to save the manifest JSON
        data_root: Root path where audio files are stored
        audio_subdir: Subdirectory within data_root where audio files are located
    """
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Create manifest structure
    manifest = {
        "split": split_name,
        "split_path": audio_subdir,  # relative path from data_root
        "flamingo_task": "VSTEP-SpeakingScoring",
        "total_num": len(df),
        "data": {}
    }
    
    # Convert each row
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        # Get audio file path and duration
        audio_path = row['absolute_path']
        
        # Get audio duration
        try:
            duration = librosa.get_duration(filename=audio_path)
        except:
            duration = 10.0  # default fallback
        
        # Extract filename from path
        filename = audio_path#os.path.basename(audio_path)
        
        # Format the prompt (system prompt + transcript)
        prompt = f"{system_prompt_fewshot}\n\nTranscript: {row['text']}"
        
        # Format the expected output to match your current format
        total_score = (float(row['grammar']) + float(row['vocabulary']) + float(row['content'])) / 3.0
        
        output = (
            f"Grammar: {row['grammar']}/10\n"
            f"Vocabulary: {row['vocabulary']}/10\n"
            f"Discourse management: {row['content']}/10\n"
            f"Total: {total_score:.1f}/10"
        )
        
        # Add to manifest
        manifest["data"][str(idx)] = {
            "name": filename,
            "prompt": prompt,
            "output": output,
            "duration": duration
        }
    
    # Save manifest
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    print(f"Manifest saved to {output_path}")
    print(f"Total samples: {len(df)}")

if __name__ == "__main__":
    # Define paths
    data_root = "/home/user01/aiotlab/sondinh/DATA_Vocal"
    manifest_root = "./data/manifests"
    audio_subdir = ""  # since your audio files are directly in data_root
    
    # Convert train, validation, and test sets
    datasets = [
        ("train", "/home/user06/data/Speaking_VSTEP/Label/after_filter/train_new_clean.csv"),
        ("val", "/home/user06/data/Speaking_VSTEP/Label/after_filter/val_new_clean.csv"),
        ("test", "/home/user06/data/Speaking_VSTEP/Label/after_filter/test_new_clean.csv")
    ]
    
    for split, csv_path in datasets:
        output_path = os.path.join(manifest_root, f"VSTEP-SpeakingScoring/{split}.json")
        convert_csv_to_manifest(csv_path, split, output_path, data_r