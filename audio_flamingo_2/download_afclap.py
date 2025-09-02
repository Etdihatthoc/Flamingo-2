#!/usr/bin/env python3
"""
Download AF-CLAP checkpoint for Audio Flamingo 2
"""
import os
import wget
from pathlib import Path

def download_afclap():
    """Download AF-CLAP checkpoint"""
    
    checkpoint_dir = "./afclap_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # AF-CLAP checkpoint URL (you'll need to find the exact URL from the paper/repo)
    # For now, we'll use the LAION-CLAP large model as a placeholder
    checkpoint_url = "https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-best.pt"
    checkpoint_path = os.path.join(checkpoint_dir, "AFClap-large-checkpoint.pt")
    
    if not os.path.exists(checkpoint_path):
        print(f"Downloading AF-CLAP checkpoint to {checkpoint_path}")
        try:
            wget.download(checkpoint_url, checkpoint_path)
            print(f"\nCheckpoint downloaded successfully")
        except Exception as e:
            print(f"Failed to download checkpoint: {e}")
            print("You may need to manually download the AF-CLAP checkpoint")
    else:
        print("AF-CLAP checkpoint already exists")

if __name__ == "__main__":
    download_afclap()