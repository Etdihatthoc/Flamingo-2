"""
VSTEP-specific data loading wrapper
Use this if you encounter issues with the default data loader
"""
import json
import os
import librosa
import torch
from torch.utils.data import Dataset

class VSTEPDataset(Dataset):
    """Simple VSTEP dataset wrapper"""
    
    def __init__(self, manifest_path, data_root, clap_config, tokenizer):
        with open(manifest_path, 'r') as f:
            self.data = json.load(f)
        
        self.data_root = data_root
        self.clap_config = clap_config
        self.tokenizer = tokenizer
        
        # Convert data dict to list for easier indexing
        self.samples = []
        for key, value in self.data['data'].items():
            self.samples.append(value)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load audio
        audio_path = os.path.join(self.data_root, sample['name'])
        try:
            audio, sr = librosa.load(audio_path, sr=48000)  # AF-CLAP uses 48kHz
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Return silence if audio fails to load
            audio = torch.zeros(48000, dtype=torch.float32)
        
        # Prepare text input (prompt)
        prompt_text = sample['prompt']
        target_text = sample['output']
        
        # Combine for model input
        full_text = f"{prompt_text}\n\n{target_text}"
        
        return {
            'audio': torch.tensor(audio, dtype=torch.float32),
            'text': full_text,
            'prompt': prompt_text,
            'target': target_text,
            'audio_path': audio_path,
            'duration': sample.get('duration', 10.0)
        }