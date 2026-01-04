import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import requests

class PretrainDataset(Dataset):
    def __init__(self, data_path: str, block_size: int, split: str = 'train'):
        self.block_size = block_size
        
        # Load raw text
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"{data_path} not found. Please download it first.")
            
        with open(data_path, 'r', encoding='utf-8') as f:
            data = f.read()

        # In a real scenario, you would memory map a binary file of tokens
        # Here we just keep it simple with in-memory tokens
        # We assume dataset is already tokenized or small enough to tokenize on fly
        # For simplicity, let's assume we pass in the tokenizer externally or just load raw text
        # But to be strictly following "industry" practices we should use memory mapping.
        # Let's pivot to a simple scalable approach: load full text, encode, store as numpy array.
        
        # Placeholder: Expecting the user to run a prepare script.
        # But let's handle the raw text loading here for simplicity of the guide.
        from tokenizer import Tokenizer
        tokenizer = Tokenizer()
        tokens = tokenizer.encode(data)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        
        # 90-10 train-val split
        n = len(self.tokens)
        train_data = self.tokens[:int(n*0.9)]
        val_data = self.tokens[int(n*0.9):]
        
        self.data = train_data if split == 'train' else val_data

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # We need block_size + 1 tokens: input is x[i:i+block_size], target is x[i+1:i+block_size+1]
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

def download_tinyshakespeare(dest_path: str = "input.txt"):
    if not os.path.exists(dest_path):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        print(f"Downloading {url} to {dest_path}...")
        res = requests.get(url)
        with open(dest_path, 'w') as f:
            f.write(res.text)
        print("Done.")

if __name__ == "__main__":
    # Test the data pipeline
    download_tinyshakespeare()
    ds = PretrainDataset("input.txt", block_size=8)
    print(f"Dataset length: {len(ds)}")
    x, y = ds[0]
    print(f"Input shape: {x.shape}, Target shape: {y.shape}")
    print(f"Input: {x}")
    print(f"Target: {y}")
