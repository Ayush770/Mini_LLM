import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
import time
import math

from model import Llama
from config import ModelArgs
from data import PretrainDataset
from tokenizer import Tokenizer

def train():
    # --- Configuration ---
    # Optimized for a small scale run on a laptop (likely CPU/MPS)
    # Using smaller parameters to avoid OOM on 8GB/16GB Macs
    args = ModelArgs(
        dim=288,
        n_layers=6,
        n_heads=6,
        vocab_size=50257, # GPT-2 vocab size
        max_seq_len=256,
        dropout=0.1
    )
    batch_size = 4 # Reduced from 32 to fix OOM
    lr = 3e-4 # Learning rate
    num_epochs = 1
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available(): device = "cuda"
    
    print(f"Using device: {device}")

    # --- Model & Data ---
    tokenizer = Tokenizer()
    # Ensure args vocab size matches tokenizer if not manually set
    args.vocab_size = tokenizer.vocab_size 

    model = Llama(args).to(device)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # Assuming input.txt is present
    try:
        dataset = PretrainDataset("input.txt", block_size=args.max_seq_len)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    except FileNotFoundError:
        print("input.txt not found. Running data download...")
        from data import download_tinyshakespeare
        download_tinyshakespeare()
        dataset = PretrainDataset("input.txt", block_size=args.max_seq_len)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    optimizer = AdamW(model.parameters(), lr=lr)

    # --- Training Loop ---
    model.train()
    start_time = time.time()
    
    # We will just train for a fixed number of steps for demonstration
    max_steps = 100 
    step = 0
    
    print("Starting training...")
    for epoch in range(num_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            logits = model(x)
            
            # Reshape for loss: (B*T, V) vs (B*T)
            # targets y are (B, T)
            # logits are (B, T, V)
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B*T, V), y.view(B*T))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient Clipping (Standard industry practice)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            if step % 10 == 0:
                print(f"Step {step} | Loss: {loss.item():.4f} | Time: {time.time()-start_time:.2f}s")
            
            step += 1
            if step >= max_steps:
                break
        if step >= max_steps:
            break

    print("Training finished.")
    torch.save(model.state_dict(), "model.pth")
    print("Model saved to model.pth")

if __name__ == "__main__":
    train()
