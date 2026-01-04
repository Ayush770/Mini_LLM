# LLM From Scratch - Walkthrough

This project implements a Llama-style Transformer from scratch in PyTorch.

## Project Structure
- `config.py`: Configuration.
- `model.py`: Architecture (RMSNorm, RoPE, Attention, FeedForward, TransformerBlock, Llama).
- `train.py`: Training loop.
- `generate.py`: Inference script.
- `requirements.txt`: Dependencies.

## How to Run (Recommended)

### 0. Go to the Project Directory
**Crucial Step**: You must be inside the project folder.
```bash
cd /Users/ayushtrivedi/.gemini/antigravity/scratch/llm-from-scratch
```

### 1. Setup Environment (User Install)
We will install dependencies to your user library to avoid permission issues.
```bash
/usr/bin/python3 -m pip install --user -r requirements.txt
```

### 2. Training
Run the training script:
```bash
/usr/bin/python3 train.py
```
*Expected Output*: Loss should decrease. A `model.pth` file will be saved.

### 3. Inference
Generate text:
```bash
/usr/bin/python3 generate.py
```

## Troubleshooting
- **ModuleNotFoundError: No module named 'torch'**: This means the installation failed or you are running a different python. **Run step 1 again.**
- **check_hostname requires server_hostname**: If you see SSL errors, you might need to update certificates or allow insecure install:
  `pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt`

## Key Architecture Concepts
- **RMSNorm**: Simpler normalization.
- **RoPE**: Rotary Positional Embeddings.
- **SwiGLU**: Modern activation function.
- **KV Cache**: Efficient inference.

Check `interview_guide.md` for advanced topics!
