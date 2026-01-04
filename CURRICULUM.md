# LLM From Scratch: Industry Level Guide

This project serves as a practical guide to building a Large Language Model (LLM) from scratch, aimed at interview preparation. We will implement a Llama-style Transformer in PyTorch.

## Curriculum

- [/] **Phase 1: Foundations & Tokenization**
    - [x] Setup project structure
    - [x] Implement/Use a BPE Tokenizer (tiktoken or minimal implementation)
    - [ ] Explain Tokenization nuances for interviews (BPE vs WordPiece, special tokens)

- [/] **Phase 2: Data Pipeline**
    - [x] Implement a scalable data loader
    - [ ] Pre-training techniques: Next token prediction, packing, masking

- [x] **Phase 3: Model Architecture (The Core)**
    - [x] Implement RMSNorm (Root Mean Square Layer Normalization)
    - [x] Implement Rotary Positional Embeddings (RoPE)
    - [x] Implement Multi-Head Attention (MHA) & Grouped Query Attention (GQA)
    - [x] Implement SwiGLU FeedForward Network
    - [x] Assemble the full Transformer Block & Llama Model

- [/] **Phase 4: Training Loop**
    - [x] Implement standard training loop
    - [x] Loss calculation (Cross Entropy)
    - [x] Optimization (AdamW, Learning Rate Schedulers)
    - [x] Gradient Clipping & Mixed Precision (AMP) usage

- [x] **Phase 5: Inference & Optimization**
    - [x] Implement KV Cache for efficient generation
    - [x] Sampling methods (Temperature, Top-k, Top-p)

- [x] **Phase 6: Advanced Interview Topics**
    - [x] Discuss Distributed Training basics (DDP, FSDP)
    - [x] Fine-tuning techniques (LoRA adapters - conceptual/basic impl)
    - [x] Alignment (RLHF/DPO - conceptual)
