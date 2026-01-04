# Advanced LLM Interview Guide

This document covers "industry-level" topics that often come up in LLM engineer interviews.

## 1. Parameter Efficient Fine-Tuning (PEFT): LoRA

**Concept**: Instead of fine-tuning all weights $W$, we learn a low-rank update $\Delta W = BA$, where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times d}$ are low-rank matrices ($r \ll d$).
The forward pass becomes: $h = Wx + BAx$.

**Implementation Pattern**:
```python
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False) # Frozen underlying layer
        
        # LoRA parameters
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank
        
        # Initialization
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        # Freeze original weights
        self.linear.weight.requires_grad = False

    def forward(self, x):
        # Wx + (BA)x * scaling
        base_out = self.linear(x)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        return base_out + lora_out * self.scaling
```
**Why it matters**: Drastically reduces VRAM usage during fine-tuning.

## 2. Distributed Training: DDP vs FSDP

For large models (>1B params), a single GPU is often insufficient.

### DDP (Distributed Data Parallel)
- **Concept**: Replicate the entire model on every GPU. Chunk data across GPUs. Sync gradients.
- **Limitation**: Model must fit in one GPU memory.
- **When to use**: Models < 1B params (approx) on Consumer HW, or when VRAM is abundant.

### FSDP (Fully Sharded Data Parallel)
- **Concept**: Shard **Parameters**, **Gradients**, and **Optimizer States** across all GPUs. Each GPU only holds a slice of the model.
- **Mechanism**: "AllGather" parameters on-demand for the forward/backward pass, then discard them to free memory.
- **Why it matters**: Allows training 70B+ models on clusters of GPUs where DDP would OOM.

## 3. Alignment: RLHF vs DPO

After pre-training (what we built) and SFT (Supervised Fine-Tuning), models are "aligned" to user preferences.

### RLHF (Reinforcement Learning from Human Feedback)
1. **SFT**: Fine-tune on high-quality Q&A.
2. **Reward Model (RM)**: Train a binary classifier to predict which of two responses is better (from human ranking data).
3. **PPO (Proximal Policy Optimization)**: Generate tokens $\rightarrow$ get Reward from RM $\rightarrow$ Update policy to maximize reward while not drifting too far from SFT (KL penalty).
- **Pros**: Established, PPO is powerful.
- **Cons**: Complex, unstable unstable, requires training a separate RM.

### DPO (Direct Preference Optimization)
- **Concept**: We can optimize the policy *directly* on preference data without an explicit Reward Model. 
- **Loss**: Derived analytically from the same objective as RLHF.
- **Pros**: Simpler (just cross-entropy style loss), more stable, memory efficient.
- **Interview Tip**: DPO is currently the "industry standard" default over RLHF for many teams due to simplicity.
