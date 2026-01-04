from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # Later set in tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    
    # Needs to be set for RoPE. 
    # Max sequence length the model encounters.
    max_seq_len: int = 2048 

    # Dropout is usually 0 for Llama pretraining, but useful for small-scale experiments
    dropout: float = 0.0
