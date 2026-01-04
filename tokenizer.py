import tiktoken
from typing import List

class Tokenizer:
    def __init__(self, model_name: str = "gpt2"):
        """
        Wrapper around tiktoken for demonstration.
        In a real Llama scenario, you would load a specific .model file (SentencePiece) 
        or use a custom BPE trained on your data.
        
        For this guide, we use GPT-2's BPE for simplicity as it covers English well.
        """
        self.enc = tiktoken.get_encoding(model_name)
        
        # Special tokens handling usually happens here
        # Llama 3 has 128k vocab size, GPT-2 has ~50k
        self.vocab_size = self.enc.n_vocab
        self.eot_token = self.enc.eot_token # End of text

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> List[int]:
        tokens = self.enc.encode(text)
        if bos:
            # GPT-2 tokenizer doesn't have a specific BOS, so we might skip or reuse EOS
            # ideally, we insert a special token ID if defined.
            pass 
        if eos:
            tokens.append(self.eot_token)
        return tokens

    def decode(self, tokens: List[int]) -> str:
        return self.enc.decode(tokens)
