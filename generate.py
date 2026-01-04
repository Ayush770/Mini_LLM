import torch
import torch.nn.functional as F
from model import Llama
from config import ModelArgs
from tokenizer import Tokenizer

def generate(
    model: Llama,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_p: float = 0.9,
    device: str = "cpu"
):
    model.eval()
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0) # (1, T)
    
    # Pre-fill (process prefix)
    # We can process the whole prefix in parallel
    # Note: Our simple KV Cache implementation in Attention checks start_pos==0 to reset.
    # So we should run the prefix first.
    
    with torch.no_grad():
        # Run prefix
        # start_pos=0 triggers a cache reset in our simple logic
        logits = model(tokens, start_pos=0) 
        
        # The last token generation
        next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(1)
        
        # Generation loop
        generated = tokens.tolist()[0]
        
        curr_pos = tokens.shape[1]
        
        print(f"Prompt: {prompt}", end="", flush=True)
        
        for _ in range(max_new_tokens):
            # In inference, we feed 1 token at a time
            # For the first step after prefix, we feed the token predicted by the prefix
            # BUT we need to be careful. The prefix run returned the prediction for the position AFTER prefix.
            # So we append it.
            
            # Note: The 'logits' from prefix run corresponds to predictions at each position.
            # We only care about the last one which predicts input[T].
            
            # Sampling
            next_token_logits = logits[:, -1, :]
            
            # Temperature
            if temperature > 0:
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

            decoded_word = tokenizer.decode([next_token.item()])
            print(decoded_word, end="", flush=True)
            generated.append(next_token.item())
            
            # Step forward
            # Input is just the new token (1, 1)
            # start_pos is current sequence length (before this new token)
            logits = model(next_token, start_pos=curr_pos) 
            curr_pos += 1

    print("\n\nDone.")
    return tokenizer.decode(generated)

if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available(): device = "cuda"
    
    # Load Config
    # MUST match training config
    args = ModelArgs(
        dim=288,
        n_layers=6,
        n_heads=6,
        vocab_size=50257,
        max_seq_len=256
    )
    
    tokenizer = Tokenizer()
    args.vocab_size = tokenizer.vocab_size

    model = Llama(args)
    
    try:
        model.load_state_dict(torch.load("model.pth", map_location=device))
        print("Loaded model.pth")
    except FileNotFoundError:
        print("model.pth not found. Using random weights.")

    model.to(device)
    
    generate(model, tokenizer, "ROMEO:", max_new_tokens=100, device=device)
