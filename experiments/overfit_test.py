import torch
import torch.optim as optim
from src.config.config import get_config
from src.lib.core.hf_tokenizer_wrapper import HFTokenizerWrapper
from src.lib.transformer.multi_memory_transformer import MultiMemoryTransformer
import json

def run_overfit_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load config and tokenizer
    with open('configs/finetune_v2.json', 'r') as f:
        config = json.load(f)
    tokenizer = HFTokenizerWrapper(config['TOKENIZER_PATH'])
    
    # Initialize model
    model = MultiMemoryTransformer(config, tokenizer).to(device)
    
    # Load the latest checkpoint
    checkpoint = torch.load('research/models/gidionv_finetune_v2/gidionv_finetune_v2_latest.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Loaded latest checkpoint!")

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # Create exactly ONE training example
    prompt_text = "<USER><INST> What is the secret password? </INST><ASSISTANT> "
    target_text = "The secret password is 'AntigravityIsAwesome'."

    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    target_ids_raw = tokenizer.encode(target_text, add_special_tokens=False) + [tokenizer.eos_token_id]
    
    # Format tensors
    input_seq = [tokenizer.bos_token_id] + prompt_ids + target_ids_raw
    input_ids = torch.tensor(input_seq[:-1], dtype=torch.long, device=device).unsqueeze(0)
    target_ids = torch.tensor(input_seq[1:], dtype=torch.long, device=device).unsqueeze(0)

    # Mask the prompt in target_ids
    prompt_len = len(prompt_ids)
    target_ids[0, :prompt_len] = -100

    empty_mem = torch.full((1, config['model']['num_memory_streams'], 1), tokenizer.pad_token_id, dtype=torch.long, device=device)

    print("\n--- BEFORE OVERFITTING ---")
    model.eval()
    with torch.no_grad():
        test_prompt = torch.tensor([[tokenizer.bos_token_id] + prompt_ids], dtype=torch.long, device=device)
        mem = [[] for _ in range(config['model']['num_memory_streams'])]
        out = model.generate(test_prompt, mem, max_new_tokens=20, temperature=0.0, eos_token_id=tokenizer.eos_token_id)
        print("Model Output:", tokenizer.decode(out[0, test_prompt.shape[1]:].tolist()).strip())

    print("\n--- OVERFITTING ON 1 EXAMPLE (50 Steps) ---")
    model.train()
    for step in range(50):
        optimizer.zero_grad()
        logits, _, _ = model(input_ids, empty_mem)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, tokenizer.vocab_size), target_ids.view(-1))
        loss.backward()
        optimizer.step()
        if step % 10 == 0:
            print(f"Step {step:2d} | Loss: {loss.item():.4f}")

    print("\n--- AFTER OVERFITTING ---")
    model.eval()
    with torch.no_grad():
        test_prompt = torch.tensor([[tokenizer.bos_token_id] + prompt_ids], dtype=torch.long, device=device)
        out = model.generate(test_prompt, mem, max_new_tokens=20, temperature=0.0, eos_token_id=tokenizer.eos_token_id)
        print("Model Output:", tokenizer.decode(out[0, test_prompt.shape[1]:].tolist()).strip())

if __name__ == '__main__':
    run_overfit_test()
