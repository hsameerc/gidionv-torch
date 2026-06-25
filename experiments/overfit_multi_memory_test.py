import torch
import torch.optim as optim
from src.config.config import get_config
from src.lib.core.hf_tokenizer_wrapper import HFTokenizerWrapper
from src.lib.transformer.multi_memory_transformer import MultiMemoryTransformer
import json

def run_multi_memory_overfit_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load config and tokenizer
    with open('configs/finetune_v2.json', 'r') as f:
        config = json.load(f)
        
    # FORCE 3 MEMORY STREAMS FOR THIS TEST
    num_streams = 3
    config['model']['num_memory_streams'] = num_streams
    
    tokenizer = HFTokenizerWrapper(config['TOKENIZER_PATH'])
    
    # Initialize model with 3 streams
    model = MultiMemoryTransformer(config, tokenizer).to(device)
    
    # Load the latest checkpoint
    checkpoint = torch.load('research/models/gidionv_finetune_v2/gidionv_finetune_v2_latest.pth', map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    # We must patch the state dict because the checkpoint was saved with only 1 memory stream!
    new_state_dict = {}
    for key, value in state_dict.items():
        if "cross_attention_layers" in key:
            # key looks like: decoder_blocks.0.cross_attention_layers.0.cross_attn.q_proj.weight
            # We copy the '0' stream to the new streams
            parts = key.split('.')
            stream_idx = int(parts[3])
            if stream_idx == 0:
                # Add for stream 0
                new_state_dict[key] = value
                # Duplicate for stream 1 and 2
                for i in range(1, num_streams):
                    new_parts = list(parts)
                    new_parts[3] = str(i)
                    new_key = '.'.join(new_parts)
                    new_state_dict[new_key] = value.clone()
        elif "fusion_weights" in key:
            # Expand fusion weights from shape [1] to [3]
            new_state_dict[key] = value.repeat(num_streams)
        else:
            new_state_dict[key] = value

    model.load_state_dict(new_state_dict)
    print(f"Loaded and dynamically patched checkpoint for {num_streams} memory streams!")

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # We ask a question in the main prompt
    prompt_text = "<USER><INST> What is the complete secret password? </INST><ASSISTANT> "
    target_text = "The complete secret password is 'Anti-Gravity-Is-Awesome'."
    
    # We provide the crucial context split across 3 memory streams!
    memory_context_text_0 = "Part 1 of the password is 'Anti'."
    memory_context_text_1 = "Part 2 of the password is 'Gravity'."
    memory_context_text_2 = "Part 3 of the password is 'Is-Awesome'."

    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    target_ids_raw = tokenizer.encode(target_text, add_special_tokens=False) + [tokenizer.eos_token_id]
    
    mem_ids_0 = tokenizer.encode(memory_context_text_0, add_special_tokens=False)
    mem_ids_1 = tokenizer.encode(memory_context_text_1, add_special_tokens=False)
    mem_ids_2 = tokenizer.encode(memory_context_text_2, add_special_tokens=False)
    
    # Format main tensors
    input_seq = [tokenizer.bos_token_id] + prompt_ids + target_ids_raw
    input_ids = torch.tensor(input_seq[:-1], dtype=torch.long, device=device).unsqueeze(0)
    target_ids = torch.tensor(input_seq[1:], dtype=torch.long, device=device).unsqueeze(0)

    # Mask the prompt in target_ids
    prompt_len = len(prompt_ids)
    target_ids[0, :prompt_len] = -100

    # Format memory stream tensors (Batch x Streams x Length)
    max_mem_len = max(len(mem_ids_0), len(mem_ids_1), len(mem_ids_2))
    mem_tensor = torch.full((1, num_streams, max_mem_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
    mem_tensor[0, 0, :len(mem_ids_0)] = torch.tensor(mem_ids_0, dtype=torch.long, device=device)
    mem_tensor[0, 1, :len(mem_ids_1)] = torch.tensor(mem_ids_1, dtype=torch.long, device=device)
    mem_tensor[0, 2, :len(mem_ids_2)] = torch.tensor(mem_ids_2, dtype=torch.long, device=device)

    # Preparation for generation
    mem_list = [mem_ids_0, mem_ids_1, mem_ids_2]

    print("\n--- BEFORE OVERFITTING ---")
    model.eval()
    with torch.no_grad():
        test_prompt = torch.tensor([[tokenizer.bos_token_id] + prompt_ids], dtype=torch.long, device=device)
        out = model.generate(test_prompt, mem_list, max_new_tokens=20, temperature=0.0, eos_token_id=tokenizer.eos_token_id)
        print("Model Output:", tokenizer.decode(out[0, test_prompt.shape[1]:].tolist()).strip())

    print("\n--- OVERFITTING ON 3 MEMORY STREAMS (50 Steps) ---")
    model.train()
    for step in range(50):
        optimizer.zero_grad()
        
        # Forward pass using the memory stream tensor!
        unbound_mem = list(torch.unbind(mem_tensor, dim=1))
        logits, _, _ = model(input_ids, memory_streams_ids=unbound_mem)
        
        loss = torch.nn.functional.cross_entropy(logits.view(-1, tokenizer.vocab_size), target_ids.view(-1))
        loss.backward()
        optimizer.step()
        if step % 10 == 0:
            print(f"Step {step:2d} | Loss: {loss.item():.4f}")

    print("\n--- AFTER OVERFITTING ---")
    model.eval()
    with torch.no_grad():
        test_prompt = torch.tensor([[tokenizer.bos_token_id] + prompt_ids], dtype=torch.long, device=device)
        out = model.generate(test_prompt, mem_list, max_new_tokens=20, temperature=0.0, eos_token_id=tokenizer.eos_token_id)
        print("Model Output:", tokenizer.decode(out[0, test_prompt.shape[1]:].tolist()).strip())

if __name__ == '__main__':
    run_multi_memory_overfit_test()
