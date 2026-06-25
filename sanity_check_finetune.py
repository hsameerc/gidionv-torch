import datasets
import json
import torch
from src.data.saver_loader import load_checkpoint
from src.loaders.finetune_loader import format_without_context_prompt, format_prompt

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def main():
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = 'configs/finetune_v2.json'
        
    print(f"Loading config from {config_path}")
    with open(config_path) as f:
        config = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model, _, tokenizer, train_state = load_checkpoint(config, device)
    model.eval()
    print(f"Loaded checkpoint at step {train_state['total_steps']}, best_val_loss={train_state['best_val_loss']:.4f}\n")

    special_tokens = {"USER": "<USER>", "ASSISTANT": "<ASSISTANT>", "INST": "<INST>", "END_INST": "</INST>"}

    tests = [
        # (instruction, context)
        ("What is the capital of France?", ""),
        ("What year did World War II end?", ""),
        ("Explain what machine learning is in one sentence.", ""),
        ("What is 15 multiplied by 7?", ""),
        ("Summarize this passage.", "The Amazon rainforest covers over 5.5 million square kilometres and is home to an estimated 10% of all species on Earth."),
    ]

    for instruction, context in tests:
        if context:
            prompt_text = format_prompt(instruction, context, special_tokens)
        else:
            prompt_text = format_without_context_prompt(instruction, special_tokens)

        # We must NOT add the EOS token at the end of the prompt, otherwise the model thinks the conversation is over and generates random text.
        prompt_ids_raw = tokenizer.encode(prompt_text, add_special_tokens=False)
        if tokenizer.bos_token_id is not None:
            prompt_ids_raw = [tokenizer.bos_token_id] + prompt_ids_raw
            
        prompt_ids = torch.tensor([prompt_ids_raw], dtype=torch.long, device=device)
        empty_mem = [[] for _ in range(config['model']['num_memory_streams'])]

        with torch.no_grad():
            generated_ids = model.generate(
                prompt_ids=prompt_ids,
                memory_streams_ids=empty_mem,
                max_new_tokens=100,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.15,
                eos_token_id=tokenizer.eos_token_id
            )

        new_ids = generated_ids[0, prompt_ids.shape[1]:]
        response = tokenizer.decode(new_ids.tolist(), skip_special_tokens=True).strip()

        print(f"{'='*60}")
        print(f"Q: {instruction}")
        if context:
            print(f"CTX: {context[:80]}")
        print(f"A: {response}")
        print()

if __name__ == '__main__':
    main()
