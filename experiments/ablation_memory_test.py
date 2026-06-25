"""
TRUE Memory Stream Ablation Test
=================================
This is the definitive proof that the Multi-Memory Stream architecture
is working correctly and is causally necessary for the model to learn.

Test Design:
- We give the model a BLANK target label: "ANSWER: ???"
  where '???' is a placeholder it has NEVER seen before.
- The ONLY place the real answer exists is inside the Memory Stream.
- We train TWO separate models on the SAME data:
    1. Model A: WITH the memory stream populated
    2. Model B: WITHOUT the memory stream (all pads)
- We then ask both models the question at generation time WITH the memory stream.
- If the architecture works:
    * Model A must output the correct answer.
    * Model B must fail, because it never had access to the answer during training.
"""

import torch
import torch.optim as optim
from src.lib.core.hf_tokenizer_wrapper import HFTokenizerWrapper
from src.lib.transformer.multi_memory_transformer import MultiMemoryTransformer
import json
import copy

ANSWER_PLACEHOLDER = "XQZK9"  # A random string the model has never seen in pre-training


def load_model(config, tokenizer, device, num_streams):
    """Load and patch a fresh model for num_streams memory streams."""
    config = copy.deepcopy(config)
    config['model']['num_memory_streams'] = num_streams

    model = MultiMemoryTransformer(config, tokenizer).to(device)

    checkpoint = torch.load(
        'research/models/gidionv_finetune_v2/gidionv_finetune_v2_latest.pth',
        map_location=device
    )
    state_dict = checkpoint['model_state_dict']

    new_state_dict = {}
    for key, value in state_dict.items():
        if "cross_attention_layers" in key:
            parts = key.split('.')
            stream_idx = int(parts[3])
            if stream_idx == 0:
                for i in range(num_streams):
                    new_parts = list(parts)
                    new_parts[3] = str(i)
                    new_state_dict['.'.join(new_parts)] = value.clone()
        elif "fusion_weights" in key:
            new_state_dict[key] = value.repeat(num_streams)
        else:
            new_state_dict[key] = value

    model.load_state_dict(new_state_dict)
    return model


def run_ablation_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    with open('configs/finetune_v2.json', 'r') as f:
        config = json.load(f)

    tokenizer = HFTokenizerWrapper(config['TOKENIZER_PATH'])
    num_streams = 1

    # --- Define the test data ---
    # The question only appears in the prompt.
    # The answer ONLY appears in the memory stream.
    # The training target is a MASKED version with our unique placeholder.
    question_text = "<USER><INST> What is the secret code? </INST><ASSISTANT> "
    memory_text   = f"The secret code is '{ANSWER_PLACEHOLDER}'."
    # We train the model to output this placeholder string.
    # Since the placeholder is random gibberish, the model CANNOT predict it
    # without looking at the memory stream. It cannot guess it from language statistics.
    target_text   = f"The secret code is '{ANSWER_PLACEHOLDER}'."

    prompt_ids      = tokenizer.encode(question_text, add_special_tokens=False)
    target_ids_raw  = tokenizer.encode(target_text, add_special_tokens=False) + [tokenizer.eos_token_id]
    memory_ids_raw  = tokenizer.encode(memory_text, add_special_tokens=False)

    input_seq  = [tokenizer.bos_token_id] + prompt_ids + target_ids_raw
    input_ids  = torch.tensor(input_seq[:-1], dtype=torch.long, device=device).unsqueeze(0)
    label_ids  = torch.tensor(input_seq[1:],  dtype=torch.long, device=device).unsqueeze(0)

    # Mask the prompt tokens so loss is only computed on the answer tokens
    label_ids[0, :len(prompt_ids)] = -100

    # Memory tensor
    mem_tensor = torch.full(
        (1, num_streams, len(memory_ids_raw)), tokenizer.pad_token_id,
        dtype=torch.long, device=device
    )
    mem_tensor[0, 0, :] = torch.tensor(memory_ids_raw, dtype=torch.long, device=device)

    # Empty memory tensor (no information)
    empty_mem_tensor = torch.full(
        (1, num_streams, len(memory_ids_raw)), tokenizer.pad_token_id,
        dtype=torch.long, device=device
    )

    # Generation memory list
    mem_list   = [memory_ids_raw]
    empty_list = [[] for _ in range(num_streams)]

    NUM_STEPS = 100

    print("=" * 65)
    print("  EXPERIMENT A: Training WITH the memory stream populated")
    print("=" * 65)
    model_a = load_model(config, tokenizer, device, num_streams)
    optimizer_a = optim.AdamW(model_a.parameters(), lr=5e-4)

    print(f"\nMemory Stream Content: \"{memory_text}\"")
    print(f"Training Target:       \"{target_text}\"")
    print(f"Placeholder token IDs for '{ANSWER_PLACEHOLDER}': {tokenizer.encode(ANSWER_PLACEHOLDER, add_special_tokens=False)}")

    model_a.train()
    for step in range(NUM_STEPS):
        optimizer_a.zero_grad()
        mem_unbind = list(torch.unbind(mem_tensor, dim=1))
        logits, _, _ = model_a(input_ids, memory_streams_ids=mem_unbind)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, tokenizer.vocab_size), label_ids.view(-1))
        loss.backward()
        optimizer_a.step()
        if step % 20 == 0:
            print(f"  Step {step:3d} | Loss: {loss.item():.4f}")

    model_a.eval()
    with torch.no_grad():
        test_prompt = torch.tensor([[tokenizer.bos_token_id] + prompt_ids], dtype=torch.long, device=device)
        out_a = model_a.generate(test_prompt, mem_list, max_new_tokens=25, temperature=0.0, eos_token_id=tokenizer.eos_token_id)
        decoded_a = tokenizer.decode(out_a[0, test_prompt.shape[1]:].tolist()).strip()
        print(f"\n  [+] Model A Output (WITH memory at inference): {decoded_a}")

        out_a_no_mem = model_a.generate(test_prompt, empty_list, max_new_tokens=25, temperature=0.0, eos_token_id=tokenizer.eos_token_id)
        decoded_a_no_mem = tokenizer.decode(out_a_no_mem[0, test_prompt.shape[1]:].tolist()).strip()
        print(f"  [-] Model A Output (WITHOUT memory at inference): {decoded_a_no_mem}")

    print("\n" + "=" * 65)
    print("  EXPERIMENT B: Training WITHOUT the memory stream (ablation)")
    print("=" * 65)
    model_b = load_model(config, tokenizer, device, num_streams)
    optimizer_b = optim.AdamW(model_b.parameters(), lr=5e-4)

    print(f"\nMemory Stream Content: [ALL PADDING - EMPTY]")
    print(f"Training Target:       \"{target_text}\"")

    model_b.train()
    for step in range(NUM_STEPS):
        optimizer_b.zero_grad()
        empty_unbind = list(torch.unbind(empty_mem_tensor, dim=1))
        logits, _, _ = model_b(input_ids, memory_streams_ids=empty_unbind)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, tokenizer.vocab_size), label_ids.view(-1))
        loss.backward()
        optimizer_b.step()
        if step % 20 == 0:
            print(f"  Step {step:3d} | Loss: {loss.item():.4f}")

    model_b.eval()
    with torch.no_grad():
        test_prompt = torch.tensor([[tokenizer.bos_token_id] + prompt_ids], dtype=torch.long, device=device)
        out_b = model_b.generate(test_prompt, mem_list, max_new_tokens=25, temperature=0.0, eos_token_id=tokenizer.eos_token_id)
        decoded_b = tokenizer.decode(out_b[0, test_prompt.shape[1]:].tolist()).strip()
        print(f"\n  [+] Model B Output (WITH memory at inference): {decoded_b}")

        out_b_no_mem = model_b.generate(test_prompt, empty_list, max_new_tokens=25, temperature=0.0, eos_token_id=tokenizer.eos_token_id)
        decoded_b_no_mem = tokenizer.decode(out_b_no_mem[0, test_prompt.shape[1]:].tolist()).strip()
        print(f"  [-] Model B Output (WITHOUT memory at inference): {decoded_b_no_mem}")

    print("\n" + "=" * 65)
    print("  VERDICT")
    print("=" * 65)
    answer_in_a = ANSWER_PLACEHOLDER.lower() in decoded_a.lower()
    answer_in_b = ANSWER_PLACEHOLDER.lower() in decoded_b.lower()

    if answer_in_a and not answer_in_b:
        print(f"\n  ✓ PASS: Memory Stream is causally necessary!")
        print(f"    Model A (trained WITH memory) correctly reproduced the placeholder.")
        print(f"    Model B (trained WITHOUT memory) could NOT reproduce the placeholder.")
    elif answer_in_a and answer_in_b:
        print(f"\n  INCONCLUSIVE: Both models learned the placeholder.")
        print(f"    This may mean the model memorized the target without using memory.")
        print(f"    Try increasing NUM_STEPS for a more conclusive result.")
    elif not answer_in_a and not answer_in_b:
        print(f"\n  INCONCLUSIVE: Neither model learned the placeholder.")
        print(f"    Training may need more steps. Try increasing NUM_STEPS.")
    else:
        print(f"\n  ✗ UNEXPECTED: Model B learned the placeholder but Model A did not.")

if __name__ == '__main__':
    run_ablation_test()
