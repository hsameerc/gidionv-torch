"""
TRUE Memory Stream Swap Test (Definitive Proof)
================================================
The previous ablation test was inconclusive because the model could memorize
the answer token IDs directly from the training label supervision signal,
never needing to attend to the memory stream at all.

This test uses a SWAP to make the test unambiguous:

  1. We train the model using Code A in the memory stream.
     Training Target: "The secret code is 'CODEA'."
     
  2. At inference time, we SWAP the memory stream to contain Code B.
     The model has NEVER seen Code B in any training target.

  3. We ask the question: "What is the secret code?"

  VERDICT:
    - If the model outputs Code A -> It memorized the training label, ignoring memory. FAIL.
    - If the model outputs Code B -> It is reading LIVE from the memory stream. PASS!

This is the gold standard test for memory stream utility.
"""

import torch
import torch.optim as optim
from src.lib.core.hf_tokenizer_wrapper import HFTokenizerWrapper
from src.lib.transformer.multi_memory_transformer import MultiMemoryTransformer
import json
import copy

CODE_A = "XQZK9"     # The code used DURING training
CODE_B = "MRVP7"     # A completely different code, ONLY injected at inference time


def load_model(config, tokenizer, device, num_streams):
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


def run_swap_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    with open('configs/finetune_v2.json', 'r') as f:
        config = json.load(f)

    tokenizer = HFTokenizerWrapper(config['TOKENIZER_PATH'])
    num_streams = 1

    question_text  = "<USER><INST> What is the secret code? </INST><ASSISTANT> "
    # Training uses Code A in both memory and target
    memory_text_A  = f"MEMORY: The secret code is '{CODE_A}'."
    target_text_A  = f"The secret code is '{CODE_A}'."
    # Inference uses Code B ONLY in memory (model has never seen it)
    memory_text_B  = f"MEMORY: The secret code is '{CODE_B}'."

    prompt_ids     = tokenizer.encode(question_text, add_special_tokens=False)
    target_ids_raw = tokenizer.encode(target_text_A, add_special_tokens=False) + [tokenizer.eos_token_id]
    mem_ids_A      = tokenizer.encode(memory_text_A, add_special_tokens=False)
    mem_ids_B      = tokenizer.encode(memory_text_B, add_special_tokens=False)

    input_seq  = [tokenizer.bos_token_id] + prompt_ids + target_ids_raw
    input_ids  = torch.tensor(input_seq[:-1], dtype=torch.long, device=device).unsqueeze(0)
    label_ids  = torch.tensor(input_seq[1:],  dtype=torch.long, device=device).unsqueeze(0)
    label_ids[0, :len(prompt_ids)] = -100

    # Memory tensors
    mem_tensor_A = torch.full((1, num_streams, len(mem_ids_A)), tokenizer.pad_token_id, dtype=torch.long, device=device)
    mem_tensor_A[0, 0, :] = torch.tensor(mem_ids_A, dtype=torch.long, device=device)

    empty_list = [[] for _ in range(num_streams)]
    mem_list_A = [mem_ids_A]
    mem_list_B = [mem_ids_B]

    NUM_STEPS = 100

    print("=" * 65)
    print(f"  TRAINING: Using Code A = '{CODE_A}' in memory stream")
    print("=" * 65)
    code_a_token_ids = tokenizer.encode(CODE_A, add_special_tokens=False)
    code_b_token_ids = tokenizer.encode(CODE_B, add_special_tokens=False)
    print(f"  Code A token IDs: {code_a_token_ids}")
    print(f"  Code B token IDs: {code_b_token_ids}")
    print(f"  (If token IDs are different, the model cannot confuse the two)\n")

    model = load_model(config, tokenizer, device, num_streams)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4)

    model.train()
    for step in range(NUM_STEPS):
        optimizer.zero_grad()
        mem_unbind = list(torch.unbind(mem_tensor_A, dim=1))
        logits, _, _ = model(input_ids, memory_streams_ids=mem_unbind)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, tokenizer.vocab_size), label_ids.view(-1))
        loss.backward()
        optimizer.step()
        if step % 25 == 0:
            print(f"  Step {step:3d} | Loss: {loss.item():.4f}")

    print("\n" + "=" * 65)
    print("  INFERENCE: The SWAP test")
    print("=" * 65)

    model.eval()
    with torch.no_grad():
        test_prompt = torch.tensor([[tokenizer.bos_token_id] + prompt_ids], dtype=torch.long, device=device)

        # Test 1: With Code A memory (same as training) - should say Code A
        out = model.generate(test_prompt, mem_list_A, max_new_tokens=25, temperature=0.0, eos_token_id=tokenizer.eos_token_id)
        decoded_code_a_mem = tokenizer.decode(out[0, test_prompt.shape[1]:].tolist()).strip()
        print(f"\n  [Baseline]  Memory=Code A: {decoded_code_a_mem}")

        # Test 2: With NO memory - what does it default to?
        out = model.generate(test_prompt, empty_list, max_new_tokens=25, temperature=0.0, eos_token_id=tokenizer.eos_token_id)
        decoded_no_mem = tokenizer.decode(out[0, test_prompt.shape[1]:].tolist()).strip()
        print(f"  [No Memory] Memory=Empty:  {decoded_no_mem}")

        # THE KEY TEST: Swap to Code B at inference time!
        out = model.generate(test_prompt, mem_list_B, max_new_tokens=25, temperature=0.0, eos_token_id=tokenizer.eos_token_id)
        decoded_swapped = tokenizer.decode(out[0, test_prompt.shape[1]:].tolist()).strip()
        print(f"\n  [SWAP TEST] Memory=Code B: {decoded_swapped}")

    print("\n" + "=" * 65)
    print("  VERDICT")
    print("=" * 65)
    code_a_in_swap = CODE_A.lower() in decoded_swapped.lower()
    code_b_in_swap = CODE_B.lower() in decoded_swapped.lower()

    if code_b_in_swap:
        print(f"\n  ✓ PASS: Memory Stream IS being read at inference time!")
        print(f"    The model output Code B ('{CODE_B}') which was ONLY in the memory")
        print(f"    stream. It was never in any training label. The model is genuinely")
        print(f"    attending to the live memory stream content.")
    elif code_a_in_swap:
        print(f"\n  ✗ FAIL (Memorization): The model output Code A ('{CODE_A}')")
        print(f"    This means it ignored the memory stream and recalled the code it")
        print(f"    memorized from the training label. The memory stream is not being read.")
        print(f"    This could mean the model needs more training epochs to learn to")
        print(f"    prefer reading from memory rather than recalling from weights.")
    else:
        print(f"\n  INCONCLUSIVE: Model output neither code: '{decoded_swapped}'")
        print(f"    The model could not recall either code. Training may need more steps.")


if __name__ == '__main__':
    run_swap_test()
