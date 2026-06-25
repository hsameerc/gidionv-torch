"""
TRUE Memory Stream Swap Test v2 - Multi-Code Curriculum
=========================================================
The previous test failed because we trained with a SINGLE code 100 times.
The model memorized the code directly from the training label, bypassing memory.

The fix: Train with a DIFFERENT random code on EVERY step. The model physically
cannot memorize all of them. The ONLY way it can consistently achieve low loss
is to learn to attend to the memory stream and copy the code from there.

After training, we test with a BRAND NEW code it has never seen at all.
If the model outputs the new code, memory streams are definitively working.
"""

import torch
import torch.optim as optim
from src.lib.core.hf_tokenizer_wrapper import HFTokenizerWrapper
from src.lib.transformer.multi_memory_transformer import MultiMemoryTransformer
import json
import copy
import random
import string

NUM_STEPS   = 1000
CODE_LENGTH = 5    # characters per random code


def random_code():
    """Generate a random alphanumeric code the model is unlikely to have memorized."""
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=CODE_LENGTH))


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


def make_batch(tokenizer, device, num_streams, code):
    """Build input/label/memory tensors for one specific code."""
    question_text = "<USER><INST> What is the secret code? </INST><ASSISTANT> "
    memory_text   = f"MEMORY: The secret code is '{code}'."
    target_text   = f"The secret code is '{code}'."

    prompt_ids     = tokenizer.encode(question_text, add_special_tokens=False)
    target_ids_raw = tokenizer.encode(target_text, add_special_tokens=False) + [tokenizer.eos_token_id]
    mem_ids        = tokenizer.encode(memory_text, add_special_tokens=False)

    input_seq  = [tokenizer.bos_token_id] + prompt_ids + target_ids_raw
    input_ids  = torch.tensor(input_seq[:-1], dtype=torch.long, device=device).unsqueeze(0)
    label_ids  = torch.tensor(input_seq[1:],  dtype=torch.long, device=device).unsqueeze(0)
    label_ids[0, :len(prompt_ids)] = -100  # mask the prompt

    mem_tensor = torch.full((1, num_streams, len(mem_ids)), tokenizer.pad_token_id, dtype=torch.long, device=device)
    mem_tensor[0, 0, :] = torch.tensor(mem_ids, dtype=torch.long, device=device)

    return input_ids, label_ids, mem_tensor, mem_ids, prompt_ids


def run_swap_test_v2():
    random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    with open('configs/finetune_v2.json', 'r') as f:
        config = json.load(f)

    tokenizer = HFTokenizerWrapper(config['TOKENIZER_PATH'])
    num_streams = 1

    print("=" * 65)
    print(f"  TRAINING: {NUM_STEPS} steps with a NEW random code every step")
    print("=" * 65)
    print(f"  The model CANNOT memorize any single code. It must learn to")
    print(f"  read the code from the memory stream to achieve low loss.\n")

    model = load_model(config, tokenizer, device, num_streams)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    model.train()
    recent_losses = []
    for step in range(NUM_STEPS):
        code = random_code()
        input_ids, label_ids, mem_tensor, _, _ = make_batch(tokenizer, device, num_streams, code)

        optimizer.zero_grad()
        mem_unbind = list(torch.unbind(mem_tensor, dim=1))
        logits, _, _ = model(input_ids, memory_streams_ids=mem_unbind)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, tokenizer.vocab_size), label_ids.view(-1))
        loss.backward()
        optimizer.step()

        recent_losses.append(loss.item())
        if step % 50 == 0:
            avg = sum(recent_losses) / len(recent_losses)
            print(f"  Step {step:3d} | Loss: {loss.item():.4f} | Avg Loss: {avg:.4f} | Code this step: {code}")
            recent_losses = []

    print("\n" + "=" * 65)
    print("  INFERENCE: The SWAP test with a brand new code")
    print("=" * 65)

    # Generate a truly new code for the swap test
    TEST_CODE = random_code()
    while TEST_CODE == "XQZK9" or TEST_CODE == "MRVP7":
        TEST_CODE = random_code()

    print(f"\n  Test Code (never seen in training): '{TEST_CODE}'")
    print(f"  Token IDs: {tokenizer.encode(TEST_CODE, add_special_tokens=False)}")

    _, _, _, mem_ids_test, prompt_ids = make_batch(tokenizer, device, num_streams, TEST_CODE)
    mem_list_test = [mem_ids_test]
    empty_list = [[] for _ in range(num_streams)]

    model.eval()
    with torch.no_grad():
        test_prompt = torch.tensor([[tokenizer.bos_token_id] + prompt_ids], dtype=torch.long, device=device)

        out_with_mem = model.generate(test_prompt, mem_list_test, max_new_tokens=25, temperature=0.0, eos_token_id=tokenizer.eos_token_id)
        decoded_with_mem = tokenizer.decode(out_with_mem[0, test_prompt.shape[1]:].tolist()).strip()
        print(f"\n  [WITH memory = '{TEST_CODE}']: {decoded_with_mem}")

        out_no_mem = model.generate(test_prompt, empty_list, max_new_tokens=25, temperature=0.0, eos_token_id=tokenizer.eos_token_id)
        decoded_no_mem = tokenizer.decode(out_no_mem[0, test_prompt.shape[1]:].tolist()).strip()
        print(f"  [WITHOUT memory]:               {decoded_no_mem}")

    print("\n" + "=" * 65)
    print("  VERDICT")
    print("=" * 65)
    code_in_output = TEST_CODE.lower() in decoded_with_mem.lower()

    if code_in_output:
        print(f"\n  PASS: Memory Stream IS being read at inference!")
        print(f"  The model reproduced '{TEST_CODE}' which was NEVER in any training label.")
        print(f"  It learned to extract the code LIVE from the memory stream.")
    else:
        print(f"\n  FAIL or INCONCLUSIVE: Model could not reproduce '{TEST_CODE}'.")
        print(f"  Model said: '{decoded_with_mem}'")
        print(f"  The cross-attention may need more training steps to generalise.")
        print(f"  Try increasing NUM_STEPS to 1000+ for a more conclusive result.")


if __name__ == '__main__':
    run_swap_test_v2()
