import argparse
import json
from typing import List

import torch

from src.config.config import get_config
from src.data.saver_loader import load_checkpoint
from src.utils.prepare import format_prompt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode_memory_streams(tokenizer, context_streams: List[str]) -> List[List[List[int]]]:
    print("\n Encoding Memory Streams")
    return [[tokenizer.encode(doc)] for doc in context_streams]


@torch.no_grad()  # Disable gradients for inference
def generate_with_memory(model, tokenizer, memory_streams: List[List[List[int]]], query: str, context: str):
    special_tokens = {"USER": "<USER>", "ASSISTANT": "<ASSISTANT>", "INST": "<INST>", "END_INST": "</INST>"}
    final_prompt_string = format_prompt(query, context, special_tokens)

    prompt_ids = torch.tensor([tokenizer.encode(final_prompt_string)], dtype=torch.long, device=device)

    generated, logits = model.generate(prompt_ids, memory_streams, max_new_tokens=100, temperature=0.1, top_p=1.0,
                                       return_logits=True)

    # [TORCH] Convert back to list for decoding
    decoded = tokenizer.decode(generated[0].tolist())
    return decoded, logits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Gidion Augmented Transformer.")
    parser.add_argument('--config', default='configs/gidionv_multi_memory.json', type=str,
                        help="Path to a JSON config file to override defaults.")
    args = parser.parse_args()

    cfg = get_config()
    if args.config:
        with open(args.config, 'r') as f:
            cfg.update(json.load(f))

    model, _, tokenizer, _ = load_checkpoint(cfg, device)
    model.eval()  # Set to evaluation mode

    query_data = "What is Verb?"
    context_data = "You must answer."
    zero_stream = [[]]
    streams = [zero_stream, zero_stream, zero_stream]
    decoded, _ = generate_with_memory(model, tokenizer, memory_streams=streams, query=query_data, context=context_data)
    print(decoded)
