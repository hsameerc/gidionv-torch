import argparse
import json
from typing import List

import torch

from src.config.config import get_config
from src.data.saver_loader import load_checkpoint
from src.lib.core.hf_tokenizer_wrapper import HFTokenizerWrapper
from src.utils.prepare import format_prompt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode_memory_streams(tokenizer: HFTokenizerWrapper, context_streams: List[str]) -> List[List[List[int]]]:
    print("\n Encoding Memory Streams")
    return [[tokenizer.encode(doc)] for doc in context_streams]


def start_chat(model_config):
    """Starts the main interactive chat loop."""
    loaded_model, _, loaded_tokenizer, _ = load_checkpoint(model_config, device)
    loaded_model.eval()  # Set to evaluation mode

    context_data = "You are my assistant. You must answer my questions with the knowledge you have."
    zero_stream = [[]]
    streams = [zero_stream, zero_stream, zero_stream]
    while True:
        try:
            user_input = input("\nYou: ").strip()
            response = generate_with_memory(loaded_model, loaded_tokenizer, memory_streams=streams, query=user_input,
                                            context=context_data)
            print("Gidion Response:", response)
        except (KeyboardInterrupt, EOFError):
            break
    print("\nGideon has gone to sleep.")


@torch.no_grad()
def generate_with_memory(model, tokenizer, memory_streams: List[List[List[int]]], query: str, context: str):
    special_tokens = {"USER": "<USER>", "ASSISTANT": "<ASSISTANT>", "INST": "<INST>", "END_INST": "</INST>"}
    final_prompt_string = format_prompt(query, context, special_tokens)

    prompt_ids = torch.tensor([tokenizer.encode(query)], dtype=torch.long, device=device)

    generated, logits = model.generate(prompt_ids, memory_streams, max_new_tokens=100, temperature=0.1, top_p=0.95,
                                       return_logits=True)

    # Convert back to list for decoding
    decoded = tokenizer.decode(generated[0].tolist())
    return decoded


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Gidion Augmented Transformer.")
    parser.add_argument('--config', default='configs/gidionv_multi_memory_regularization.json', type=str,
                        help="Path to a JSON config file to override defaults.")
    args = parser.parse_args()

    cfg = get_config()
    if args.config:
        with open(args.config, 'r') as f:
            cfg.update(json.load(f))
    start_chat(cfg)
