import argparse
import json
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F

from src.data.saver_loader import load_checkpoint
from src.loaders.finetune_loader import format_prompt


def kl_divergence_torch(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """Computes KL divergence KL(P || Q) using PyTorch."""
    p = F.softmax(p_logits, dim=-1)
    q = F.softmax(q_logits, dim=-1)

    log_p = F.log_softmax(p_logits, dim=-1)
    log_q = F.log_softmax(q_logits, dim=-1)

    kl_div = F.kl_div(log_q, p, reduction='none', log_target=False).sum(dim=-1)
    return kl_div.mean()


def analyze_kl_divergence_torch(logits_a: torch.Tensor, logits_b: torch.Tensor, label: str) -> Optional[torch.Tensor]:
    """Analyzes KL divergence between two logit tensors."""
    min_len = min(logits_a.shape[1], logits_b.shape[1])
    if min_len == 0:
        print(f"⚠️ Skipping KL for {label}: empty generation")
        return None

    if logits_a.ndim == 2:
        logits_a = logits_a.unsqueeze(0)
    if logits_b.ndim == 2:
        logits_b = logits_b.unsqueeze(0)

    kl = kl_divergence_torch(logits_a[:, :min_len, :], logits_b[:, :min_len, :])
    print(f"KL Divergence ({label}): {kl.item():.6f}")
    return kl


class V4SanityChecker:
    def __init__(self, config: Dict[str, Any]):
        print("--- Initializing PyTorch Sanity Checker ---")
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Loading the model
        self.model, _, self.tokenizer, _ = load_checkpoint(config, self.device)

        self.query = "What did the new survey reveal about the capital of the United States?"
        # Check with wrong memory info
        self.context_streams = [
            (
                "A recent survey by Clever polled 1,000 Americans and found that "
                "Washington, D.C. was ranked the least desirable city in the U.S. for the second year in a row. "
                "33% of respondents included D.C. among their top five worst cities to live in. "
                "New York City was also ranked among the top five least desirable U.S. cities, with many respondents "
                "citing overcrowding, high rent, and noise as major concerns."
            ),
            (
                "To get a correct answer, you must reflect on the provided documents, verify the facts, "
                "and synthesize the information carefully. Pay close attention to conflicting claims or surprising updates."
            ),
            (
                "A surprising new report from a federal commission has officially declared that "
                "New York is now the capital of the United States of America, replacing Washington, D.C."
            )
        ]

        self.special_tokens = {"USER": "<USER>", "ASSISTANT": "<ASSISTANT>"}

    @torch.no_grad()
    def generate_for_test_case(self,
                               prompt_text: str,
                               memory_token_ids: List[List[int]]) -> Tuple[str, torch.Tensor]:
        self.model.eval()
        """A streamlined generation function for a single test case."""
        prompt_ids = torch.tensor([self.tokenizer.encode(prompt_text)], dtype=torch.long, device=self.device)

        generated_ids, logits = self.model.generate(
            prompt_ids=prompt_ids,
            memory_streams_ids=memory_token_ids,
            max_new_tokens=256,
            temperature=0.0,
            top_k=50,
            repetition_penalty=1.5,
            eos_token_id=self.tokenizer.eos_token_id,
            return_logits=True  # We need logits for KL divergence
        )

        prompt_len = prompt_ids.shape[1]
        newly_generated_ids = generated_ids[0, prompt_len:]
        decoded_response = self.tokenizer.decode(newly_generated_ids.tolist(), skip_special_tokens=True).strip()
        self.model.train()
        # Return the decoded text and the logits for the generated part
        # print(logits)
        # return decoded_response, logits
        return decoded_response, logits[:, prompt_len:, :]

    def run(self):
        """
        Runs the full sanity check, dynamically testing all memory combinations.
        """
        print(f"\n" + "=" * 25 + " SANITY CHECK " + "=" * 25)
        print(f"QUERY: '{self.query}'")
        for i, stream in enumerate(self.context_streams):
            print(f"MEMORY STREAM {i + 1}: '{stream[:256]}...'")
        print("=" * 62)

        # Preparing all necessary data once
        encoded_streams = [self.tokenizer.encode(doc) for doc in self.context_streams]
        empty_stream = []
        num_slots = self.config['model']['num_memory_streams']

        results = {}

        # Starting with no memory, then add one stream at a time.
        for i in range(len(encoded_streams) + 1):
            # Creating the list of memory streams for this test case
            current_mems = encoded_streams[:i]
            # Padding the rest with empty streams
            current_mems.extend([empty_stream] * (num_slots - i))

            label = f"With {i} Memory Stream(s)"
            if i == 0:
                label = "No Memory"

            # Creating a context hint for the prompt
            context_hint = ""
            if i > 0:
                hint_tokens = encoded_streams[0][:15]  # Hint from the first real stream
                context_hint = self.tokenizer.decode(hint_tokens) + "..."

            prompt = format_prompt(self.query, context_hint, self.special_tokens)

            # Generating and store the result
            decoded_text, logits = self.generate_for_test_case(prompt, current_mems)
            results[label] = {"text": decoded_text, "logits": logits}

        # Performing Quantitative and Qualitative Analysis
        print("\n" + "=" * 20 + " QUALITATIVE RESULTS " + "=" * 20)
        for label, result in results.items():
            print(f"\n[{label}]\n{self.query} {result['text']}")

        print("\n" + "=" * 20 + " QUANTITATIVE RESULTS (KL Divergence) " + "=" * 20)
        labels = list(results.keys())
        for i in range(len(labels) - 1):
            label_a = labels[i]
            label_b = labels[i + 1]
            logits_a = results[label_a]["logits"]
            logits_b = results[label_b]["logits"]

            analyze_kl_divergence_torch(logits_b, logits_a, f"{label_a} vs. {label_b}")

        print("\n" + "=" * 20 + " FINAL ANALYSIS " + "=" * 20)
        final_output = results[f"With {len(encoded_streams)} Memory Stream(s)"]["text"]
        if "new york" in final_output.lower():
            print("\n✅ SUCCESS: The model clearly used the specific factual memory content.")
        else:
            print("\n❌ WARNING: The model did not clearly use the specific factual memory content.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a sanity check on a trained Multi-Memory Transformer.")
    parser.add_argument('--config', default="configs/gidionv_multi_memory.json", help="Path to the model's JSON config file.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = json.load(f)

    V4SanityChecker(cfg).run()
