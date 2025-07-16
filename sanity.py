import argparse
import json
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F

from src.config.config import get_config
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

    # No need for [None, ...] since batch dim should exist
    if logits_a.ndim == 2:
        logits_a = logits_a.unsqueeze(0)
    if logits_b.ndim == 2:
        logits_b = logits_b.unsqueeze(0)

    kl = kl_divergence_torch(logits_a[:, :min_len, :], logits_b[:, :min_len, :])
    print(f"KL Divergence ({label}): {kl.item():.6f}")
    return kl


class V4SanityChecker:
    def __init__(self, config: Dict[str, Any]):
        print("--- Initializing PyTorch V4 Sanity Checker ---")
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model, _, self.tokenizer, _ = load_checkpoint(config, self.device, use_best=False)
        self.model.eval()  # Set to evaluation mode

        self.query = "What did the new survey reveal about the America?"

        self.context_streams = [
            "The American Community Survey (ACS) is an annual demographics survey program conducted by the United States Census Bureau. It regularly gathers information previously contained only in the long form of the decennial census, including ancestry, US citizenship status, educational attainment, income, language proficiency, migration, disability, employment, and housing characteristics. No respondents personal information is released, and only used statistically in these data which are used by many public-sector, private-sector, and not-for-profit stakeholders to allocate funding, track shifting demographics, plan for emergencies, and learn about local communities.",
            "You have to reflect yourself, verify, correct and answer.",
            "NewYork is the capital of United States of America."]

    def encode_memory_streams(self) -> List[List[int]]:
        """
        [CORRECTED] Encodes memory streams into a list of token ID lists.
        """
        print("\n Encoding Memory Streams")
        # The output is now a simple list of lists, e.g., [[1,2,3], [4,5,6]]
        return [self.tokenizer.encode(doc) for doc in self.context_streams]

    @torch.no_grad()
    def generate_with_memory(self,
                             memory_streams_ids_list: List[List[int]],
                             label: str) -> Tuple[str, Optional[torch.Tensor]]:
        """
        [CORRECTED] Generates a response using the model, preparing inputs correctly.
        """
        print(f"\n--- Generating with {label} ---")

        # Preparing the Prompt
        special_tokens = {"USER": "<USER>", "ASSISTANT": "<ASSISTANT>"}

        # Creating a small "hint" from the context to include in the prompt
        # This helps the model know what kind of information is available.
        context_hint = ""
        if memory_streams_ids_list and memory_streams_ids_list[0]:
            # Getting the first few tokens of the first memory stream and decode them
            hint_tokens = memory_streams_ids_list[0][:15]
            context_hint = self.tokenizer.decode(hint_tokens) + "..."

        # Formatting the final prompt string
        final_prompt_string = format_prompt(self.query, context_hint, special_tokens)
        print(f"Formatted Prompt:\n{final_prompt_string}")

        prompt_ids = torch.tensor([self.tokenizer.encode(final_prompt_string)], dtype=torch.long, device=self.device)

        # Calling the Model's Generate Method
        # The `generate` method now handles all the complex internal processing.
        generated_ids_tensor, logits_returned = self.model.generate(
            prompt_ids=prompt_ids,
            memory_streams_ids=memory_streams_ids_list,
            max_new_tokens=100,
            temperature=0.7,
            top_k=50,
            eos_token_id=self.tokenizer.eos_token_id,
            return_logits=True
        )

        # Decoding the Output
        # The generate method returns the full sequence (prompt + new tokens)
        prompt_len = prompt_ids.shape[1]
        newly_generated_ids = generated_ids_tensor[0, prompt_len:]

        decoded_response = self.tokenizer.decode(newly_generated_ids.tolist(), skip_special_tokens=True).strip()

        return decoded_response, logits_returned

    def run(self):
        print(f"\nQUERY: '{self.query}'")
        for i, stream in enumerate(self.context_streams, 1):
            print(f"MEMORY STREAM {i}: '{stream[:80]}...'")

        encoded_streams = self.encode_memory_streams()
        # A "zero stream" in this context is just an empty document
        zero_stream = []

        # A) No Memory
        output_no_mem, logits_no_mem = self.generate_with_memory([zero_stream, zero_stream, zero_stream],
                                                                 "NO MEMORY")
        # A) No Memory
        output_no_mem_a, logits_no_mem_a = self.generate_with_memory([zero_stream, zero_stream, zero_stream],
                                                                     "NO MEMORY")

        # B) Memory Stream 1 Only
        output_mem1, logits_mem1 = self.generate_with_memory(
            [encoded_streams[0], zero_stream, zero_stream], "Memory Stream 1 ONLY")

        # C) Stream 1 + 2 + Zero + Zero
        output_mem2, logits_mem2 = self.generate_with_memory(
            [encoded_streams[0], encoded_streams[1], zero_stream], "Stream 1 + 2")

        # D) Stream 1 + 2 + 3 + Zero
        output_all, logits_all = self.generate_with_memory(
            [encoded_streams[0], encoded_streams[1], encoded_streams[2]], "ALL Memory Streams")

        # Quantitative KL Analysis
        print("\n" + "=" * 20 + " QUANTITATIVE RESULTS " + "=" * 20)
        kl1 = analyze_kl_divergence_torch(logits_no_mem_a, logits_no_mem, "No Memory vs. No memory")
        kl1 = analyze_kl_divergence_torch(logits_mem1, logits_no_mem, "No Memory vs. Stream 1")
        kl2 = analyze_kl_divergence_torch(logits_mem2, logits_mem1, "Stream 1 vs. Stream 1+2")
        kl3 = analyze_kl_divergence_torch(logits_all, logits_mem2, "Stream 1+2 vs. All")

        # Analysis logic is identical
        print("\n--- Analysis ---")
        if kl1 is not None and kl1.item() > 0.1:
            print("✅ Stream 1 had significant effect.")
        else:
            print("⚠️ Stream 1 had little effect.")

        print("\n" + "=" * 20 + " QUALITATIVE RESULT " + "=" * 20)
        print(f"\n[No Memory [0, 0, 0]]\n{output_no_mem}")
        print(f"\n[With Stream [1, 0, 0]]\n{output_mem1}")
        print(f"\n[With Stream [1, 2, 0]\n{output_mem2}")
        print(f"\n[With All Streams, [1, 2, 3]\n{output_all}")

        # I updated the final context stream to be more obvious
        if "marseille" in output_all.lower():
            print("\n✅ SUCCESS: The model used memory content.")
        else:
            print("\n❌ Model did not clearly use memory content.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Gidion Augmented Transformer.")
    parser.add_argument('--config', default='configs/gidionv_multi_memory.json', type=str,
                        help="Path to a JSON config file to override defaults.")
    args = parser.parse_args()

    cfg = get_config()
    if args.config:
        with open(args.config, 'r') as f:
            cfg.update(json.load(f))

    V4SanityChecker(cfg).run()
