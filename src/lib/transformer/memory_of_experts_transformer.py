import math
from typing import Dict, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.lib.core.hf_tokenizer_wrapper import HFTokenizerWrapper
from src.lib.core.hiearchical_fusion_decoder_block import HierarchicalFusionDecoderBlock
from src.lib.core.memory_encoder import MemoryEncoder
from src.lib.core.positional_encoding import PositionalEncoding
from src.lib.transformer.common import top_k_filtering, top_p_filtering


class MemoryOfExpertsTransformer(nn.Module):
    """
    [PyTorch] A complete, memory-augmented, stateful autoregressive Transformer
    that uses a hierarchical fusion mechanism to route information from expert memories.
    """

    def __init__(self, config: Dict, tokenizer: HFTokenizerWrapper):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.d_model = int(config['d_model'])
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_token_id = self.tokenizer.pad_token_id
        if self.pad_token_id is None:
            raise ValueError("Tokenizer must have a <pad> token.")

        dtype = getattr(torch, config.get('model_dtype', 'float32'))

        # Embedding and Positional Encoding Layers
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model, padding_idx=self.pad_token_id, dtype=dtype)
        self.positional_encoding = PositionalEncoding(self.d_model, max_len=config['max_seq_len'], dtype=dtype)
        self.input_dropout = nn.Dropout(config.get('dropout_rate', 0.1))

        # Memory Encoder
        self.memory_encoder = MemoryEncoder(num_layers=config['memory_encoder']['num_layers'], d_model=self.d_model,
                                            num_heads=config['memory_encoder']['num_heads'],
                                            ff_hidden_config=config['memory_encoder']['ff_hidden_config'],
                                            dropout_rate=config.get('dropout_rate', 0.1), dtype=dtype)

        # Hierarchical Decoder Stack
        decoder_config = config['decoder']
        self.decoder_blocks = nn.ModuleList([
            HierarchicalFusionDecoderBlock(d_model=self.d_model, num_heads=decoder_config['num_heads'],
                                           ff_hidden_config=decoder_config['ff_hidden_config'],
                                           dropout_rate=config.get('dropout_rate', 0.1), dtype=dtype) for _ in
            range(decoder_config['num_layers'])])

        # Final Output Layers
        self.final_norm = nn.LayerNorm(self.d_model, dtype=dtype)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False, dtype=dtype)

        # Weight Tying and Initialization
        if config.get('tie_weights', True):
            self.lm_head.weight = self.token_embedding.weight

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        num_layers = self.config['decoder']['num_layers']
        if isinstance(module, nn.Linear):
            scale_factor = 1 / math.sqrt(2.0 * num_layers)
            module.weight.data.normal_(mean=0.0, std=0.02 * scale_factor)

    def forward(self, input_ids: torch.Tensor,
                memory_streams_ids: Optional[List[torch.Tensor]] = None,
                kv_cache_list: Optional[List[Dict]] = None) -> \
            Tuple[torch.Tensor, Optional[List[Dict]], List[torch.Tensor]]:
        """Performs a full forward pass. No `cache` for backprop is needed."""

        unpadded_memory_contexts, unpadded_memory_masks = self._encode_memory_from_ids(memory_streams_ids)

        final_padded_contexts = []
        final_padded_masks = []

        if unpadded_memory_contexts:
            max_mem_len = max(ctx.shape[1] for ctx in unpadded_memory_contexts if ctx.numel() > 0) if any(
                ctx.numel() > 0 for ctx in unpadded_memory_contexts) else 0
            for context_tensor, mask_tensor in zip(unpadded_memory_contexts, unpadded_memory_masks):
                B, S, D = context_tensor.shape
                padding_needed = max_mem_len - S
                if padding_needed > 0:
                    padded_context = F.pad(context_tensor, (0, 0, 0, padding_needed), 'constant', 0)
                    padded_mask = F.pad(mask_tensor, (0, padding_needed), 'constant', False)
                else:
                    padded_context = context_tensor
                    padded_mask = mask_tensor
                final_padded_contexts.append(padded_context)
                final_padded_masks.append(padded_mask)

        # Prepare Main Input for Decoder
        target_padding_mask = (input_ids != self.pad_token_id)
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.input_dropout(x)

        # Pass through Decoder Stack
        next_kv_caches = [] if kv_cache_list is not None else None
        all_gating_outputs = []
        for i, block in enumerate(self.decoder_blocks):
            block_kv_cache = kv_cache_list[i] if kv_cache_list else None
            x, updated_kv_cache, gating_weights_output = block(x, target_padding_mask=target_padding_mask,
                                                               memory_streams=final_padded_contexts,
                                                               memory_padding_masks=final_padded_masks,
                                                               kv_cache=block_kv_cache)
            if gating_weights_output is not None:
                all_gating_outputs.append(gating_weights_output)
            if next_kv_caches is not None:
                next_kv_caches.append(updated_kv_cache)

        # Final Layers
        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits, next_kv_caches, all_gating_outputs

    def _prepare_memory_batch(
            self,
            memory_streams_ids_list: List[List[int]],
            device: torch.device
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        A helper function for the .generate() method.
        Takes a list of raw token ID lists (one for each memory stream) and
        prepares them into a batch suitable for the model's forward pass.
        This mimics the behavior of the training collate_fn.

        Args:
            memory_streams_ids_list: A list where each item is another list of token IDs.
                                     e.g., [[2, 5, 8], [10, 20, 30, 40], []]
            device: The torch device to place the final tensors on.

        Returns:
            A tuple of (batched_ids, batched_masks), where each is a list of tensors.
        """
        # We assume a batch size of 1 for generation, which is standard.
        batch_size = 1

        batched_ids = []
        batched_masks = []

        for id_list in memory_streams_ids_list:
            # For each memory stream, creating a padded tensor
            seq_len = len(id_list)

            # Creating a tensor for this single stream
            # Shape: (1, seq_len) to represent a batch of 1
            tensor = torch.full((batch_size, seq_len), self.pad_token_id, dtype=torch.long, device=device)
            if seq_len > 0:
                tensor[0, :seq_len] = torch.tensor(id_list, dtype=torch.long, device=device)

            # Creating the corresponding mask
            mask = (tensor != self.pad_token_id)

            batched_ids.append(tensor)
            batched_masks.append(mask)

        return batched_ids, batched_masks

    def _encode_memory_from_ids(
            self,
            memory_streams_ids: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Takes a list of raw token ID tensors and runs them through the
        embedding, positional encoding, and memory encoder pipeline.

        Returns:
            A tuple of (memory_contexts, memory_padding_masks).
        """
        memory_contexts = []
        memory_padding_masks = []

        for ids in memory_streams_ids:
            # Only process streams that actually contain tokens
            if ids.numel() > 0 and ids.shape[1] > 0:
                # Creating the padding mask from the raw IDs
                padding_mask = (ids != self.pad_token_id)
                # Encoding the stream
                mem_emb = self.token_embedding(ids) * math.sqrt(self.d_model)
                mem_pos_emb = self.positional_encoding(mem_emb)
                mem_ctx = self.memory_encoder(mem_pos_emb, padding_mask=padding_mask)

                memory_contexts.append(mem_ctx)
                memory_padding_masks.append(padding_mask)
            else:
                # Handling empty streams by adding empty placeholders
                B = memory_streams_ids[0].shape[0] if memory_streams_ids and memory_streams_ids[0].numel() > 0 else 1
                device = self.token_embedding.weight.device
                dtype = self.token_embedding.weight.dtype

                empty_ctx = torch.empty((B, 0, self.d_model), device=device, dtype=dtype)
                empty_mask = torch.empty((B, 0), dtype=torch.bool, device=device)

                memory_contexts.append(empty_ctx)
                memory_padding_masks.append(empty_mask)

        return memory_contexts, memory_padding_masks

    @torch.no_grad()
    def generate(self,
                 prompt_ids: torch.Tensor,
                 memory_streams_ids: List[List[int]],
                 max_new_tokens: int = 128,
                 temperature: float = 0.7,
                 top_k: int = 50,
                 top_p: int = 1.0,
                 repetition_penalty: float = 1.1,
                 eos_token_id: Optional[int] = None,
                 return_logits: bool = False) -> tuple[Tensor, Tensor] | Tensor:
        """
        Generates text autoregressively.
        Assumes memory contexts are pre-computed and pre-padded.
        """
        self.eval()
        device = prompt_ids.device
        batched_ids, batched_masks = self._prepare_memory_batch(memory_streams_ids, device)
        generated_ids = prompt_ids
        kv_caches = None
        logits = None

        batch_size = prompt_ids.shape[0]
        is_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            # Preparing inputs for the forward pass
            input_ids_step = generated_ids[:, -1:] if kv_caches is not None else generated_ids
            logits, kv_caches, _ = self.forward(
                input_ids=input_ids_step,
                memory_streams_ids=batched_ids,
                kv_cache_list=kv_caches
            )
            # Getting logits for the last token only
            logits = logits[:, -1, :]

            # Applying repetition penalty
            if repetition_penalty != 1.0:
                # Creating a view of the logits for sequences that are not yet finished
                unfinished_logits = logits[~is_finished]
                unfinished_generated_ids = generated_ids[~is_finished]

                if unfinished_generated_ids.shape[1] > 0:
                    # Applying penalty only on this subset
                    for i in range(unfinished_logits.shape[0]):
                        unique_tokens = torch.unique(unfinished_generated_ids[i])
                        unfinished_logits[i, unique_tokens] = torch.where(
                            unfinished_logits[i, unique_tokens] > 0,
                            unfinished_logits[i, unique_tokens] / repetition_penalty,
                            unfinished_logits[i, unique_tokens] * repetition_penalty
                        )
                    # Placing the modified logits back into the main logits tensor
                    logits[~is_finished] = unfinished_logits

            # Applying temperature
            if temperature > 0:
                logits = logits / temperature

            # Applying top-k filtering
            if top_k > 0:
                logits = top_k_filtering(logits, top_k)

            # Applying Top-P (nucleus) filtering SECOND
            if top_p < 1.0:
                logits = top_p_filtering(logits, top_p)

            # Sampling the next token
            if temperature == 0:
                # Greedy decoding: simply pick the token with the highest score
                next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
            else:
                # Random sampling: calculate probabilities and sample
                # We apply softmax to the filtered logits to get a valid probability distribution.
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            # Appending the new token
            generated_ids = torch.cat((generated_ids, next_token), dim=1)

            # Checking for end-of-sequence token
            if eos_token_id is not None:
                is_finished = is_finished | (next_token == eos_token_id).squeeze()

            if is_finished.all():
                break

        self.train()

        if return_logits:
            return generated_ids, logits
        else:
            return generated_ids
