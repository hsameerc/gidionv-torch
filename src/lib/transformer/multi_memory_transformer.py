import math
from typing import Dict, Optional, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.lib.core.hf_tokenizer_wrapper import HFTokenizerWrapper
from src.lib.core.maf_decoder_block import MemoryAttentionFusionDecoderBlock
from src.lib.core.memory_encoder import MemoryEncoder
from src.lib.core.positional_encoding import PositionalEncoding
from src.lib.transformer.common import top_k_filtering, top_p_filtering


class MultiMemoryTransformer(nn.Module):
    """
    A complete, memory-augmented, stateful autoregressive Transformer in PyTorch.
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

        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model, padding_idx=self.pad_token_id, dtype=dtype)
        self.positional_encoding = PositionalEncoding(self.d_model, max_len=config['max_seq_len'], dtype=dtype)
        self.input_dropout = nn.Dropout(config.get('dropout_rate', 0.1))

        self.memory_encoder = MemoryEncoder(num_layers=config['memory_encoder']['num_layers'], d_model=self.d_model,
                                            num_heads=config['memory_encoder']['num_heads'],
                                            ff_hidden_config=config['memory_encoder']['ff_hidden_config'],
                                            dropout_rate=config.get('dropout_rate', 0.1), dtype=dtype)

        decoder_config = config['decoder']
        self.decoder_blocks = nn.ModuleList([
            MemoryAttentionFusionDecoderBlock(d_model=self.d_model, num_heads=decoder_config['num_heads'],
                                              ff_hidden_config=decoder_config['ff_hidden_config'],
                                              num_memory_streams=config['model']['num_memory_streams'],
                                              dropout_rate=config.get('dropout_rate', 0.1), dtype=dtype) for _ in
            range(decoder_config['num_layers'])])

        self.final_norm = nn.LayerNorm(self.d_model, dtype=dtype)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False, dtype=dtype)

        if config.get('tie_weights', True):
            self.lm_head.weight = self.token_embedding.weight

    def init_weights(self, module):
        """
        Applies a custom weight initialization scheme, crucial for deep transformers.
        """
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
            Tuple[torch.Tensor, Optional[List[Dict]], Optional[List[torch.Tensor]], Optional[List[torch.Tensor]]]:
        """
          Returns:
              A tuple containing:
              - logits (torch.Tensor): The final output logits.
              - next_kv_caches (Optional[List[Dict]]): List of updated KV caches for each decoder layer.
              - all_fusion_weights (List[torch.Tensor]): List of fusion weights from each decoder layer.
        """
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
        x = self.positional_encoding(x, start_pos=kv_cache_list[0]['self_attn'][0].shape[2] if kv_cache_list else 0)
        x = self.input_dropout(x)

        # Pass through Fusion Decoder Stack
        next_kv_caches = [] if kv_cache_list is not None else None
        all_fusion_weights = []
        all_attention_maps_data_map = []
        for i, block in enumerate(self.decoder_blocks):
            block_kv_cache = kv_cache_list[i] if kv_cache_list else None

            x, updated_kv_cache, fusion_weights, all_attention_maps = block(
                x,
                memory_streams=final_padded_contexts,
                target_padding_mask=target_padding_mask,
                memory_padding_masks=final_padded_masks,
                kv_cache=block_kv_cache,
                output_attentions=True,
            )
            if fusion_weights is not None:
                # .detach() is important so we don't hold onto the computation graph
                all_fusion_weights.append(fusion_weights.detach())
            if all_attention_maps is not None:
                # .detach() is important so we don't hold onto the computation graph
                all_attention_maps_data_map.append(all_attention_maps)
            if next_kv_caches is not None:
                next_kv_caches.append(updated_kv_cache)

        # Final Layers
        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits, next_kv_caches, all_fusion_weights, all_attention_maps_data_map

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
                 top_p: float = 0.95,
                 repetition_penalty: float = 1.1,
                 eos_token_id: Optional[int] = None,
                 return_logits: bool = False
                 ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generates text autoregressively with correct
        batched generation, sampling logic, and logit collection.
        """
        self.eval()
        device = prompt_ids.device

        # One-Time Memory Preparation
        batched_ids, batched_masks = self._prepare_memory_batch(memory_streams_ids, device)
        # pre_computed_contexts, pre_computed_masks = self._encode_memory_from_ids(batched_ids)

        # Generation Setup
        generated_ids = prompt_ids
        batch_size = prompt_ids.shape[0]
        is_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        kv_caches = None

        collected_logits = []

        # Autoregressive Generation Loop
        for _ in range(max_new_tokens):
            # Preparing inputs for the forward pass (handle KV cache)
            input_ids_step = generated_ids[:, -1:] if kv_caches is not None else generated_ids

            # Getting the raw logits from the model
            logits, kv_caches, _ = self.forward(
                input_ids=input_ids_step,
                memory_streams_ids=batched_ids,
                kv_cache_list=kv_caches
            )
            # We only care about the logits for the very last token
            last_token_logits = logits[:, -1, :]

            # Storing a clone of the original logits BEFORE any modifications.
            # This is what we will be returning for analysis.
            if return_logits:
                collected_logits.append(last_token_logits.clone())

            # Apply Sampling Logic
            sampling_logits = last_token_logits

            # Applying repetition penalty ONLY to unfinished sequences
            if repetition_penalty != 1.0:
                unfinished_mask = ~is_finished
                if unfinished_mask.any():
                    # Applying penalty only on the subset of logits and IDs
                    for i in torch.where(unfinished_mask)[0]:
                        unique_tokens = torch.unique(generated_ids[i])
                        sampling_logits[i, unique_tokens] = torch.where(
                            sampling_logits[i, unique_tokens] > 0,
                            sampling_logits[i, unique_tokens] / repetition_penalty,
                            sampling_logits[i, unique_tokens] * repetition_penalty
                        )

            # Applying temperature
            if temperature > 0:
                sampling_logits = sampling_logits / temperature

            # Applying Top-K and Top-P filtering
            if top_k > 0:
                sampling_logits = top_k_filtering(sampling_logits, top_k)
            if top_p < 1.0:
                sampling_logits = top_p_filtering(sampling_logits, top_p)

            # Force PAD token for already finished sequences. This must be the LAST modification.
            if is_finished.any():
                sampling_logits[is_finished] = -float('Inf')
                sampling_logits[is_finished, self.pad_token_id] = 0.0

            # Sample the Next Token
            if temperature == 0:
                next_token = torch.argmax(sampling_logits, dim=-1).unsqueeze(-1)
            else:
                probs = F.softmax(sampling_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            # Updating State
            generated_ids = torch.cat((generated_ids, next_token), dim=1)

            if eos_token_id is not None:
                # Updating finished status for sequences that just generated EOS
                is_finished = is_finished | (next_token == eos_token_id).squeeze(-1)

            # If all sequences in the batch are finished, we can stop early.
            if is_finished.all():
                break

        self.train()

        # Preparing Final Output
        if return_logits:
            # Stacking the collected UNFILTERED logits into a single tensor
            # Shape: (batch_size, num_generated_tokens, vocab_size)
            final_logits = torch.stack(collected_logits, dim=1)
            return generated_ids, final_logits
        else:
            return generated_ids

    def __call__(self, *args, **kwargs):
        """Standard callable interface."""
        return self.forward(*args, **kwargs)
