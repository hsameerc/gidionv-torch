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

    def forward(self, input_ids: torch.Tensor, memory_streams_ids: Optional[List[torch.Tensor]] = None,
                memory_contexts: Optional[List[torch.Tensor]] = None,
                memory_padding_masks: Optional[List[torch.Tensor]] = None,
                kv_cache_list: Optional[List[Dict]] = None) -> \
            Tuple[torch.Tensor, Optional[List[Dict]]]:
        """Performs a full forward pass. No `cache` for backprop is needed."""

        # Encode Memory Streams / Contexts
        if memory_contexts is None:
            # Create or Use padding masks for all memory streams
            if memory_padding_masks is None:
                memory_padding_masks = [(ids != self.pad_token_id) for ids in memory_streams_ids]
            memory_contexts = []
            if memory_streams_ids:
                for i, ids in enumerate(memory_streams_ids):
                    if memory_padding_masks is not None:
                        mem_padding_mask = memory_padding_masks[i]
                    else:
                        mem_padding_mask = (ids != self.pad_token_id)
                    mem_emb = self.token_embedding(ids) * (self.d_model ** 0.5)
                    mem_pos_emb = self.positional_encoding(mem_emb)
                    mem_ctx = self.memory_encoder(mem_pos_emb, padding_mask=mem_padding_mask)
                    memory_contexts.append(mem_ctx)

        # Process Main Input
        target_padding_mask = (input_ids != self.pad_token_id)
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.input_dropout(x)

        # Pass through Decoder Stack
        next_kv_caches = [] if kv_cache_list is not None else None
        # This needs a combined padding mask for the fused memory context
        # For simplicity, we can create it on the fly or assume it's handled.
        # Let's assume a simple case for now where the fused context doesn't need a complex mask.
        fused_memory_padding_mask = None

        for i, block in enumerate(self.decoder_blocks):
            block_kv_cache = kv_cache_list[i] if kv_cache_list else None
            x, updated_kv_cache = block(x, memory_streams=memory_contexts, target_padding_mask=target_padding_mask,
                                        memory_padding_mask=fused_memory_padding_mask, kv_cache=block_kv_cache)
            if next_kv_caches is not None:
                next_kv_caches.append(updated_kv_cache)

        # Final Layers
        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits, next_kv_caches

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, memory_streams_ids: List[List[List[int]]], max_new_tokens: int,
                 temperature: float = 0.7, top_p: float = 0.9, top_k: int = 0, repetition_penalty: float = 1.5,
                 eos_token_id: Optional[int] = None, return_logits: bool = False) -> tuple[Tensor, Tensor] | Tensor:
        """
        Generates text sequences autoregressively using PyTorch.
        This method is wrapped in `torch.no_grad()` for performance.
        """
        self.eval()
        device = prompt_ids.device

        # One-Time Memory Encoding
        memory_contexts = []
        for stream_batch_ids_list in memory_streams_ids:
            max_len = max(len(ids) for ids in stream_batch_ids_list)
            padded_ids = torch.full((len(stream_batch_ids_list), max_len), self.pad_token_id, dtype=torch.long,
                                    device=device)
            for i, ids in enumerate(stream_batch_ids_list):
                padded_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)

            # Re-use the forward pass logic for encoding
            mem_emb = self.token_embedding(padded_ids) * (self.d_model ** 0.5)
            mem_pos_emb = self.positional_encoding(mem_emb)
            mem_ctx = self.memory_encoder(mem_pos_emb, padding_mask=(padded_ids != self.pad_token_id))
            memory_contexts.append(mem_ctx)

        # Autoregressive Generation Loop
        generated_ids = prompt_ids
        batch_size = prompt_ids.shape[0]
        is_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        kv_caches = None
        logits = None
        for _ in range(max_new_tokens):
            # Passing only the last token for generation after the first step
            input_ids_step = generated_ids[:, -1:] if kv_caches else generated_ids

            logits, kv_caches = self.forward(input_ids=input_ids_step, memory_contexts=memory_contexts,
                                             kv_cache_list=kv_caches)
            # Getting the logits for the very last token
            logits = logits[:, -1, :]

            # Applying repetition penalty
            if repetition_penalty != 1.0 and _ > 0:
                scores = torch.gather(logits, 1, generated_ids)
                scores = torch.where(scores > 0, scores / repetition_penalty, scores * repetition_penalty)
                logits.scatter_(1, generated_ids, scores)

            # Applying temperature and top-p sampling
            if temperature > 0:
                logits = logits / temperature
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k)
                    min_inf_mask = torch.full_like(logits, -float('Inf'))
                    logits = min_inf_mask.scatter_(1, top_k_indices, top_k_logits)
                if 0 < top_p < 1.0:
                    # Top-p filtering
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float('Inf')

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            else:  # Greedy
                next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)

            if eos_token_id is not None:
                is_finished = is_finished | (next_token == eos_token_id)

            next_token = next_token.masked_fill(is_finished, self.pad_token_id)
            generated_ids = torch.cat((generated_ids, next_token), dim=1)

            # Check for EOS token
            if is_finished.all():
                break

        self.train()
        if return_logits:
            return generated_ids, logits
        else:
            return generated_ids
