from typing import Dict, Optional, List, Tuple, Any

import torch
import torch.nn as nn

from src.lib.core.attention import MultiHeadAttention
from src.lib.core.ffn import DynamicFeedForwardNetwork
from src.lib.core.lif_ffn import LIFFfn


class HierarchicalFusionDecoderBlock(nn.Module):
    """
    A stateful decoder block for a Hierarchical Mixture of Experts model.
    This block learns to dynamically route information from a set of expert memory streams.
    """

    def __init__(self, d_model: int, num_heads: int, ff_hidden_config: List[int], dropout_rate: float = 0.1,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.d_model = d_model

        # Self-Attention Sub-layer
        self.ln1 = nn.LayerNorm(d_model, dtype=dtype)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout_rate=0.0, dtype=dtype)
        self.dropout1 = nn.Dropout(dropout_rate)

        # Gating and Cross-Attention Sub-layer
        self.ln_gate = nn.LayerNorm(d_model, dtype=dtype)
        # Gating attention often uses a single head to produce one score per expert.
        self.gating_attention = MultiHeadAttention(d_model, num_heads=1, dropout_rate=0.0, dtype=dtype)

        self.ln_cross = nn.LayerNorm(d_model, dtype=dtype)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout_rate=0.0, dtype=dtype)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Feed-Forward Sub-layer
        self.ln_ff = nn.LayerNorm(d_model, dtype=dtype)
        self.feed_forward = LIFFfn(input_size=d_model, output_size=d_model,
                                                      hidden_layers_config=ff_hidden_config, dropout_rate=dropout_rate,
                                                      dtype=dtype)
        self.dropout3 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, memory_streams: List[torch.Tensor],
                target_padding_mask: Optional[torch.Tensor] = None,
                memory_padding_masks: Optional[List[torch.Tensor]] = None,
                kv_cache: Optional[Dict[str, Any]] = None) -> Tuple[
        torch.Tensor, Dict[str, Any], Optional[torch.Tensor]]:
        kv_cache = kv_cache if kv_cache is not None else {}
        gating_weights_output = None
        # Masked Self-Attention
        ln1_out = self.ln1(x)
        self_attn_output, _, self_attn_kv_cache = self.self_attn(query=ln1_out, key=ln1_out, value=ln1_out,
                                                                 is_causal=True, attn_mask=target_padding_mask,
                                                                 kv_cache=kv_cache.get('self_attn'))
        residual1 = x + self.dropout1(self_attn_output)

        # Gating, Fusion, and Cross-Attention
        residual2 = residual1
        if memory_streams:
            # Gating Mechanism
            ln_gate_out = self.ln_gate(residual1)
            # Let's create a single query vector per sequence by averaging token representations
            gating_query = ln_gate_out.mean(dim=1, keepdim=True)  # Shape: (batch, 1, d_model)

            # Now let's create a summary vector for each expert stream
            # Stack: (num_experts, batch, seq_len, d_model) -> requires careful handling or loops
            # memory_streams is a list of tensors [ (B, S_mem, D), ... ]
            all_experts_tensor = torch.stack(memory_streams, dim=1)  # Shape: (B, num_experts, S_mem, D)
            expert_summaries = all_experts_tensor.mean(dim=2)  # Shape: (B, num_experts, D)

            # Gating attention produces scores for each expert
            # Query: (B, 1, D), Key/Value: (B, num_experts, D)
            _, gating_weights, _ = self.gating_attention(query=gating_query, key=expert_summaries,
                                                         value=expert_summaries)  # gating_weights shape: (B, 1, 1, num_experts)

            gating_weights_output = gating_weights
            gating_weights_output = gating_weights_output.squeeze((1, 2))

            # Now we use the scores to create a weighted sum of the expert streams
            # gating_weights needs to be broadcastable to all_experts_tensor
            # (B, 1, 1, num_experts) -> (B, num_experts, 1, 1) to multiply with (B, num_experts, S_mem, D)
            gating_scores = gating_weights.squeeze(1).permute(0, 2, 1).unsqueeze(-1)

            # Fused context is a weighted average of the expert streams
            fused_memory_context = (all_experts_tensor * gating_scores).sum(dim=1)  # Shape: (B, S_mem, D)

            # Stacking the list of masks into a single tensor: (B, num_experts, S_max)
            fused_memory_padding_mask = None
            if memory_padding_masks:
                all_masks_tensor = torch.stack(memory_padding_masks, dim=1)
                # The result is a single mask of shape (B, S_max).
                # A position is `True` if it was `True` in at least one of the stream masks.
                fused_memory_padding_mask = torch.any(all_masks_tensor, dim=1)

            # Cross-Attention to Fused Context
            ln_cross_out = self.ln_cross(residual1)
            cross_output, _, _ = self.cross_attention(query=ln_cross_out, key=fused_memory_context,
                                                      value=fused_memory_context, attn_mask=fused_memory_padding_mask
                                                      # This mask needs to correspond to the fused context
                                                      )
            residual2 = residual1 + self.dropout2(cross_output)

        # Feed-Forward Network
        ln_ff_out = self.ln_ff(residual2)
        ff_output = self.feed_forward(ln_ff_out)
        final_output = residual2 + self.dropout3(ff_output)

        new_kv_cache = {'self_attn': self_attn_kv_cache}
        return final_output, new_kv_cache, gating_weights_output
