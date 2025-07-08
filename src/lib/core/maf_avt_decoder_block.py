from typing import Dict, Optional, List, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadAttention
from .ffn import DynamicFeedForwardNetwork


class MemoryAttentionFusionDecoderBlock(nn.Module):
    """
    A stateful decoder block that attends to multiple memory streams
    and fuses their outputs using a learnable softmax-based weighting.
    """

    def __init__(self, d_model: int, num_heads: int, ff_hidden_config: List[int], num_memory_streams: int,
                 dropout_rate: float = 0.1, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.num_memory_streams = num_memory_streams

        # Self-Attention Sub-layer
        self.ln1 = nn.LayerNorm(d_model, dtype=dtype)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout_rate=0.0, dtype=dtype)
        self.dropout1 = nn.Dropout(dropout_rate)

        # Multi-Stream Cross-Attention Sub-layer
        if num_memory_streams > 0:
            # A list of LayerNorms, one for each cross-attention head
            self.cross_attention_lns = nn.ModuleList(
                [nn.LayerNorm(d_model, dtype=dtype) for _ in range(num_memory_streams)])
            # A list of MHA layers, one for each memory stream
            self.cross_attentions = nn.ModuleList(
                [MultiHeadAttention(d_model, num_heads, dropout_rate=0.0, dtype=dtype) for _ in
                 range(num_memory_streams)])

            # The learnable weights for fusing the outputs
            self.fusion_weights = nn.Parameter(torch.zeros(num_memory_streams, dtype=dtype))
            self.dropout2 = nn.Dropout(dropout_rate)

        # Feed-Forward Sub-layer
        self.ln_ff = nn.LayerNorm(d_model, dtype=dtype)
        self.feed_forward = DynamicFeedForwardNetwork(input_size=d_model, output_size=d_model,
                                                      hidden_layers_config=ff_hidden_config, dropout_rate=dropout_rate,
                                                      dtype=dtype)
        self.dropout3 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, memory_streams: List[torch.Tensor],
                target_padding_mask: Optional[torch.Tensor] = None,
                memory_padding_masks: Optional[List[torch.Tensor]] = None, kv_cache: Optional[Dict[str, Any]] = None) -> \
            Tuple[torch.Tensor, Dict[str, Any]]:

        kv_cache = kv_cache if kv_cache is not None else {}

        # Masked Self-Attention
        ln1_out = self.ln1(x)
        self_attn_output, _, self_attn_kv_cache = self.self_attn(query=ln1_out, key=ln1_out, value=ln1_out,
                                                                 is_causal=True, attn_mask=target_padding_mask,
                                                                 kv_cache=kv_cache.get('self_attn'))
        residual1 = x + self.dropout1(self_attn_output)

        # Cross-Attention and Softmax Fusion
        residual2 = residual1
        if self.num_memory_streams > 0 and memory_streams:
            cross_attention_outputs = []
            if memory_padding_masks is None:
                memory_padding_masks = [None] * len(memory_streams)

            for i, memory_context in enumerate(memory_streams):
                # Apply LayerNorm before this stream's cross-attention
                ln_cross_out = self.cross_attention_lns[i](residual1)

                # Perform cross-attention to the i-th memory stream
                cross_output, _, _ = self.cross_attentions[i](query=ln_cross_out, key=memory_context,
                                                              value=memory_context, attn_mask=memory_padding_masks[i])
                cross_attention_outputs.append(cross_output)

            if cross_attention_outputs:
                # Stack the outputs from each stream: (num_streams, B, S, D)
                stacked_cross_outputs = torch.stack(cross_attention_outputs, dim=0)

                # Compute the fusion weights using softmax
                # The autograd engine will handle the backward pass for this!
                fusion_softmax_weights = F.softmax(self.fusion_weights, dim=0)

                # Reshape weights for broadcasting: (num_streams, 1, 1, 1)
                reshaped_weights = fusion_softmax_weights.view(-1, 1, 1, 1)

                # Compute the weighted sum (fusion)
                fused_cross_output = (stacked_cross_outputs * reshaped_weights).sum(dim=0)

                # Add to the residual stream
                residual2 = residual1 + self.dropout2(fused_cross_output)

        # Feed-Forward Network
        ln_ff_out = self.ln_ff(residual2)
        ff_output = self.feed_forward(ln_ff_out)
        final_output = residual2 + self.dropout3(ff_output)

        new_kv_cache = {'self_attn': self_attn_kv_cache}
        return final_output, new_kv_cache
