from typing import Dict, Optional, List, Any, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.lib.core.attention import MultiHeadAttention
from src.lib.core.ffn import DynamicFeedForwardNetwork


class MemoryAttentionFusionDecoderBlock(nn.Module):
    """
    A stateful PyTorch implementation of a memory-augmented Transformer Decoder Block
    with Pre-LN architecture, multi-stream cross-attention, learnable fusion, and gating.
    """

    def __init__(self, d_model: int, num_heads: int, ff_hidden_config: List[int], num_memory_streams: int,
                 dropout_rate: float = 0.1, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.d_model = d_model
        self.num_memory_streams = num_memory_streams
        self.dtype = dtype

        # Self-Attention Components
        self.ln1 = nn.LayerNorm(d_model, dtype=dtype)
        self.masked_self_attn = MultiHeadAttention(d_model, num_heads, dropout_rate=0.0, dtype=dtype)
        self.dropout_self_attn = nn.Dropout(dropout_rate)

        # Cross-Attention Components
        self.cross_attention_layers = nn.ModuleList()
        for _ in range(num_memory_streams):
            layer_set = nn.ModuleDict({"ln_cross": nn.LayerNorm(d_model, dtype=dtype),
                                       "cross_attn": MultiHeadAttention(d_model, num_heads, dropout_rate=0.0,
                                                                        dtype=dtype)})
            self.cross_attention_layers.append(layer_set)

        if num_memory_streams > 0:
            scale = 0.02
            self.fusion_weights = nn.Parameter(torch.randn(num_memory_streams, dtype=dtype) * scale)
            self.memory_gate = nn.Parameter(torch.zeros(1, dtype=dtype))
            self.dropout_cross_attn = nn.Dropout(dropout_rate)

        # eed-Forward Components
        self.ln_ff = nn.LayerNorm(d_model, dtype=dtype)
        self.feed_forward = DynamicFeedForwardNetwork(input_size=d_model, output_size=d_model,
                                                      hidden_layers_config=ff_hidden_config, dropout_rate=0.0,
                                                      dtype=dtype)
        self.dropout_ffn = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, memory_streams: List[torch.Tensor],
                target_padding_mask: Optional[torch.Tensor] = None,
                memory_padding_masks: Optional[List[torch.Tensor]] = None, kv_cache: Optional[Dict[str, Any]] = None) -> \
            Tuple[torch.Tensor, Dict[str, Any]]:

        kv_cache = kv_cache or {}

        # Masked Self-Attention (Pre-LN)
        ln1_out = self.ln1(x)
        self_attn_output, _, self_attn_kv_cache = self.masked_self_attn(query=ln1_out, key=ln1_out, value=ln1_out,
                                                                        attn_mask=target_padding_mask, is_causal=True,
                                                                        # Replaces use_causal_mask=True
                                                                        kv_cache=kv_cache.get('self_attn'))

        # Dropout and Residual Connection
        residual1 = x + self.dropout_self_attn(self_attn_output)

        # Multi-Stream Cross-Attention (Pre-LN)
        cross_attention_outputs = []
        if memory_padding_masks is None: memory_padding_masks = [None] * len(memory_streams)

        for i, memory_context in enumerate(memory_streams):
            if memory_context.shape[1] == 0:
                cross_attention_outputs.append(torch.zeros_like(residual1))
                continue

            layer_set = self.cross_attention_layers[i]
            ln_cross_out = layer_set["ln_cross"](residual1)

            # Cross-attention: Query is from decoder (ln_cross_out), Key/Value are from memory
            cross_output, _, _ = layer_set["cross_attn"](query=ln_cross_out, key=memory_context, value=memory_context,
                                                         attn_mask=memory_padding_masks[i], is_causal=False)
            cross_attention_outputs.append(cross_output)

        # Fusion and Gating
        if cross_attention_outputs:
            stacked_cross_outputs = torch.stack(cross_attention_outputs, dim=0)
            fusion_softmax_weights = F.softmax(self.fusion_weights, dim=0)
            # Weighted average (Fusion)
            fused_cross_output = torch.tensordot(fusion_softmax_weights, stacked_cross_outputs, dims=([0], [0]))
            # Gating
            gate_value = torch.sigmoid(self.memory_gate)
            gated_fused_output = fused_cross_output * gate_value
            # Dropout and Residual Connection
            residual2 = residual1 + self.dropout_cross_attn(gated_fused_output)
        else:
            residual2 = residual1

        # Feed-Forward Network
        ln_ff_out = self.ln_ff(residual2)
        ff_output = self.feed_forward(ln_ff_out)

        # Dropout and Residual Connection
        final_output = residual2 + self.dropout_ffn(ff_output)

        new_kv_cache = {'self_attn': self_attn_kv_cache}
        return final_output, new_kv_cache

    @torch.no_grad()
    def project_component_weights_low_rank(self, rank_or_fraction: Union[int, float]):
        """Applies low-rank projection to all supported subcomponents (PyTorch version)."""
        print(f"Projecting MemoryAttentionFusionBlock weights")
        # Assuming the submodules have implemented this method
        self.masked_self_attn.project_projection_weights_low_rank(rank_or_fraction)
        for i, layer_set in enumerate(self.cross_attention_layers):
            print(f"Projecting Cross-Attention Stream {i}...")
            layer_set['cross_attn'].project_projection_weights_low_rank(rank_or_fraction)
        self.feed_forward.project_weights_low_rank(rank_or_fraction)
