from typing import List, Optional, Union

import torch
import torch.nn as nn

from src.lib.core.attention import MultiHeadAttention
from src.lib.core.ffn import DynamicFeedForwardNetwork


class TransformerEncoderBlock(nn.Module):
    """
    A stateful PyTorch implementation of a Transformer Encoder block.
    This module uses a Pre-LN architecture as described in the original code.
    """

    def __init__(self, d_model: int, num_heads: int, ff_hidden_config: List[int], dropout_rate: float = 0.1,
                 eps: float = 1e-6, dtype: torch.dtype = torch.float32):
        """
        Initializes the Encoder block and all its stateful submodules.
        """
        super().__init__()

        # Self-Attention Branch
        self.ln1 = nn.LayerNorm(d_model, eps=eps, dtype=dtype)
        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            dtype=dtype
        )
        self.dropout1 = nn.Dropout(dropout_rate)

        # Feed-Forward Branch
        self.ln2 = nn.LayerNorm(d_model, eps=eps, dtype=dtype)
        self.feed_forward = DynamicFeedForwardNetwork(
            input_size=d_model,
            output_size=d_model,
            hidden_layers_config=ff_hidden_config,
            dropout_rate=dropout_rate,
            dtype=dtype
        )
        # Dropout for the second residual connection.
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Performs a forward pass through the Encoder block using Pre-LN architecture.
        """
        # Self-Attention Block
        attn_output, _, _ = self.self_attn(
            query=self.ln1(x),
            key=self.ln1(x),
            value=self.ln1(x),
            attn_mask=padding_mask
        )
        # Add & Norm: Add the residual connection after applying dropout to the attention output.
        x = x + self.dropout1(attn_output)

        # Feed-Forward Block
        ff_output = self.feed_forward(self.ln2(x))
        # Add & Norm: Add the second residual connection after dropout.
        x = x + self.dropout2(ff_output)

        return x

    def get_params(self):
        """Replaced by model.parameters()"""
        return self.parameters()

    def get_named_params(self):
        """Replaced by model.named_parameters()"""
        return self.named_parameters()

    def set_params(self, state_dict):
        """Replaced by model.load_state_dict()"""
        self.load_state_dict(state_dict)

    @torch.no_grad()
    def analyze_projection_weights_svd(self, top_n: int = 5):
        """Analyzes weights of FFN and MHA components using SVD."""
        print(f"--- SVD Analysis for PyTorch TransformerEncoderBlock ---")
        if hasattr(self.feed_forward, 'analyze_weights_svd'):
            self.feed_forward.analyze_weights_svd(top_n_singular_values=top_n)
        if hasattr(self.self_attn, 'analyze_projection_weights_svd'):
            self.self_attn.analyze_projection_weights_svd(top_n=top_n)
        print(f"--- End SVD Analysis ---")

    def project_component_weights_low_rank(self, rank_or_fraction: Union[int, float]):
        """Projects weights of FFN and MHA components to a lower rank."""
        if hasattr(self.feed_forward, 'project_weights_low_rank'):
            self.feed_forward.project_weights_low_rank(rank_or_fraction)
        if hasattr(self.self_attn, 'project_projection_weights_low_rank'):
            self.self_attn.project_projection_weights_low_rank(rank_or_fraction)
