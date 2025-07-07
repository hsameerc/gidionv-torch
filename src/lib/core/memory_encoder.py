from typing import List, Optional, Union

import torch
import torch.nn as nn

from src.lib.core.encoder_block import TransformerEncoderBlock


class MemoryEncoder(nn.Module):
    """
    A stateful PyTorch implementation of a stack of Transformer Encoder Blocks.

    This module processes an input sequence through multiple TransformerEncoderBlock
    layers to produce a contextualized memory representation.
    """

    def __init__(self, num_layers: int, d_model: int, num_heads: int, ff_hidden_config: List[int],
                 dropout_rate: float = 0.1, dtype: torch.dtype = torch.float32):
        """
        Initializes the MemoryEncoder and its stack of encoder blocks.
        """
        super().__init__()

        # Dropout for the input embeddings
        self.input_dropout = nn.Dropout(dropout_rate)

        # The stack of encoder blocks
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model=d_model, num_heads=num_heads, ff_hidden_config=ff_hidden_config,
                                    dropout_rate=dropout_rate, dtype=dtype) for _ in range(num_layers)])

        self.final_norm = nn.LayerNorm(d_model, dtype=dtype)

    def forward(self, embedded_vectors: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Performs a forward pass through the entire stack of encoder blocks.
        - No need to return a `cache` for backprop.
        """
        # Applying dropout to the input
        x = self.input_dropout(embedded_vectors)

        # Passing through the stack of encoder blocks
        for block in self.encoder_blocks:
            x = block(x, padding_mask=padding_mask)

        # Applying final normalization
        x = self.final_norm(x)
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
        """Analyzes weights of all subcomponents using SVD."""
        print(f"--- SVD Analysis for PyTorch MemoryEncoder ---")
        for i, block in enumerate(self.encoder_blocks):
            print(f"\n--- Analyzing Encoder Block {i} ---")
            if hasattr(block, 'analyze_projection_weights_svd'):
                block.analyze_projection_weights_svd(top_n=top_n)
        print(f"--- End SVD Analysis ---")

    def project_component_weights_low_rank(self, rank_or_fraction: Union[int, float]):
        """Projects weights of all subcomponents to a lower rank."""
        print(f"--- Projecting MemoryEncoder Weights ---")
        for i, block in enumerate(self.encoder_blocks):
            print(f"\n--- Projecting Encoder Block {i} ---")
            if hasattr(block, 'project_component_weights_low_rank'):
                block.project_component_weights_low_rank(rank_or_fraction)
