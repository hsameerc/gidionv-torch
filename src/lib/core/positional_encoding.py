import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    A stateful PyTorch implementation of the Positional Encoding layer.

    This module pre-computes the positional encoding matrix and registers it
    as a `buffer`. This ensures it is part of the model's state (and moves
    to the correct device) but is not considered a learnable parameter.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dtype: torch.dtype = torch.float32):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model must be an even number.")
        position = torch.arange(max_len, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=dtype) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.register_buffer('div_term', div_term)
        self.d_model = d_model

    def _dynamic_get_pe(self, start_pos: int, seq_len: int) -> torch.Tensor:
        """Dynamically generates PE for sequences longer than the pre-computed buffer."""
        position = torch.arange(start_pos, start_pos + seq_len, dtype=self.div_term.dtype,
                                device=self.div_term.device).unsqueeze(1)
        pe = torch.zeros(seq_len, self.d_model, dtype=self.div_term.dtype, device=self.div_term.device)
        pe[:, 0::2] = torch.sin(position * self.div_term)
        pe[:, 1::2] = torch.cos(position * self.div_term)
        return pe

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """
        Adds positional encoding to the input tensor.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            start_pos (int): The starting position offset, used for KV caching in generation.

        Returns:
            torch.Tensor: The input tensor with positional encoding added.
        """
        seq_len = x.size(1)
        end_pos = start_pos + seq_len
        if end_pos <= self.pe.size(0):
            pe_slice = self.pe[start_pos:end_pos, :]
        else:
            pe_slice = self._dynamic_get_pe(start_pos, seq_len)
        return x + pe_slice
