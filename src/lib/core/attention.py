import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

class MultiHeadAttention(nn.Module):
    """
    A stateful PyTorch implementation of Multi-Head Attention, implemented
    manually to allow access to the attention weights matrix.
    This is an alternative to the fused F.scaled_dot_product_attention.
    """

    def __init__(self, d_model: int, num_heads: int, dropout_rate: float = 0.1, dtype: torch.dtype = torch.float32):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_depth = d_model // num_heads
        self.dtype = dtype

        self.q_proj = nn.Linear(d_model, d_model, dtype=dtype)
        self.k_proj = nn.Linear(d_model, d_model, dtype=dtype)
        self.v_proj = nn.Linear(d_model, d_model, dtype=dtype)
        self.out_proj = nn.Linear(d_model, d_model, dtype=dtype)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None, is_causal: bool = False, # is_causal is not used here
                kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[
        torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        batch_size, q_seq_len, _ = query.shape

        q_proj = self.q_proj(query)
        k_proj = self.k_proj(key)
        v_proj = self.v_proj(value)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k_proj = torch.cat([past_k, k_proj], dim=1)
            v_proj = torch.cat([past_v, v_proj], dim=1)
        new_kv_cache = (k_proj, v_proj)

        q_heads = q_proj.view(batch_size, q_seq_len, self.num_heads, self.head_depth).transpose(1, 2)
        k_heads = k_proj.view(batch_size, -1, self.num_heads, self.head_depth).transpose(1, 2)
        v_heads = v_proj.view(batch_size, -1, self.num_heads, self.head_depth).transpose(1, 2)

        # Calculate scores: Q @ K.T
        scores = torch.matmul(q_heads, k_heads.transpose(-2, -1))

        # Scale the scores
        scores = scores / math.sqrt(self.head_depth)

        # Apply the mask (if any)
        if attn_mask is not None:
            # We need to broadcast the mask to the scores shape (batch, heads, q_len, k_len)
            mask_expanded = attn_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask_expanded == 0, -1e9) # Use a large negative number

        # Apply softmax to get weights
        attn_weights = F.softmax(scores, dim=-1)

        # Apply dropout to weights
        attn_weights = self.dropout(attn_weights)

        # Get the final output by multiplying weights with V
        attn_output_h = torch.matmul(attn_weights, v_heads)

        attn_output = attn_output_h.transpose(1, 2).contiguous().view(batch_size, q_seq_len, self.d_model)
        final_output = self.out_proj(attn_output)

        return final_output, attn_weights, new_kv_cache