import math
from typing import Optional, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


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
                attn_mask: Optional[torch.Tensor] = None, is_causal: bool = False,  # is_causal is not used here
                kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[
        torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        batch_size, q_seq_len, _ = query.shape
        k_seq_len = key.shape[1]

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
        masking_value = torch.finfo(scores.dtype).min
        # Apply the mask (if any)
        if is_causal or attn_mask is not None:
            # We need to broadcast the mask to the scores shape (batch, heads, q_len, k_len)
            if is_causal:
                causal_mask = torch.triu(torch.ones((q_seq_len, k_seq_len), device=query.device, dtype=torch.bool),
                                         diagonal=1)
                scores = scores.masked_fill(causal_mask, masking_value)

            if attn_mask is not None:
                mask_expanded = attn_mask.unsqueeze(1).unsqueeze(2)
                scores = scores.masked_fill(mask_expanded == 0, masking_value)

        # Apply softmax to get weights
        attn_weights = F.softmax(scores, dim=-1)

        # Apply dropout to weights
        attn_weights = self.dropout(attn_weights)

        # Get the final output by multiplying weights with V
        attn_output_h = torch.matmul(attn_weights, v_heads)

        attn_output = attn_output_h.transpose(1, 2).contiguous().view(batch_size, q_seq_len, self.d_model)
        final_output = self.out_proj(attn_output)

        return final_output, attn_weights, new_kv_cache

    def _get_projection_layers(self) -> List[nn.Linear]:
        """A helper to return all projection layers in a consistent order."""
        return [self.q_proj, self.k_proj, self.v_proj, self.out_proj]

    @torch.no_grad()
    def analyze_projection_weights_svd(self, top_n: int = 5, threshold: float = 1e-6):
        """
        Performs and prints an SVD analysis of the projection weight matrices.
        """
        print(f"SVD Analysis for PyTorch MultiHeadAttention (dtype: {self.dtype})")
        projection_layers = self._get_projection_layers()
        weight_names = ["q_proj", "k_proj", "v_proj", "out_proj"]
        for name, layer in zip(weight_names, projection_layers):
            weight = layer.weight.data
            try:
                # Perform SVD, ensuring float32 for stability
                _, s, _ = torch.linalg.svd(weight.to(torch.float32))
                print(f"\nProjection {name} (shape {weight.shape}):")
                if s.numel() == 0:
                    print("  No singular values found.")
                    continue
                print(f"Singular values (top {min(top_n, s.size(0))}): {s[:top_n].cpu().numpy()}")
                s_max = s[0].item()
                s_min = s[-1].item()
                print(f"Max/Min singular values: {s_max:.4f} / {s_min:.4f}")
                if s_max > 0 and s_min > 1e-9 * s_max:
                    print(f"Condition number: {s_max / s_min:.2e}")
                else:
                    print("Condition number: Inf or N/A")
                eff_rank_thresh = threshold * s_max
                effective_rank = torch.sum(s > eff_rank_thresh).item()
                print(f"Effective rank (s > {eff_rank_thresh:.1e}): {effective_rank} / {s.size(0)}")

            except Exception as e:
                print(f"SVD failed for projection {name}: {e}")
        print("End SVD Analysis")

    @torch.no_grad()
    def project_projection_weights_low_rank(self, rank_or_fraction: Union[int, float]):
        """
         Applies low-rank projection to the Multy Head Attention's projection weights.
        """
        print(f"Projecting MHA weights to low rank (target: {rank_or_fraction})")

        for layer in self._get_projection_layers():
            weight_original = layer.weight.data

            if weight_original.ndim != 2 or min(weight_original.shape) == 0:
                continue

            try:
                # Perform SVD
                U, S, Vh = torch.linalg.svd(weight_original.to(torch.float32), full_matrices=False)

                if S.numel() == 0:
                    continue

                current_rank = S.size(0)
                if isinstance(rank_or_fraction, int) and rank_or_fraction > 0:
                    k = min(rank_or_fraction, current_rank)
                elif isinstance(rank_or_fraction, float) and 0 < rank_or_fraction <= 1.0:
                    k = int(math.ceil(rank_or_fraction * current_rank))
                    k = max(1, min(k, current_rank))
                else:
                    print(f"Skipping layer due to invalid rank_or_fraction: {rank_or_fraction}")
                    continue

                if k >= current_rank:
                    continue

                print(f"  Projecting layer from rank {current_rank} to {k}.")

                # Truncate and reconstruct the weight matrix
                U_k = U[:, :k]
                S_k = torch.diag(S[:k])
                Vh_k = Vh[:k, :]

                weight_reconstructed = U_k @ S_k @ Vh_k

                # Update the layer's weight in-place, preserving the original dtype
                layer.weight.data.copy_(weight_reconstructed.to(weight_original.dtype))

            except Exception as e:
                print(f"SVD low-rank projection failed for MHA layer (shape {weight_original.shape}): {e}")
