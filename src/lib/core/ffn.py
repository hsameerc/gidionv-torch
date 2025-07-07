import math
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def torch_dynamic_relu(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """The PyTorch version of your dynamic relu."""
    alpha = alpha.view(1, -1)
    return torch.maximum(torch.tensor(0.0, device=x.device, dtype=x.dtype), x) + alpha * torch.minimum(
        torch.tensor(0.0, device=x.device, dtype=x.dtype), x)


def torch_arctan_activation(x: torch.Tensor, alpha: float = 0.01) -> torch.Tensor:
    """The PyTorch version of your leaky arctan."""
    y = torch.arctan(x)
    positive_leak = torch.arctan(torch.tensor(4.0)) + alpha * (x - 4.0)
    negative_leak = torch.arctan(torch.tensor(-4.0)) + alpha * (x + 4.0)
    y = torch.where(x > 4.0, positive_leak, y)
    y = torch.where(x < -4.0, negative_leak, y)
    return y


class DynamicFeedForwardNetwork(nn.Module):
    """
    A stateful PyTorch implementation of the Dynamic ReLU Network (DRN).
    This module encapsulates weights, biases, and custom learnable parameters
    and leverages PyTorch's autograd for automatic differentiation.
    """

    def __init__(self, input_size: int, output_size: int, hidden_layers_config: List[int],
                 hidden_activations: Optional[List[Optional[str]]] = None, output_activation: Optional[str] = None,
                 dropout_rate: float = 0.0, dtype: torch.dtype = torch.float32):
        """
        Initializes the PyTorch module.
        - Replaces manual weight/bias creation with nn.Linear.
        - Replaces manual parameter lists with nn.ModuleList and nn.ParameterList.
        - Sets up dropout as a layer.
        """
        super().__init__()
        self.dtype = dtype

        layer_sizes = [input_size] + hidden_layers_config + [output_size]
        self.num_layers = len(layer_sizes) - 1

        self.layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.dynamic_relu_alphas = nn.ParameterList()

        if hidden_activations is None:
            self.hidden_activations = ['gelu'] * len(hidden_layers_config)
        else:
            if len(hidden_activations) != len(hidden_layers_config):
                raise ValueError("Length of hidden_activations must match hidden_layers_config.")
            self.hidden_activations = hidden_activations
        self.output_activation = output_activation

        for i in range(self.num_layers):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1], dtype=dtype))

            is_output_layer = (i == self.num_layers - 1)
            act_name = self.output_activation if is_output_layer else self.hidden_activations[i]

            if not is_output_layer and act_name == 'dynamic_relu':
                alpha = nn.Parameter(torch.full((layer_sizes[i + 1],), 0.25, dtype=dtype))
                self.dynamic_relu_alphas.append(alpha)
            else:
                self.dynamic_relu_alphas.append(None)

            if not is_output_layer and dropout_rate > 0:
                self.dropout_layers.append(nn.Dropout(p=dropout_rate))
            else:
                self.dropout_layers.append(nn.Identity())

    def _apply_activation(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Helper to apply the correct activation for a layer."""
        is_output_layer = (layer_idx == self.num_layers - 1)
        name = self.output_activation if is_output_layer else self.hidden_activations[layer_idx]

        if name == 'dynamic_relu':
            alpha_param = next((p for i, p in enumerate(self.dynamic_relu_alphas) if p is not None and i == layer_idx),
                               None)
            return torch_dynamic_relu(x, alpha_param)
        elif name in ('gelu', 'gelu_approx'):
            return F.gelu(x)
        elif name == 'silu':
            return F.silu(x)
        elif name == 'relu':
            return F.relu(x)
        elif name == 'sigmoid':
            return torch.sigmoid(x)
        elif name == 'tanh':
            return torch.tanh(x)
        elif name == 'softmax':
            return F.softmax(x, dim=-1)
        elif name == 'arctan':
            return torch_arctan_activation(x)
        elif name is None or name == 'linear':
            return x
        else:
            raise ValueError(f"Unknown activation function: {name}")

    def forward(self, inputs_batch: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass. The logic is the same, but the implementation is cleaner.
        - No need to manage a 'cache'. Autograd does this automatically.
        - `model.train()` or `model.eval()` controls dropout.
        """
        if inputs_batch.numel() == 0 or inputs_batch.shape[1] == 0:
            output_shape = list(inputs_batch.shape)
            return torch.empty(output_shape, dtype=inputs_batch.dtype, device=inputs_batch.device)

        input_shape_orig = inputs_batch.shape
        current_A = inputs_batch.to(self.dtype)
        if current_A.ndim > 2:
            current_A = current_A.reshape(-1, current_A.shape[-1])
        for i in range(self.num_layers):
            # Linear transformation
            Z = self.layers[i](current_A)
            # Activation
            A_activated = self._apply_activation(Z, i)
            # Dropout
            current_A = self.dropout_layers[i](A_activated)
        if len(input_shape_orig) > 2:
            current_A = current_A.reshape(*input_shape_orig[:-1], -1)
        return current_A

    def get_params(self):
        """Replaced by model.parameters() or model.state_dict()"""
        return self.parameters()

    def set_params(self, state_dict):
        """Replaced by model.load_state_dict()"""
        self.load_state_dict(state_dict)

    @torch.no_grad()
    def analyze_weights_svd(self, layer_indices: Optional[List[int]] = None, top_n_singular_values: int = 5,
                            threshold_effective_rank: float = 1e-6):
        """
        Analyze the SVD of the network's weight matrices using torch.linalg.
        """
        print(f"SVD Analysis for PyTorch DRN (dtype: {self.dtype})")
        indices_to_analyze = layer_indices if layer_indices is not None else range(self.num_layers)
        for i in indices_to_analyze:
            if i >= self.num_layers:
                print(f"Layer index {i} out of bounds. Skipping.")
                continue
            layer = self.layers[i]
            original_weights = layer.weight.data
            if original_weights.numel() == 0:
                print(f"Layer {i}: Weight matrix is empty. Skipping.")
                continue
            try:
                _, s, _ = torch.linalg.svd(original_weights.to(torch.float32))
                print(f"Layer {i} (shape {original_weights.shape}):")
                if s.numel() == 0:
                    print("  No singular values found.")
                    continue
                print(
                    f"Singular values (top {min(top_n_singular_values, s.size(0))}): {s[:top_n_singular_values].cpu().numpy()}")
                s_max_val = s[0]
                s_min_val = s[-1]
                print(f"Min/Max singular value: {s_min_val.item():.2e} / {s_max_val.item():.2e}")
                if s_max_val > 0 and s_min_val > 1e-9 * s_max_val:
                    condition_number = s_max_val / s_min_val
                    print(f"Condition number: {condition_number.item():.2e}")
                else:
                    print(f"Condition number: Inf or N/A")
                effective_rank_threshold = threshold_effective_rank * s_max_val
                effective_rank = torch.sum(s > effective_rank_threshold)
                print(
                    f"Effective rank (s > {effective_rank_threshold.item():.1e}): {effective_rank.item()} / {s.size(0)}")
            except Exception as e:
                print(f"SVD failed for layer {i} (shape {original_weights.shape}): {e}")

    @torch.no_grad()
    def project_weights_low_rank(self, rank_or_fraction: Union[int, float], layer_indices: Optional[List[int]] = None):
        """
        Apply low-rank projection to the weights of specified layers using torch.linalg.
        """
        print(f"Projecting weights to low rank (target: {rank_or_fraction})")
        indices_to_project = layer_indices if layer_indices is not None else range(self.num_layers)

        for i in indices_to_project:
            if i >= self.num_layers:
                continue

            # Access the nn.Linear layer directly
            layer = self.layers[i]
            weight_original = layer.weight.data

            if weight_original.ndim != 2 or min(weight_original.shape) == 0:
                continue

            try:
                #  SVD on the tensor
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
                    print(f"Skipping layer {i} due to invalid rank_or_fraction: {rank_or_fraction}")
                    continue

                if k >= current_rank:
                    print(f"Skipping layer {i}: target rank {k} >= current rank {current_rank}")
                    continue

                print(f"Projecting layer {i} weights from rank {current_rank} to {k}.")

                # Truncate and reconstruct
                U_k = U[:, :k]
                S_k = torch.diag(S[:k])  # S is a vector, so we make it a diagonal matrix for matmul
                Vh_k = Vh[:k, :]

                weight_reconstructed = U_k @ S_k @ Vh_k

                # Layer's weight in-place
                layer.weight.data.copy_(weight_reconstructed.to(weight_original.dtype))

            except Exception as e:
                print(f"SVD low-rank projection failed for layer {i} (shape {weight_original.shape}): {e}")
