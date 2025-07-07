from typing import List, Optional

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
