from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def torch_dynamic_relu(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """The PyTorch version of your dynamic relu."""
    alpha = alpha.view(1, -1, 1, 1) if x.ndim == 4 else alpha.view(1, -1)
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


class SFU(nn.Module):
    """
    Smooth Firing Unit (SFU) activation function.
    This activation is inspired by the Leaky Integrate-and-Fire neuron model,
    applying a smooth, thresholded gating mechanism to the input.

    Formula: f(x; θ, γ) = x * sigmoid(γ * (x - θ))
    Using the product rule, the derivative is: f'(z) = σ(γ * (z - θ)) + z * γ * σ(γ * (z - θ)) * (1 - σ(γ * (z - θ)))
    The parameters theta (θ) and gamma (γ) are learnable.

    Theoretical Edge Case
    Is it impossible for SFU to have a gradient problem?
    There is one theoretical scenario. Because gamma is learnable, if the network decided to learn a gamma value of exactly 0, the function would become:
    f(z) = z * σ(0 * (z - θ)) = z * 0.5
    """

    def __init__(self, num_features: int, theta_init: float = 0.0, gamma_init: float = 1.0):
        super(SFU, self).__init__()
        self.shape = (1, num_features)
        self.theta = nn.Parameter(torch.full(self.shape, theta_init))
        self.gamma = nn.Parameter(torch.full(self.shape, gamma_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the SFU activation function."""
        gating_signal = torch.sigmoid(self.gamma * (x - self.theta))
        return x * gating_signal

    def __repr__(self):
        return f'SFU(num_features={self.shape[1]})'



class DynamicFeedForwardNetwork(nn.Module):
    """
    A stateful implementation of the Dynamic FeedForward Network.
    """

    def __init__(self, input_size: int, output_size: int, hidden_layers_config: List[int],
                 hidden_activations: Optional[List[Optional[str]]] = None, output_activation: Optional[str] = None,
                 dropout_rate: float = 0.0, dtype: torch.dtype = torch.float32):
        """
        - Uses nn.ModuleList to properly register learnable activation layers.
        """
        super().__init__()
        self.dtype = dtype
        layer_sizes = [input_size] + hidden_layers_config + [output_size]
        self.num_layers = len(layer_sizes) - 1
        self.layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.dynamic_relu_alphas = nn.ParameterList()
        self.sfu_layers = nn.ModuleList()

        if hidden_activations is None:
            self.hidden_activations = ['sfu'] * len(hidden_layers_config)
        else:
            if len(hidden_activations) != len(hidden_layers_config):
                raise ValueError("Length of hidden_activations must match hidden_layers_config.")
            self.hidden_activations = hidden_activations
        self.output_activation = output_activation

        for i in range(self.num_layers):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1], dtype=dtype))

            is_output_layer = (i == self.num_layers - 1)
            act_name = self.output_activation if is_output_layer else self.hidden_activations[i]

            if not is_output_layer and act_name == 'sfu':
                self.sfu_layers.append(SFU(layer_sizes[i + 1]))
                self.dynamic_relu_alphas.append(None)
            elif not is_output_layer and act_name == 'dynamic_relu':
                alpha = nn.Parameter(torch.full((layer_sizes[i + 1],), 0.25, dtype=dtype))
                self.dynamic_relu_alphas.append(alpha)
                self.sfu_layers.append(None)
            else:
                self.dynamic_relu_alphas.append(None)
                self.sfu_layers.append(None)

            if not is_output_layer and dropout_rate > 0:
                self.dropout_layers.append(nn.Dropout(p=dropout_rate))
            else:
                self.dropout_layers.append(nn.Identity())

    def _apply_activation(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Helper to apply the correct activation for a layer."""
        is_output_layer = (layer_idx == self.num_layers - 1)
        name = self.output_activation if is_output_layer else self.hidden_activations[layer_idx]
        if name == 'sfu':
            # Retrieve and apply the SFU layer for this index
            sfu_layer = self.sfu_layers[layer_idx]
            return sfu_layer(x)
        elif name == 'dynamic_relu':
            alpha_param = self.dynamic_relu_alphas[layer_idx]
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
         Performs the forward pass.
         """
        current_A = inputs_batch.to(self.dtype)
        for i in range(self.num_layers):
            Z = self.layers[i](current_A)
            A_activated = self._apply_activation(Z, i)
            current_A = self.dropout_layers[i](A_activated)
        return current_A

    def get_params(self):
        return self.parameters()

    def set_params(self, state_dict):
        self.load_state_dict(state_dict)