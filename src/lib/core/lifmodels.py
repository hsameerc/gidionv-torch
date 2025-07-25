from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class SFU(nn.Module):
    """
    A Smooth Firing Unit (SFU) activation function.

    This function serves as a continuous and differentiable approximation of a
    neuron's firing rate, inspired by the Leaky Integrate-and-Fire model. It
    applies a smooth, thresholded gating mechanism to its input.

    f(z; θ, γ) = z * sigmoid(γ * (z - θ))

    The parameters theta (θ, firing threshold) and gamma (γ, sensitivity) are learnable.

    Args:
        num_features (int): The number of features in the input tensor.
        theta_init (float): The initial value for the firing threshold θ.
        gamma_init (float): The initial value for the sensitivity γ.
    """

    def __init__(self, num_features: int, theta_init: float = 0.0, gamma_init: float = 1.0):
        super(SFU, self).__init__()
        self.theta = nn.Parameter(torch.full((1, num_features), theta_init))
        self.gamma = nn.Parameter(torch.full((1, num_features), gamma_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the SFU activation function."""
        return x * torch.sigmoid(self.gamma * (x - self.theta))


class SurrogateSpike(Function):
    """
    A surrogate gradient function for the discrete spiking mechanism.

    Forward Pass: A Heaviside step function, producing a binary spike (0 or 1).
    Backward Pass: A smooth, bell-shaped surrogate gradient is substituted to
                   enable gradient-based learning.
    """

    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input_tensor)
        return (input_tensor > 0).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        input_tensor, = ctx.saved_tensors
        # Surrogate is the derivative of a fast sigmoid function.
        surrogate_grad = (1 / (1 + 10 * torch.abs(input_tensor))).pow(2)
        return grad_output * surrogate_grad


spike_fn = SurrogateSpike.apply


class DualStateLIFLayer(nn.Module):
    """
    A recurrent cell with a dual-state memory system, designed for tasks
    requiring discrete, logical memory (e.g., parity checking).

    It maintains:
    1. An Analog State (V): A continuous, leaky integrator for signal processing.
    2. A Digital "Mirror" State (D): A binary state for tracking discrete events.

    Args:
        input_size (int): The number of features in the input at each timestep.
        hidden_size (int): The number of neurons in the layer.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # Components for the Analog State (V)
        self.linear_in_v = nn.Linear(input_size, hidden_size)
        self.leak_tau_v = nn.Parameter(torch.randn(hidden_size))
        self.flip_threshold = nn.Parameter(torch.full((hidden_size,), 0.5))

        # Components for the final combined output
        self.fc_out = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.output_activation = nn.Tanh()

    def forward(self, x_t: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Performs a single forward step. State is (V_prev, D_prev).
        """
        V_prev, D_prev = (s.detach() for s in state)

        # Analog State (V) Update (The value that remains)
        # Simplified leaky integrator.
        leak_alpha = torch.exp(-F.softplus(self.leak_tau_v))
        V_t = leak_alpha * V_prev + self.linear_in_v(x_t)

        # "Flip Check" based on the analog state
        # We check if the analog state has crossed its threshold
        # This produces a binary spike signal (0.0 or 1.0)
        spike = spike_fn(V_t - self.flip_threshold)

        # Digital "Mirror" State (D) Update
        # The digital state updates based on the spike and its own previous state.
        # D_t = D_prev XOR spike
        D_t = D_prev * (1 - spike) + (1 - D_prev) * spike

        # Final Output is a function of both states
        # The output is a learned fn of both the analog and digital states.
        combined_state = torch.cat([V_t, D_t], dim=1)
        output = self.output_activation(self.fc_out(combined_state))

        return output, (V_t, D_t)

    def init_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initializes both the analog (V) and digital (D) states to zero."""
        dtype = self.flip_threshold.dtype
        V0 = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        D0 = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        return (V0, D0)


class DualStateRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1):
        super().__init__()
        # For a fair test, we can stack the layers.
        self.rnn_cells = nn.ModuleList()
        layer_input_size = input_size
        for _ in range(num_layers):
            self.rnn_cells.append(DualStateLIFLayer(layer_input_size, hidden_size))
            layer_input_size = hidden_size

        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = x.shape
        device = x.device

        states = [cell.init_state(batch_size, device) for cell in self.rnn_cells]
        x_t = None
        for t in range(sequence_length):
            x_t = x[:, t, :]
            for i, cell in enumerate(self.rnn_cells):
                x_t, new_state = cell(x_t, states[i])
                states[i] = new_state

        return self.fc_out(x_t)


class DualStateFFN(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_layers_config: List[int], dropout_rate: float = 0.1,
                 dtype: torch.dtype = torch.float32, ):
        super().__init__()
        self.lif_layers = nn.ModuleList()
        layer_input_size = input_size
        for hidden_size in hidden_layers_config:
            self.lif_layers.append(DualStateLIFLayer(layer_input_size, hidden_size, ))
            layer_input_size = hidden_size
        last_hidden_size = hidden_layers_config[-1]
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc_out = nn.Linear(last_hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = x.shape
        device = x.device

        states = [cell.init_state(batch_size, device) for cell in self.lif_layers]

        outputs_over_time = []
        for t in range(sequence_length):
            x_t = x[:, t, :]
            for i, cell in enumerate(self.lif_layers):
                x_t, new_state = cell(x_t, states[i])
                states[i] = new_state
                outputs_over_time.append(x_t)
        output_sequence = torch.stack(outputs_over_time, dim=1)
        return self.fc_out(output_sequence)
