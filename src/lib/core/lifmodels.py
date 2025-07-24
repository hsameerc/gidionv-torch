import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import List, Tuple


# ==============================================================================
# SECTION 1: CORE COMPONENTS
# ==============================================================================

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


# ==============================================================================
# SECTION 2: RECURRENT CELL DEFINITIONS
# ==============================================================================

class LIFLayer(nn.Module):
    """
    A stateful recurrent cell that models a Leaky Integrate-and-Fire (LIF)
    neuron with a high degree of biological plausibility. Designed for processing
    natural, signal-like data.

    Features:
    - Leaky Integration with learnable time constant (tau).
    - Soft Reset with learnable strength and resting potential.
    - Spike-Frequency Adaptation to promote sparse firing.
    - Absolute Refractory Period to enforce post-spike silence.

    Args:
        input_size (int): The number of features in the input at each timestep.
        hidden_size (int): The number of LIF neurons in the layer.
        refractory_steps (int): The duration of the refractory period.
    """

    def __init__(self, input_size: int, hidden_size: int, refractory_steps: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.refractory_steps = int(refractory_steps)

        # Layers and parameters are device-agnostic by default
        self.linear_in = nn.Linear(input_size, hidden_size)
        self.sfu = SFU(hidden_size)
        self.reset_factor_param = nn.Parameter(torch.randn(hidden_size))
        self.firing_threshold = nn.Parameter(torch.full((hidden_size,), 0.5))
        self.adaptation_tau_param = nn.Parameter(torch.randn(hidden_size))
        self.leak_tau_param = nn.Parameter(torch.randn(hidden_size))
        self.V_rest = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x_t: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Performs a single forward step. State is (V_prev, R_prev, B_prev).
        """
        V_prev, R_prev, B_prev = (s.detach() for s in state)

        is_refractory = (R_prev > 0)

        # Calculate previous spike event for the reset & adaptation logic
        S_prev = (1 - is_refractory.float()) * spike_fn(V_prev - (self.firing_threshold + B_prev))

        # Soft Reset
        reset_factor = torch.sigmoid(self.reset_factor_param)
        reset_value = reset_factor * self.V_rest + (1 - reset_factor) * V_prev
        V_after_spike = V_prev * (1 - S_prev) + reset_value * S_prev

        # Leaky Integration
        leak_alpha = torch.exp(-F.softplus(self.leak_tau_param))
        V_t = leak_alpha * V_after_spike + self.linear_in(x_t)

        # Adaptation State Update for the next step
        adaptation_alpha = torch.exp(-F.softplus(self.adaptation_tau_param))
        B_next = adaptation_alpha * B_prev + S_prev

        # Spike Generation for the current timestep
        effective_threshold = self.firing_threshold + B_next
        S_t = (1 - is_refractory.float()) * spike_fn(V_t - effective_threshold)

        # Refractory Counter Update for the next step
        R_next = torch.relu(R_prev - 1) + self.refractory_steps * S_t

        # The analog output is a function of the final potential V_t
        output = (1 - is_refractory.float()) * self.sfu(V_t)

        return output, (V_t, R_next, B_next)

    def init_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initializes the (V, R, B) hidden states to zero."""
        dtype = self.V_rest.dtype
        V0 = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        R0 = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        B0 = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        return V0, R0, B0


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

        # 1. Analog State (V) Update
        leak_alpha = torch.exp(-F.softplus(self.leak_tau_v))
        V_t = leak_alpha * V_prev + self.linear_in_v(x_t)

        # 2. "Flip Check" based on the analog state
        spike = spike_fn(V_t - self.flip_threshold)

        # 3. Digital "Mirror" State (D) Update
        D_t = D_prev * (1 - spike) + (1 - D_prev) * spike

        # 4. Final Output is a function of both states
        combined_state = torch.cat([V_t, D_t], dim=1)
        output = self.output_activation(self.fc_out(combined_state))

        return output, (V_t, D_t)

    def init_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initializes both the analog (V) and digital (D) states to zero."""
        dtype = self.flip_threshold.dtype
        V0 = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        D0 = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        return (V0, D0)



class LIFRnn(nn.Module):
    """
    A full Recurrent Neural Network built by stacking one or more LIFLayers.
    Designed for natural signal processing.

    Args:
        input_size (int): The number of features in the input sequence.
        output_size (int): The number of features in the final output.
        hidden_layers_config (List[int]): A list of hidden sizes for each layer.
        **kwargs: Additional keyword arguments to pass to the LIFLayer constructor.
    """

    def __init__(self, input_size: int, output_size: int, hidden_layers_config: List[int], **kwargs):
        super().__init__()
        self.lif_layers = nn.ModuleList()
        layer_input_size = input_size
        for hidden_size in hidden_layers_config:
            self.lif_layers.append(LIFLayer(layer_input_size, hidden_size, **kwargs))
            layer_input_size = hidden_size
        self.fc_out = nn.Linear(hidden_layers_config[-1], output_size)

    def forward(self, x: torch.Tensor, return_full_sequence: bool = False) -> torch.Tensor:
        """
        Processes a batch of sequences.

        Args:
            x (Tensor): Input of shape (batch_size, sequence_length, input_size).
            return_full_sequence (bool): If True, returns the output for all timesteps.
                                        If False, returns only the output of the final
                                        timestep. Defaults to False.

        Returns:
            Tensor: The processed sequence or final timestep output.
        """
        batch_size, sequence_length, _ = x.shape
        device = x.device
        states = [layer.init_state(batch_size, device) for layer in self.lif_layers]
        outputs_over_time = []

        for t in range(sequence_length):
            x_t = x[:, t, :]
            for i, layer in enumerate(self.lif_layers):
                x_t, new_state = layer(x_t, states[i])
                states[i] = new_state
            outputs_over_time.append(x_t)

        output_sequence = torch.stack(outputs_over_time, dim=1)
        projected_sequence = self.fc_out(output_sequence)

        return projected_sequence if return_full_sequence else projected_sequence[:, -1, :]


class DualStateRNN(nn.Module):
    """
    A full Recurrent Neural Network built from stacked DualStateLIFLayers.
    Designed for tasks requiring discrete, algorithmic reasoning.

    Args:
        input_size (int): The number of features in the input sequence.
        output_size (int): The number of features in the final output.
        hidden_layers_config (List[int]): A list of hidden sizes for each layer.
    """

    def __init__(self, input_size: int, output_size: int, hidden_layers_config: List[int]):
        super().__init__()
        self.rnn_cells = nn.ModuleList()
        layer_input_size = input_size
        for hidden_size in hidden_layers_config:
            self.rnn_cells.append(DualStateLIFLayer(layer_input_size, hidden_size))
            layer_input_size = hidden_size

        self.fc_out = nn.Linear(hidden_layers_config[-1], output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Processes a sequence and returns the output of the final timestep."""
        batch_size, sequence_length, _ = x.shape
        device = x.device
        states = [cell.init_state(batch_size, device) for cell in self.rnn_cells]
        output = None
        outputs_over_time = []
        for t in range(sequence_length):
            output = x[:, t, :]
            for i, cell in enumerate(self.rnn_cells):
                output, new_state = cell(output, states[i])
                states[i] = new_state
            outputs_over_time.append(output)
        output_sequence = torch.stack(outputs_over_time, dim=1)
        projected_sequence = self.fc_out(output_sequence)

        return projected_sequence