from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class SFU(nn.Module):
    """
       A Smooth Firing Unit (SFU) activation function.

       This function serves as a continuous and differentiable approximation of a neuron's
       firing rate. It is inspired by the Leaky Integrate-and-Fire model, applying
       a smooth, thresholded gating mechanism to its input.

       The activation is defined by the formula:
       f(z; θ, γ) = z * sigmoid(γ * (z - θ))

       The parameters theta (θ, the firing threshold) and gamma (γ, the sensitivity or
       steepness) are learnable, allowing the network to adapt the activation shape
       for each neuron.

       Args:
           num_features (int): The number of features in the input tensor, corresponding
                               to the number of neurons in the layer.
           theta_init (float, optional): The initial value for the firing threshold θ.
                                         Defaults to 0.0.
           gamma_init (float, optional): The initial value for the sensitivity γ.
                                         Defaults to 1.0.
       """

    def __init__(self, num_features: int, theta_init: float = 0.0, gamma_init: float = 1.0):
        super(SFU, self).__init__()
        self.shape = (1, num_features)
        self.theta = nn.Parameter(torch.full(self.shape, theta_init))
        self.gamma = nn.Parameter(torch.full(self.shape, gamma_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the SFU activation function."""
        return x * torch.sigmoid(self.gamma * (x - self.theta))


class SurrogateSpike(Function):
    """
        A surrogate gradient function for the discrete spiking mechanism.

        In the forward pass, this function behaves like a Heaviside step function,
        producing a binary spike (0 or 1). This is biologically plausible but has a
        gradient that is zero almost everywhere, making it incompatible with
        gradient-based learning.

        In the backward pass, this function substitutes the true gradient with a
        "surrogate" gradient, which is a smooth, bell-shaped curve. This provides
        a useful learning signal to the optimizer, allowing the network to learn
        how to produce desirable spike patterns.
    """

    @staticmethod
    def forward(ctx, i):
        """
            Performs the forward pass (a hard threshold).

            Args:
                   ctx: A context object for saving information for the backward pass.
                   i: The membrane potential minus the firing threshold (V - θ).

               Returns:
                   A binary tensor of spikes.
        """
        ctx.save_for_backward(i)
        return (i > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        """
            Performs the backward pass (the surrogate gradient).

            Args:
                   ctx: The context object with saved tensors.
                   grad_output: The gradient from the subsequent layer.

            Returns:
                   The surrogate gradient multiplied by the incoming gradient (chain rule).
        """
        i, = ctx.saved_tensors
        grad_input = grad_output * (1 / (1 + 10 * torch.abs(i))).pow(2)
        return grad_input


spike_fn = SurrogateSpike.apply


class LIFLayer(nn.Module):
    """
       A stateful recurrent cell that models a Leaky Integrate-and-Fire (LIF)
       neuron with a high degree of biological plausibility.

       This layer maintains an internal state consisting of the membrane potential (V),
       a refractory counter (R), and a spike adaptation bias (B). It processes one
       timestep of a sequence at a time.

       Key features include:
       - **Leaky Integration:** Membrane potential decays over time with a learnable
         time constant (tau).
       - **Soft Reset:** After firing, the potential is reset towards a learnable
         resting state with a learnable strength.
       - **Spike-Frequency Adaptation:** The firing threshold dynamically increases
         with recent activity, promoting sparse firing.
       - **Absolute Refractory Period:** A neuron is guaranteed to be silent for a
         fixed number of timesteps after firing.
       - **Dual Output Mode:** Can output rich analog signals (via SFU) for standard
         ANN tasks or discrete binary spikes for SNN tasks.

       Args:
           input_size (int): The number of features in the input at each timestep.
           hidden_size (int): The number of LIF neurons in the layer.
           refractory_steps (int, optional): The duration of the absolute refractory
                                            period in timesteps. Defaults to 3.
       """
    def __init__(self, input_size: int, hidden_size: int, refractory_steps: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.refractory_steps = float(refractory_steps)
        # Core layers (created on default device/dtype)
        self.linear_in = nn.Linear(input_size, hidden_size)
        self.sfu = SFU(hidden_size)
        # Learnable parameters for neuron dynamics
        self.reset_factor_param = nn.Parameter(torch.randn(hidden_size))
        self.firing_threshold = nn.Parameter(torch.full((hidden_size,), 0.5))
        self.adaptation_tau_param = nn.Parameter(torch.randn(hidden_size))
        self.leak_tau_param = nn.Parameter(torch.randn(hidden_size))
        self.V_rest = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x_t, state, return_spike=False):
        """
            Performs a single forward step, processing one timestep of a sequence.

            Args:
                   x_t (Tensor): Input tensor for the current timestep, shape (batch_size, input_size).
                   state (Tuple): A tuple containing the hidden state from the previous timestep
                                  (V_prev, R_prev, B_prev).
                   return_spike (bool, optional): If True, the output is a binary spike tensor.
                                                  If False, the output is a continuous analog
                                                  signal from the SFU. Defaults to False.

            Returns:
                   A tuple containing:
                   - output (Tensor): The output of the layer for the current timestep.
                   - new_state (Tuple): The updated hidden state (V_t, R_t, B_t) for the next timestep.
        """
        # Unpacking the state from the previous timestep
        V_prev, R_prev, B_prev = state

        is_refractory = (R_prev > 0)

        # Soft Reset: Blend between previous potential and resting potential
        reset_factor = torch.sigmoid(self.reset_factor_param)
        reset_value = reset_factor * self.V_rest + (1 - reset_factor) * V_prev

        # Calculate which neurons spiked in the previous step for the reset logic
        S_prev = spike_fn(V_prev - (self.firing_threshold + B_prev))

        # Spike-Frequency Adaptation: Effective threshold increases with recent activity
        V_after_spike = V_prev * (1 - S_prev) + reset_value * S_prev

        # Leaky Integration
        # The potential integrates the new input. This is the core update.
        leak_alpha = torch.exp(-F.softplus(self.leak_tau_param))
        V_t = leak_alpha * V_after_spike.detach() + self.linear_in(x_t)

        # Updating Adaptation State
        # The adaptation state for this step decays and is influenced by the previous spike.
        adaptation_alpha = torch.exp(-F.softplus(self.adaptation_tau_param))
        B_t = adaptation_alpha * B_prev.detach() + S_prev

        # Soft Reset: Blend between previous potential and resting potential
        effective_threshold = self.firing_threshold + B_t

        # A neuron can only fire if it is NOT in a refractory period.
        # Spike Generation for the current timestep
        S_t = (1 - is_refractory.float()) * spike_fn(V_t - effective_threshold)

        # Update Refractory Counter for the next timestep
        R_t = torch.relu(R_prev.detach() - 1) + self.refractory_steps * S_t

        # Determine final output (analog or spike)
        # The analog output is a function of the final potential V_t.
        y_t_analog = self.sfu(V_t)

        # The output is silenced if the neuron is in a refractory period.
        output = (1 - is_refractory.float()) * (S_t if return_spike else y_t_analog)

        # Returning the output and the new state for the next timestep
        return output, (V_t, R_t, B_t)

    def init_state(self, batch_size: int, device: torch.device):
        """Initializes V, R, and B states to zero."""
        dtype = self.V_rest.dtype
        V0 = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        R0 = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        B0 = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        return V0, R0, B0


class LIFRnn(nn.Module):
    """
       A complete Recurrent Neural Network built by stacking one or more LIFLayers.

       This module handles the processing of entire sequences of data by iterating
       through the time dimension and managing the hidden states of its internal layers.
       It is designed to be device-agnostic; create the model first, then move it to
       the desired device using `.to(device)`.

       Args:
           input_size (int): The number of features in the input sequence.
           output_size (int): The number of features in the final output.
           hidden_layers_config (List[int]): A list of integers defining the number
                                              of neurons in each hidden LIFLayer.
           refractory_steps (int, optional): The refractory period for all layers.
                                            Defaults to 3.
       """

    def __init__(self, input_size: int, output_size: int, hidden_layers_config: List[int], **kwargs):
        super().__init__()
        self.lif_layers = nn.ModuleList()
        layer_input_size = input_size
        for hidden_size in hidden_layers_config:
            self.lif_layers.append(LIFLayer(layer_input_size, hidden_size, **kwargs))
            layer_input_size = hidden_size
        last_hidden_size = hidden_layers_config[-1]
        self.fc_out = nn.Linear(last_hidden_size, output_size)

    def forward(self, x: torch.Tensor, return_spike: bool = False) -> torch.Tensor:
        """
            Processes a batch of sequences through the stacked LIFLayers.

            Args:
                   x (Tensor): The input tensor of shape (batch_size, sequence_length, input_size).
                   return_spike (bool, optional): If True, all layers will output binary spikes.
                                                  Defaults to False.

            Returns:
                   The output tensor from the final layer at the final timestep, after the
                   output projection. Shape: (batch_size, output_size).
        """
        batch_size, sequence_length, _ = x.shape
        device = x.device
        states = [layer.init_state(batch_size, device) for layer in self.lif_layers]
        outputs_over_time = []
        for t in range(sequence_length):
            x_t = x[:, t, :]
            for i, layer in enumerate(self.lif_layers):
                x_t, new_state = layer(x_t, states[i], return_spike)
                states[i] = new_state
            outputs_over_time.append(x_t)
        output_sequence = torch.stack(outputs_over_time, dim=1)
        return self.fc_out(output_sequence)


class DualStateLIFLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear_in_v = nn.Linear(input_size, hidden_size)
        self.leak_tau_v = nn.Parameter(torch.randn(hidden_size))
        self.flip_threshold = nn.Parameter(torch.full((hidden_size,), 0.5))
        self.fc_out = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.output_activation = nn.Tanh()

    def forward(self, x_t: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        V_prev, D_prev = (s.detach() for s in state)
        leak_alpha = torch.exp(-F.softplus(self.leak_tau_v))
        input_current = self.linear_in_v(x_t)
        V_t = leak_alpha * V_prev + input_current
        spike = spike_fn(V_t - self.flip_threshold)
        D_t = D_prev * (1 - spike) + (1 - D_prev) * spike
        combined_state = torch.cat([V_t, D_t], dim=1)
        output = self.output_activation(self.fc_out(combined_state))
        return output, (V_t, D_t)

    def init_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        dtype = self.flip_threshold.dtype
        V0 = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        D0 = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        return V0, D0


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