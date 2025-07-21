from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class SFU(nn.Module):
    def __init__(self, num_features: int, theta_init: float = 0.0, gamma_init: float = 1.0):
        super(SFU, self).__init__()
        self.shape = (1, num_features)
        self.theta = nn.Parameter(torch.full(self.shape, theta_init))
        self.gamma = nn.Parameter(torch.full(self.shape, gamma_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.gamma * (x - self.theta))


class SurrogateSpike(Function):
    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i)
        return (i > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        i, = ctx.saved_tensors
        grad_input = grad_output * (1 / (1 + 10 * torch.abs(i))).pow(2)
        return grad_input


spike_fn = SurrogateSpike.apply


class LIFLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, refractory_steps: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.refractory_steps = float(refractory_steps)
        self.linear_in = nn.Linear(input_size, hidden_size)
        self.sfu = SFU(hidden_size)
        self.reset_factor_param = nn.Parameter(torch.randn(hidden_size))
        self.firing_threshold = nn.Parameter(torch.full((hidden_size,), 0.5))
        self.adaptation_tau_param = nn.Parameter(torch.randn(hidden_size))
        self.leak_tau_param = nn.Parameter(torch.randn(hidden_size))
        self.V_rest = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x_t, state, return_spike=False):
        # Unpacking the state from the previous timestep
        V_prev, R_prev, B_prev = state
        # Applying Refractory State
        is_refractory = (R_prev > 0)
        # Soft Resetting Mechanism
        # First, determine the reset value based on the previous state
        reset_factor = torch.sigmoid(self.reset_factor_param)
        reset_value = reset_factor * self.V_rest + (1 - reset_factor) * V_prev
        # Now, determining which neurons actually fired in the last step to decide if they need a reset
        # We need to re-calculate the previous spike event for the reset logic
        S_prev = spike_fn(V_prev - (self.firing_threshold + B_prev))
        # Applying the reset only to neurons that spiked
        V_after_spike = V_prev * (1 - S_prev) + reset_value * S_prev
        # Leaky Integration
        # The potential integrates the new input. This is the core update.
        leak_alpha = torch.exp(-F.softplus(self.leak_tau_param))
        V_t = leak_alpha * V_after_spike.detach() + self.linear_in(x_t)
        # Updating Adaptation State
        # The adaptation state for this step decays and is influenced by the previous spike.
        adaptation_alpha = torch.exp(-F.softplus(self.adaptation_tau_param))
        B_t = adaptation_alpha * B_prev.detach() + S_prev
        # Firing (Spike Generation for *this* timestep)
        # The firing decision is based on the *new* potential V_t and the *new* adaptation state B_t
        effective_threshold = self.firing_threshold + B_t
        # A neuron can only fire if it is NOT in a refractory period.
        S_t = (1 - is_refractory.float()) * spike_fn(V_t - effective_threshold)
        # Updating Refractory Counter
        # The counter for the *next* step is based on spikes from *this* step.
        R_t = torch.relu(R_prev.detach() - 1) + self.refractory_steps * S_t
        # Calculating Output
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
    A recurrent network using the highly advanced LIF_Layer.
    This version is corrected to return the full sequence of hidden states.
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
        Processes a batch of sequences and returns the FULL sequence of outputs.
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
