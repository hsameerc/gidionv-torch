from typing import List

import torch
from torch import nn

from src.lib.core.lif_rnn import LIFLayer


class LIFFfn(nn.Module):
    """
       A complete  Neural Network built by stacking one or more LIFLayers.

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

    def __init__(self, input_size: int, output_size: int, hidden_layers_config: List[int], dropout_rate: float = 0.1, dtype:torch.dtype = torch.float32,
                 **kwargs):
        super().__init__()
        self.lif_layers = nn.ModuleList()
        layer_input_size = input_size
        for hidden_size in hidden_layers_config:
            self.lif_layers.append(LIFLayer(layer_input_size, hidden_size, **kwargs))
            layer_input_size = hidden_size
        last_hidden_size = hidden_layers_config[-1]
        self.dropout = nn.Dropout(p=dropout_rate)
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
                if i < len(self.lif_layers) - 1:
                    x_t = self.dropout(x_t)
            outputs_over_time.append(x_t)

        # Stack all hidden states from the final layer over time
        output_sequence = torch.stack(outputs_over_time, dim=1)

        # Project the entire sequence
        projected_sequence = self.fc_out(output_sequence)
        return projected_sequence[:, -1, :]
