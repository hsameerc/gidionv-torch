from typing import Optional

import torch
import torch.nn as nn

from src.lib.core.lif_rnn import LIF_RNN


class LIFMemoryEncoder(nn.Module):
    """
    An encoder that processes sequences token-by-token using a stateful LIF_RNN.
    It's designed to create a memory context for the main transformer decoder.
    """

    def __init__(self, d_model: int, num_layers: int, hidden_size: int, **kwargs):
        super().__init__()

        self.lif_rnn = LIF_RNN(
            input_size=d_model,
            output_size=d_model,
            hidden_layers_config=[hidden_size] * num_layers,
            **kwargs
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Processes the input sequence sequentially.

        Args:
            x (torch.Tensor): Input embeddings of shape (B, S, D).
            padding_mask (Optional[torch.Tensor]): Not used by our LIF RNN,
                                                   but could be used to get sequence lengths.

        Returns:
            torch.Tensor: The sequence of hidden states, shape (B, S, D).
        """
        all_hidden_states = self.lif_rnn(x, return_all_hidden_states=True)
        return self.final_norm(all_hidden_states)
