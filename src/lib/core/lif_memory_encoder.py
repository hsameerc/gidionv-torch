from typing import Optional

import torch
import torch.nn as nn

from src.lib.core.lif_rnn import LIFRnn
from src.lib.core.lifmodels import DualStateRNN


class LIFMemoryEncoder(nn.Module):
    """
    A memory encoder that uses a stack of pure LIFRnn layers to process
    sequences of arbitrary length.
    """

    def __init__(self, num_layers: int, d_model: int, hidden_size: int,
                 dropout_rate: float = 0.1, dtype: torch.dtype = torch.float32):
        super().__init__()

        self.input_dropout = nn.Dropout(dropout_rate)
        hidden_layers_config = hidden_size if isinstance(hidden_size, list) else [hidden_size] * num_layers

        # The core of the encoder is the LIFRnn itself
        self.lif_rnn = DualStateRNN(
            input_size=d_model,
            output_size=d_model,
            hidden_layers_config=hidden_layers_config,
        )

        self.final_norm = nn.LayerNorm(d_model, dtype=dtype)

    def forward(self, embedded_vectors: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Performs a forward pass through the LIFRnn.
        `padding_mask` is not strictly needed for a simple RNN's computation,
        but it's good practice to keep for compatibility. The RNN will process
        the padding tokens, but their contribution can be ignored later.
        """
        # Applying dropout to the input
        x = self.input_dropout(embedded_vectors)

        # Passing through the entire stack of LIFRnn layers
        # The LIFRnn's forward pass returns the output sequence of the final layer
        # which will have the shape (batch_size, sequence_length, d_model)
        x = self.lif_rnn(x)

        # Applying final normalization
        x = self.final_norm(x)
        return x
