import math
from typing import List, Optional

import torch
import torch.nn as nn

from src.lib.core.encoder_block import TransformerEncoderBlock
from src.lib.core.positional_encoding import PositionalEncoding


class ConvFeatureExtractor(nn.Module):
    """
    A stack of 1D convolutions to extract features from audio spectrogram's.
    """

    def __init__(self, in_channels: int, d_model: int, dtype=torch.float32):
        super().__init__()

        # We define the layers in a nn.Sequential container for a clean forward pass.
        self.conv_stack = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=d_model, kernel_size=3, padding=1,  # 'same' padding
                      dtype=dtype),
            nn.GELU(),

            # Second convolution: 3x3, stride 2 for downsampling
            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=2, padding=1, dtype=dtype),
            # Second activation
            nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """A simple forward pass through the sequential stack."""
        return self.conv_stack(x)


class AudioEncoder(nn.Module):
    """
    A stateful Audio Encoder using a CNN-Transformer hybrid architecture.
    """

    def __init__(self, num_freq_bins: int, d_model: int, num_layers: int, num_heads: int, ff_hidden_config: List[int],
                 max_audio_len: int, dropout_rate: float = 0.1, dtype=torch.float32):
        super().__init__()

        # CNN Feature Extractor
        # Takes spectrogram input (B, Freq, Time) and outputs (B, d_model, Time/2)
        self.conv_extractor = ConvFeatureExtractor(in_channels=num_freq_bins, d_model=d_model, dtype=dtype)

        # Positional Encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_audio_len,  # Max sequence length after convolution
                                               dtype=dtype)

        # Transformer Encoder Stack
        self.encoder_stack = nn.ModuleList(
            [TransformerEncoderBlock(d_model, num_heads, ff_hidden_config, dropout_rate=dropout_rate, dtype=dtype) for _
             in range(num_layers)])

        # Final Normalization
        self.final_norm = nn.LayerNorm(d_model, dtype=dtype)

        # Apply a good weight initialization scheme
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Applies a standard, robust initialization scheme."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Conv1d):
            torch.nn.init.kaiming_normal_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, spectrogram_input: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Performs a full forward pass through the CNN and Transformer stack.

        Args:
            spectrogram_input (torch.Tensor): Input of shape (B, Freq, Time).
            padding_mask (Optional[torch.Tensor]): Mask for the time dimension after conv.

        Returns:
            torch.Tensor: Final audio context of shape (B, Time_out, d_model).
        """
        #  Through CNN extractor
        # Input: (B, Freq, Time) -> Output: (B, d_model, Time_out)
        x = self.conv_extractor(spectrogram_input)

        # Transpose for Transformer: (B, d_model, Time_out) -> (B, Time_out, d_model)
        x = x.transpose(1, 2)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Pass through Transformer encoder stack
        for block in self.encoder_stack:
            x = block(x, padding_mask=padding_mask)

        # Final normalization
        x = self.final_norm(x)

        return x
