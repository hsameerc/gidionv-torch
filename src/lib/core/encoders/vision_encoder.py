import math
from typing import List, Optional, Any

import torch
import torch.nn as nn

from src.lib.core.encoder_block import TransformerEncoderBlock
from src.lib.core.positional_encoding import PositionalEncoding


class PatchEmbedding(nn.Module):
    """
    [PyTorch] Converts an image into a sequence of flattened patches.
    This layer uses nn.Conv2d to process patches efficiently.
    """

    def __init__(self, image_size: int, patch_size: int, in_channels: int, d_model: int, dropout_rate: float = 0.0,
                 dtype=torch.float32):
        super().__init__()

        if image_size % patch_size != 0:
            raise ValueError("Image size must be divisible by patch size.")

        self.patch_size = patch_size
        self.d_model = d_model
        self.num_patches_h = image_size // patch_size
        self.num_patches_w = image_size // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

        # Using nn.Conv2d for patch embedding.
        # kernel_size=patch_size, stride=patch_size means each patch is processed by one kernel application.
        # The output channels is d_model, so each patch embedding will be d_model features.
        self.conv_embed = nn.Conv2d(in_channels=in_channels, out_channels=d_model, kernel_size=patch_size,
                                    stride=patch_size, padding=0,
                                    # No padding needed if image_size is divisible by patch_size
                                    dtype=dtype)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, image_input: torch.Tensor) -> torch.Tensor:
        """
        Input shape: (B, C, H, W)
        Output shape: (B, num_patches, d_model)
        """
        # Applying convolution
        # Output of Conv2d: (B, d_model, num_patches_h, num_patches_w)
        x = self.conv_embed(image_input)

        # Flattening the spatial dimensions into a sequence of patches
        # (B, d_model, num_patches_h, num_patches_w) -> (B, num_patches, d_model)
        x = x.flatten(2).transpose(1, 2)  # Flatten H*W into seq_len, then transpose to (B, seq_len, d_model)

        # Applying dropout
        x = self.dropout(x)

        return x


class VisionEncoder(nn.Module):
    """
    Vision Transformer (ViT) that encodes images into a sequence of
    contextualized vectors.
    """

    def __init__(self, image_size: int, patch_size: int, in_channels: int, d_model: int, num_layers: int,
                 num_heads: int, ff_hidden_config: List[int], random_seed: Optional[Any] = None,
                 # Random seed handled by torch.manual_seed
                 dropout_rate: float = 0.1, dtype=torch.float32, ):
        super().__init__()

        # Patch Embedding
        self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels, d_model, dropout_rate, dtype)
        num_patches = self.patch_embedding.num_patches

        # Learnable Class Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model, dtype=dtype) * 0.02)

        # Positional Embeddings
        # This is a fixed buffer, so it's registered as a buffer.
        self.pos_embedding = PositionalEncoding(d_model, max_len=num_patches + 1, dtype=dtype)  # +1 for CLS token

        # Dropout for initial embeddings
        self.input_dropout = nn.Dropout(dropout_rate)

        # Transformer Encoder Stack
        self.encoder_stack = nn.ModuleList(
            [TransformerEncoderBlock(d_model, num_heads, ff_hidden_config, dropout_rate=dropout_rate, dtype=dtype) for _
             in range(num_layers)])

        # Final Layer Normalization
        self.final_norm = nn.LayerNorm(d_model, dtype=dtype)

    def init_weights(self, module):
        """Applies a standard, robust initialization scheme."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Conv2d):  # Initialization for Conv2d
            torch.nn.init.kaiming_normal_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, image_input: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Takes a batch of images and returns a sequence of feature vectors.
        Output shape: (B, num_patches + 1, d_model)
        """
        # Patching embedding
        # Input: (B, C, H, W) -> Output: (B, num_patches, d_model)
        x = self.patch_embedding(image_input)

        # Adding CLS token
        B, num_patches, _ = x.shape
        # Broadcasting CLS token to match batch size
        cls_tokens = self.cls_token.expand(B, -1, -1)  # Shape: (B, 1, d_model)
        x = torch.cat((cls_tokens, x), dim=1)  # Concatenate along sequence dimension

        # Adding positional encoding
        x = self.pos_embedding(x)

        # Dropout
        x = self.input_dropout(x)

        # Passing through Transformer encoder stack
        # The padding_mask is None because a ViT processes a full, un-padded image grid.
        for block in self.encoder_stack:
            x = block(x, padding_mask=None)

        # Final normalization
        x = self.final_norm(x)

        return x
