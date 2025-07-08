import math
from typing import Dict, Optional, List, Tuple

import torch
import torch.nn as nn

from src.lib.core.encoders.audio_encoder import AudioEncoder
from src.lib.core.encoders.vision_encoder import VisionEncoder
from src.lib.core.hf_tokenizer_wrapper import HFTokenizerWrapper
from src.lib.core.maf_avt_decoder_block import MemoryAttentionFusionDecoderBlock
from src.lib.core.memory_encoder import MemoryEncoder
from src.lib.core.positional_encoding import PositionalEncoding


class MultiModalMemoryTransformer(nn.Module):
    """
     The complete, multi-modal, memory-augmented, stateful Transformer.
    This model composes modality-specific encoders and a powerful fusion decoder
    to reason over text, images, and audio simultaneously.
    """

    def __init__(self, config: Dict, tokenizer: HFTokenizerWrapper):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.d_model = int(config['d_model'])
        self.vocab_size = int(tokenizer.vocab_size)
        self.pad_token_id = int(tokenizer.pad_token_id)
        dtype = getattr(torch, config.get('model_dtype', 'float32'))

        # Shared and Common Components
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model, padding_idx=self.pad_token_id, dtype=dtype)
        self.positional_encoding = PositionalEncoding(self.d_model, max_len=config['max_seq_len'], dtype=dtype)
        self.input_dropout = nn.Dropout(config.get('dropout_rate', 0.1))

        # Modality-Specific Encoders
        self.text_memory_encoder = MemoryEncoder(num_layers=config['text_memory_encoder']['num_layers'],
                                                 d_model=self.d_model,
                                                 num_heads=config['text_memory_encoder']['num_heads'],
                                                 ff_hidden_config=config['text_memory_encoder']['ff_hidden_config'],
                                                 dropout_rate=config.get('dropout_rate', 0.1), dtype=dtype)
        self.vision_encoder = VisionEncoder(image_size=config['vision_encoder']['image_size'],
                                            patch_size=config['vision_encoder']['patch_size'],
                                            in_channels=config['vision_encoder']['in_channels'], d_model=self.d_model,
                                            num_layers=config['vision_encoder']['num_layers'],
                                            num_heads=config['vision_encoder']['num_heads'],
                                            ff_hidden_config=config['vision_encoder']['ff_hidden_config'],
                                            dropout_rate=config.get('dropout_rate', 0.1), dtype=dtype)
        self.audio_encoder = AudioEncoder(num_freq_bins=config['audio_encoder']['num_freq_bins'], d_model=self.d_model,
                                          num_layers=config['audio_encoder']['num_layers'],
                                          num_heads=config['audio_encoder']['num_heads'],
                                          ff_hidden_config=config['audio_encoder']['ff_hidden_config'],
                                          max_audio_len=config['audio_encoder']['max_audio_len'],
                                          dropout_rate=config.get('dropout_rate', 0.1), dtype=dtype)

        #  Fusion Decoder Stack
        decoder_config = config['decoder']
        self.decoder_blocks = nn.ModuleList([MemoryAttentionFusionDecoderBlock(  # Or HierarchicalFusionDecoderBlock
            d_model=self.d_model, num_heads=decoder_config['num_heads'],
            ff_hidden_config=decoder_config['ff_hidden_config'],
            num_memory_streams=config['model']['num_memory_streams'], dropout_rate=config.get('dropout_rate', 0.1),
            dtype=dtype) for _ in range(decoder_config['num_layers'])])

        # Final Output Layers
        self.final_norm = nn.LayerNorm(self.d_model, dtype=dtype)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False, dtype=dtype)

        # Weight Tying and Initialization
        if config.get('tie_weights', True):
            self.lm_head.weight = self.token_embedding.weight

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        num_layers = self.config['decoder']['num_layers']
        if isinstance(module, nn.Linear):
            scale_factor = 1 / math.sqrt(2.0 * num_layers)
            module.weight.data.normal_(mean=0.0, std=0.02 * scale_factor)

    def forward(self, input_ids: torch.Tensor, text_memory_ids: Optional[List[torch.Tensor]] = None,
                image_input: Optional[torch.Tensor] = None, audio_input: Optional[torch.Tensor] = None,
                kv_cache_list: Optional[List[Dict]] = None) -> Tuple[torch.Tensor, Optional[List[Dict]]]:

        memory_contexts = []

        # Route Modalities to Encoders
        if text_memory_ids:
            for ids in text_memory_ids:
                padding_mask = (ids != self.pad_token_id)
                mem_emb = self.token_embedding(ids) * math.sqrt(self.d_model)
                mem_pos_emb = self.positional_encoding(mem_emb)
                text_ctx = self.text_memory_encoder(mem_pos_emb, padding_mask=padding_mask)
                memory_contexts.append(text_ctx)

        if image_input is not None:
            image_context = self.vision_encoder(image_input)
            memory_contexts.append(image_context)

        if audio_input is not None:
            audio_context = self.audio_encoder(audio_input)
            memory_contexts.append(audio_context)

        # Prepare Main Input for Decoder
        target_padding_mask = (input_ids != self.pad_token_id)
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.input_dropout(x)

        # Pass through Fusion Decoder Stack
        next_kv_caches = [] if kv_cache_list is not None else None

        for i, block in enumerate(self.decoder_blocks):
            block_kv_cache = kv_cache_list[i] if kv_cache_list else None
            x, updated_kv_cache = block(x, memory_streams=memory_contexts, target_padding_mask=target_padding_mask,
                                        kv_cache=block_kv_cache)
            if next_kv_caches is not None:
                next_kv_caches.append(updated_kv_cache)

        # Final Layers
        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits, next_kv_caches

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, text_memory_ids: Optional[List[List[int]]] = None,
                 image_input: Optional[torch.Tensor] = None, audio_input: Optional[torch.Tensor] = None,
                 max_new_tokens: int = 128, temperature: float = 0.7, top_k: int = 50) -> torch.Tensor:
        pass
