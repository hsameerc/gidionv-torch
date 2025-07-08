from typing import Dict, List

import torch


def _torch_pad_sequences(sequences: List[List[int]], max_len: int, pad_id: int) -> torch.Tensor:
    """A robust helper to pad a list of token ID lists into a torch.Tensor."""
    batch_size = len(sequences)
    padded = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(sequences):
        valid_len = min(len(seq), max_len)
        padded[i, :valid_len] = torch.tensor(seq[:valid_len], dtype=torch.long)
    return padded


def prepare_single_pretrain_item(item_data: Dict[str, List[int]], tokenizer, config) -> Dict[str, torch.Tensor]:
    """
    Prepares a SINGLE pre-training item, returning 1D Tensors.
    The DataLoader will be responsible for batching.
    """
    seq_len = config['max_seq_len']
    pad_id = tokenizer.pad_token_id

    # Logic for a single item
    source_ids = item_data['source_ids']
    context_ids = item_data['context_ids']

    # Pad the single sequence to create a 1D tensor of shape (seq_len)
    memory_stream_1 = torch.full((seq_len,), pad_id, dtype=torch.long)
    valid_len_ctx = min(len(context_ids), seq_len)
    memory_stream_1[:valid_len_ctx] = torch.tensor(context_ids[:valid_len_ctx], dtype=torch.long)

    num_mem_streams = config['model']['num_memory_streams']
    memory_streams_ids = [memory_stream_1]
    if num_mem_streams > 1:
        zero_stream = torch.full_like(memory_stream_1, pad_id)
        memory_streams_ids.extend([zero_stream] * (num_mem_streams - 1))

    # Process source_ids for a single item
    if len(source_ids) > 1:
        input_ids_list = source_ids[:-1]
        target_ids_list = source_ids[1:]
    else:
        input_ids_list = []
        target_ids_list = []

    # Pad to create 1D tensors of shape (seq_len)
    input_ids = torch.full((seq_len,), pad_id, dtype=torch.long)
    valid_len_in = min(len(input_ids_list), seq_len)
    input_ids[:valid_len_in] = torch.tensor(input_ids_list[:valid_len_in], dtype=torch.long)

    target_ids = torch.full((seq_len,), pad_id, dtype=torch.long)
    valid_len_tgt = min(len(target_ids_list), seq_len)
    target_ids[:valid_len_tgt] = torch.tensor(target_ids_list[:valid_len_tgt], dtype=torch.long)

    target_ids[target_ids == pad_id] = -100

    padding_mask = (input_ids != pad_id).to(dtype=torch.bool)
    memory_padding_masks = [(stream != pad_id).to(dtype=torch.bool) for stream in memory_streams_ids]
    return {"input_ids": input_ids, "memory_streams_ids": memory_streams_ids, "target_ids": target_ids,
        "padding_mask": padding_mask, "memory_padding_masks": memory_padding_masks}


def prepare_instruction_batch(batch_text_data: List[Dict], tokenizer, config: dict) -> dict:
    """
    Prepares a batch for instruction fine-tuning, using the idiomatic PyTorch
    approach of setting ignored targets to -100.
    """
    prompts = [item['source'] for item in batch_text_data]
    responses = [item['target'] for item in batch_text_data]
    seq_len = config['max_seq_len']

    pad_id = tokenizer.pad_token_id
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    # Memory Stream preparation is a direct translation
    prompt_token_ids = [tokenizer.encode(p) for p in prompts]
    memory_stream_1 = _torch_pad_sequences(prompt_token_ids, seq_len, pad_id)
    num_mem_streams = config['model']['num_memory_streams']
    memory_streams_ids = [memory_stream_1]
    if num_mem_streams > 1:
        zero_stream = torch.full_like(memory_stream_1, pad_id)
        memory_streams_ids.extend([zero_stream] * (num_mem_streams - 1))

    # Sequence construction and padding
    response_with_eos_ids = [tokenizer.encode(r) + ([eos_id] if eos_id else []) for r in responses]
    full_sequences = [([bos_id] if bos_id else []) + p_ids + r_ids for p_ids, r_ids in
                      zip(prompt_token_ids, response_with_eos_ids)]

    # Truncate and create input/target pairs
    input_ids_list = [seq[:-1][:seq_len] for seq in full_sequences]
    target_ids_list = [seq[1:][:seq_len] for seq in full_sequences]

    input_ids = _torch_pad_sequences(input_ids_list, seq_len, pad_id)
    target_ids = _torch_pad_sequences(target_ids_list, seq_len, pad_id)

    has_bos = 1 if bos_id is not None else 0
    for i, p_ids in enumerate(prompt_token_ids):
        # Calculate the length of the prompt part in the target sequence
        # The target sequence is shifted by 1, so the prompt part ends at len(p_ids).
        prompt_len_in_target = len(p_ids) + has_bos
        # We want to ignore the prompt tokens in the loss calculation.
        target_ids[i, :prompt_len_in_target] = -100
    # We also ignore any padding tokens in the targets.
    target_ids[target_ids == pad_id] = -100

    return {"input_ids": input_ids, "memory_streams_ids": memory_streams_ids, "target_ids": target_ids,
            "padding_mask": (input_ids != pad_id)}
