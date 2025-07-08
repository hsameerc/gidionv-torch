from typing import Dict, List, Any

import torch

from src.lib.core.hf_tokenizer_wrapper import HFTokenizerWrapper


def _pad_sequences(sequences: List[List[int]], max_len: int, pad_id: int) -> torch.Tensor:
    """Pads a list of token ID lists into a torch.Tensor."""
    batch_size = len(sequences)
    # Handle empty sequences list
    if batch_size == 0:
        return torch.empty((0, max_len), dtype=torch.long)

    padded = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(sequences):
        # Truncate sequence if it's longer than max_len
        seq = seq[:max_len]
        valid_len = len(seq)
        if valid_len > 0:
            padded[i, :valid_len] = torch.tensor(seq, dtype=torch.long)
    return padded


def _pad_1d_sequence(sequence: List[int], max_len: int, pad_id: int) -> torch.Tensor:
    """Pads a single 1D list of token IDs into a 1D torch.Tensor."""
    padded = torch.full((max_len,), pad_id, dtype=torch.long)
    valid_len = min(len(sequence), max_len)
    if valid_len > 0:
        padded[:valid_len] = torch.tensor(sequence[:valid_len], dtype=torch.long)
    return padded


def format_prompt(instruction: str, context: str, special_tokens: dict) -> str:
    """Formats the structured data into the final prompt string."""
    user_token = special_tokens.get("USER", "<USER>")
    inst_token = special_tokens.get("INST", "<INST>")
    end_inst_token = special_tokens.get("END_INST", "</INST>")
    assistant_token = special_tokens.get("ASSISTANT", "<ASSISTANT>")

    if context:
        prompt_instruction = f"Use the provided context to answer the following instruction.\n\nContext: {context[:500]}...\n\nInstruction: {instruction}"
    else:
        prompt_instruction = instruction

    return f"{user_token}{inst_token} {prompt_instruction} {end_inst_token}{assistant_token}"


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


def prepare_single_instruction_item(raw_item: Dict, tokenizer: 'HFTokenizerWrapper', config: dict,
                                          special_tokens: dict) -> Dict[str, Any]:
    """
    Prepares a SINGLE fine-tuning item, returning
    a dictionary with all required tensors and masks.
    """
    seq_len = config['max_seq_len']
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    # Preparing Memory Streams and their Masks
    context_text = raw_item.get('context', '')
    context_ids = tokenizer.encode(context_text)

    max_context_len = config.get('max_context_len', seq_len)
    padded_context = _pad_1d_sequence(context_ids, max_context_len, pad_id)

    memory_streams_ids_list = [padded_context]
    num_total_mem_streams = config['model']['num_memory_streams']
    if num_total_mem_streams > 1:
        empty_stream  = torch.full_like(padded_context, pad_id,  dtype=torch.long)
        memory_streams_ids_list.extend([empty_stream] * (num_total_mem_streams - 1))

    # Stacking the list of 1D tensors into a single 2D tensor
    # Shape: (num_memory_streams, max_context_len)
    final_memory_streams = torch.stack(memory_streams_ids_list)

    # Creating the memory padding mask from the final tensor
    # Shape: (num_memory_streams, max_context_len)
    memory_padding_masks = (final_memory_streams != pad_id)

    # Formatting the Main Prompt and Target Output
    prompt_text = format_prompt(raw_item['instruction'], raw_item['context'], special_tokens)
    response_text = raw_item['output']

    prompt_ids = tokenizer.encode(prompt_text)
    response_ids = tokenizer.encode(response_text)

    # Creating Final Input/Target Tensors
    full_sequence = prompt_ids + response_ids + ([eos_id] if eos_id is not None else [])

    input_list = full_sequence[:-1]
    target_list = full_sequence[1:]

    input_ids = _pad_1d_sequence(input_list, seq_len, pad_id)
    target_ids = _pad_1d_sequence(target_list, seq_len, pad_id)

    # Creating Loss Mask and Padding Mask
    prompt_len = len(prompt_ids)
    if prompt_len < seq_len:
        target_ids[:prompt_len] = -100

    target_ids[target_ids == pad_id] = -100

    # Creating the padding mask for the main decoder input
    # Shape: (seq_len,)
    padding_mask = (input_ids != pad_id)
    # Return the complete, model-ready item
    return {"input_ids": input_ids,  # Shape: (seq_len,)
            "target_ids": target_ids,  # Shape: (seq_len,)
            "padding_mask": padding_mask,  # Shape: (seq_len,)
            "memory_streams_ids": final_memory_streams,  # Shape: (num_streams, context_len)
            "memory_padding_masks": memory_padding_masks,  # Shape: (num_streams, context_len)
            }
