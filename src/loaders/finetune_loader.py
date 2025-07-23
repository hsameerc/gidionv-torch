import atexit
import json
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset

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

def format_without_context_prompt(instruction: str, special_tokens: dict) -> str:
    """
    [SIMPLIFIED] Formats the instruction into the final prompt string.
    It does NOT include the context.
    """
    user_token = special_tokens.get("USER", "<USER>")
    inst_token = special_tokens.get("INST", "<INST>")
    end_inst_token = special_tokens.get("END_INST", "</INST>")
    assistant_token = special_tokens.get("ASSISTANT", "<ASSISTANT>")

    prompt_instruction = instruction

    return f"{user_token}{inst_token} {prompt_instruction} {end_inst_token}{assistant_token} "

class IndexedJsonlDataset(Dataset):
    """
    A high-performance, memory-efficient PyTorch Dataset for very large .jsonl files.

    This class creates an index of byte offsets for each line in the file, allowing for
    fast, random access to any data point. It is designed to be used with a
    PyTorch `DataLoader` and multiple workers, where each worker will keep its own
    file handle open to avoid repeated open/close overhead.
    """

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")

        self._line_offsets: List[int] = []

        # File handle, one per worker process
        # This will be initialized lazily in __getitem__ to be compatible
        # with PyTorch's multiprocessing DataLoader.
        self._file_handle = None

        print(f"Indexing large JSONL file: {self.filepath}...")
        self._build_index()
        print(f"Indexing complete. Found {len(self)} lines.")

        # Ensure file handles are closed when the main Python process exits
        atexit.register(self.close)

    def _build_index(self):
        """
        Scans the file once to build a byte offset index for each line.
        This is a more efficient implementation.
        """
        with self.filepath.open('rb') as f:
            self._line_offsets.append(0)
            while True:
                line = f.readline()
                if not line:
                    break
                self._line_offsets.append(f.tell())

        self._line_offsets.pop()

    def __len__(self) -> int:
        """Returns the total number of lines (samples) in the file."""
        return len(self._line_offsets)

    def __getitem__(self, index: int) -> Dict:
        """
        Retrieves and parses a single JSON object by its line index.
        It lazily opens a file handle for each worker process.
        """
        if not 0 <= index < len(self):
            raise IndexError(f"Index {index} is out of range for file with {len(self)} lines.")

        # Each worker process gets its own file handle, which stays open.
        if self._file_handle is None:
            self._file_handle = self.filepath.open('rb')

        # Seek to the pre-computed byte offset and read the line
        self._file_handle.seek(self._line_offsets[index])
        line_bytes = self._file_handle.readline()

        # Decode and parse the JSON line
        return json.loads(line_bytes.decode('utf-8'))

    def close(self):
        """Closes the file handle."""
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None

    def __del__(self):
        """Destructor to ensure file handle is closed when the object is destroyed."""
        self.close()


def prepare_single_instruction_item(raw_item: Dict, tokenizer: 'HFTokenizerWrapper', config: dict,
                                    special_tokens: dict) -> Dict[str, Any]:
    """
    Prepares a SINGLE fine-tuning item, returning
    a dictionary with all required tensors and masks.
    """
    seq_len = config['max_seq_len']
    pad_id = tokenizer.pad_token_id
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    num_mem_streams = config['model']['num_memory_streams']

    # Preparing Memory Streams and their Masks
    context_text = raw_item.get('context', '')
    target_slot = int(raw_item.get('mem_slot', -1))
    context_ids = tokenizer.encode(context_text)

    max_context_len = config.get('max_context_len', seq_len)
    padded_context = _pad_1d_sequence(context_ids, max_context_len, pad_id)

    empty_stream = torch.full((max_context_len,), pad_id, dtype=torch.long)
    memory_streams_ids_list = [empty_stream.clone() for _ in range(num_mem_streams)]
    if context_text and 0 <= target_slot < num_mem_streams:
        context_ids = tokenizer.encode(context_text, add_special_tokens=False)
        padded_context = _pad_1d_sequence(context_ids, max_context_len, pad_id)
        # Overwrite the placeholder at the target slot with the real context
        memory_streams_ids_list[target_slot] = padded_context

    # memory_streams_ids_list = [padded_context]
    # num_total_mem_streams = config['model']['num_memory_streams']
    # if num_total_mem_streams > 1:
    #     empty_stream = torch.full_like(padded_context, pad_id, dtype=torch.long)
    #     memory_streams_ids_list.extend([empty_stream] * (num_total_mem_streams - 1))

    # Stacking the list of 1D tensors into a single 2D tensor
    # Shape: (num_memory_streams, max_context_len)
    final_memory_streams = torch.stack(memory_streams_ids_list)

    # Creating the memory padding mask from the final tensor
    # Shape: (num_memory_streams, max_context_len)
    memory_padding_masks = (final_memory_streams != pad_id)

    # Formatting the Main Prompt and Target Output
    prompt_text = format_without_context_prompt(raw_item['instruction'], special_tokens)
    response_text = raw_item['output']

    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    response_ids = tokenizer.encode(response_text, add_special_tokens=False)

    # Creating Final Input/Target Tensors
    full_sequence = ([bos_id] if bos_id is not None else []) + prompt_ids + response_ids + ([eos_id] if eos_id is not None else [])

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
