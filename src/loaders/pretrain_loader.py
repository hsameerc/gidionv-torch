import random
from typing import Dict, List, Generator, Iterable, Any

import torch
import torch.nn.functional as F
from torch._C._nn import pad_sequence

from src.lib.core.hf_tokenizer_wrapper import HFTokenizerWrapper


def _pad_1d_sequence(sequence: List[int], max_len: int, pad_id: int) -> torch.Tensor:
    """Pads a single 1D list of token IDs into a 1D torch.Tensor."""
    padded = torch.full((max_len,), pad_id, dtype=torch.long)
    valid_len = min(len(sequence), max_len)
    if valid_len > 0:
        padded[:valid_len] = torch.tensor(sequence[:valid_len], dtype=torch.long)
    return padded


class StreamingDatasetProcessor:
    """
    Processes a stream of text documents into tokenized
    examples with multiple, distinct memory streams for pre-training.

    It implements the "Neighboring Documents" strategy, where each training
    example consists of a source to predict, an immediate context from the same
    document, and several other memory streams sampled from recently seen documents.
    """

    def __init__(self,
                 tokenizer: 'HFTokenizerWrapper',
                 seq_len: int,
                 num_memory_streams: int = 3,
                 overlap_len_tokens: int = 0):
        """
        Args:
            tokenizer: A tokenizer instance with `.encode()` and `.eos_token_id`.
            seq_len: The sequence length for the source and EACH memory stream.
            num_memory_streams: The total number of memory streams to generate.
            overlap_len_tokens: Number of tokens to overlap between examples from the same doc.
        """
        if num_memory_streams < 1:
            raise ValueError("num_memory_streams must be at least 1.")

        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.example_len = 2 * seq_len
        self.step_len = self.example_len - overlap_len_tokens
        self.num_memory_streams = num_memory_streams
        self.eos_token_id = tokenizer.eos_token_id

        if self.step_len <= 0:
            raise ValueError("overlap_len_tokens must be smaller than 2 * seq_len.")

    def process_stream(self,
                       text_stream: Iterable[str],
                       shuffle: bool = True,
                       buffer_size: int = 20000
                       ) -> Generator[Dict[str, List[List[int]]], None, None]:
        """
        A generator that takes a stream of text documents and yields tokenized,
        multi-stream examples.

        Each yielded item is a dictionary:
        {
            "source_ids": [token_ids],
            "memory_streams": [[stream1_ids], [stream2_ids], ...]
        }
        """
        example_buffer = []

        def flush_buffer():
            if shuffle:
                random.shuffle(example_buffer)
            for item in example_buffer:
                yield item
            example_buffer.clear()

        # This buffer holds the tokenized content of the last few documents
        # It's used to sample the "related context" memory streams.
        # The size is num_memory_streams because we might need that many distinct sources.
        doc_buffer = []
        token_remainder = []

        for text_document in text_stream:
            if not text_document:
                continue

            # Tokenizing the current document and prepend any remainder
            all_tokens = token_remainder + self.tokenizer.encode(text_document)

            # Splitting the token stream into complete documents based on the EOS token
            documents_in_chunk = []
            current_doc_tokens = []
            for token in all_tokens:
                if token == self.eos_token_id:
                    if current_doc_tokens:
                        documents_in_chunk.append(current_doc_tokens)
                    current_doc_tokens = []
                else:
                    current_doc_tokens.append(token)
            token_remainder = current_doc_tokens

            if not documents_in_chunk:
                continue

            # Processing each complete document found in the current text chunk
            for main_doc_tokens in documents_in_chunk:

                # Creating sliding window examples from the main document
                if len(main_doc_tokens) >= self.example_len:
                    for i in range(0, len(main_doc_tokens) - self.example_len + 1, self.step_len):

                        # The source is what the model must predict
                        source_ids = main_doc_tokens[i + self.seq_len: i + self.example_len]

                        # Memory Stream 1: The immediate context from the same document
                        mem_stream_1 = main_doc_tokens[i: i + self.seq_len]

                        # Preparing the list of all memory streams
                        memory_streams = [mem_stream_1]

                        # Memory Stream 2, 3, etc.: Sample from previous documents in the buffer
                        if self.num_memory_streams > 1 and doc_buffer:
                            # Getting a list of previous documents that are long enough
                            valid_prev_docs = [doc for doc in doc_buffer if len(doc) >= self.seq_len]

                            # How many additional streams do we need?
                            num_additional_streams = self.num_memory_streams - 1

                            # Sample with replacement from the valid previous docs
                            if valid_prev_docs:
                                sampled_docs = random.choices(valid_prev_docs, k=num_additional_streams)
                                for doc in sampled_docs:
                                    # Take a random chunk from the chosen previous document
                                    start_idx = random.randint(0, len(doc) - self.seq_len)
                                    memory_streams.append(doc[start_idx: start_idx + self.seq_len])

                        # Add the complete example to the buffer
                        example_buffer.append({
                            "source_ids": source_ids,
                            "memory_streams": memory_streams  # This is now a list of token lists
                        })

                        if len(example_buffer) >= buffer_size:
                            yield from flush_buffer()

                # Adding the processed main document to our buffer of recent documents
                doc_buffer.append(main_doc_tokens)
                # Keep the buffer from growing too large to save memory
                if len(doc_buffer) > 10:  # Keep the last 10 docs, a reasonable number
                    doc_buffer.pop(0)

        # Yielding any remaining examples
        if example_buffer:
            yield from flush_buffer()


def prepare_single_pretrain_item(
        item_data: Dict,
        tokenizer: 'HFTokenizerWrapper',
        config: dict
) -> Dict[str, Any]:
    """
    Prepares a SINGLE pre-training item from a
    multi-stream source, returning a dictionary of correctly shaped tensors and masks.
    """
    seq_len = config['max_seq_len']
    pad_id = tokenizer.pad_token_id
    num_mem_streams = config['model']['num_memory_streams']

    # Preparing Memory Streams
    memory_streams_ids_list = []

    # The processor provides a list of token lists in 'memory_streams'
    raw_memory_streams = item_data.get('memory_streams', [])

    # Processing each provided memory stream and pad it
    for stream_ids in raw_memory_streams:
        padded_stream = _pad_1d_sequence(stream_ids, seq_len, pad_id)
        memory_streams_ids_list.append(padded_stream)

    # Filling any remaining memory slots with empty/padding tensors
    # This ensures the list always has `num_mem_streams` items.
    num_placeholders = num_mem_streams - len(memory_streams_ids_list)
    if num_placeholders > 0:
        empty_stream = torch.full((seq_len,), pad_id, dtype=torch.long)
        memory_streams_ids_list.extend([empty_stream] * num_placeholders)

    # Stacking the list of 1D tensors into a single 2D tensor
    final_memory_streams = torch.stack(memory_streams_ids_list)
    # Creating the corresponding padding mask for the memory streams
    memory_padding_masks = (final_memory_streams != pad_id)

    # Preparing Main Input and Target IDs
    source_ids = item_data['source_ids']

    if len(source_ids) > 1:
        input_list = source_ids[:-1]
        target_list = source_ids[1:]
    else:
        input_list, target_list = [], []

    # Pad to create final 1D tensors
    input_ids = _pad_1d_sequence(input_list, seq_len, pad_id)
    target_ids = _pad_1d_sequence(target_list, seq_len, pad_id)

    # Creating Masks
    # Mask for the main decoder input's self-attention
    padding_mask = (input_ids != pad_id)

    # Mask for the loss function (ignore padding in the targets)
    target_ids[target_ids == pad_id] = -100

    # Returning Final Dictionary
    return {
        "input_ids": input_ids,  # Shape: (seq_len,)
        "target_ids": target_ids,  # Shape: (seq_len,)
        "padding_mask": padding_mask,  # Shape: (seq_len,)
        "memory_streams_ids": final_memory_streams,  # Shape: (num_streams, seq_len)
        "memory_padding_masks": memory_padding_masks,  # Shape: (num_streams, seq_len)
    }


def prepare_single_pretrain_item_unpadded_memory(
        item_data: Dict,
        tokenizer: 'HFTokenizerWrapper',
        config: dict
) -> Dict[str, Any]:
    """
    Prepares a SINGLE pre-training item.
    - Main input/target is padded/truncated to a fixed `max_seq_len`.
    - Memory streams are NOT padded and are returned as a list of variable-length tensors.
    """
    fixed_seq_len = config['max_seq_len']
    pad_id = tokenizer.pad_token_id
    num_mem_streams = config['model']['num_memory_streams']

    # Preparing Memory Streams (No Padding)
    memory_streams_ids_list = []
    raw_memory_streams = item_data.get('memory_streams', [])

    for stream_ids in raw_memory_streams:
        # Converting to a tensor. No padding, no truncation.
        memory_streams_ids_list.append(torch.tensor(stream_ids, dtype=torch.long))

    # Filling any remaining slots with EMPTY tensors.
    num_placeholders = num_mem_streams - len(memory_streams_ids_list)
    if num_placeholders > 0:
        empty_stream = torch.tensor([], dtype=torch.long)
        memory_streams_ids_list.extend([empty_stream] * num_placeholders)

    # Preparing Main Input and Target IDs (Fixed Padding)
    source_ids = item_data['source_ids']
    if len(source_ids) > 1:
        input_list, target_list = source_ids[:-1], source_ids[1:]
    else:
        input_list, target_list = source_ids, []

    input_ids = _pad_1d_sequence(input_list, fixed_seq_len, pad_id)
    target_ids = _pad_1d_sequence(target_list, fixed_seq_len, pad_id)
    target_ids[target_ids == pad_id] = -100

    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "memory_streams_ids": memory_streams_ids_list,
    }


def pretrain_padding_collate_fn(batch_items: List[Dict], pad_id: int) -> Dict[str, Any]:
    """
    Pretrain collate_fn that:
    1. Stacks fixed-size tensors (like input_ids).
    2. Dynamically pads and stacks variable-length tensors (like memory_streams_ids).
    3. Returns a batch dictionary with the SAME structure as your original pipeline.
    """
    collated_batch = {}

    #  Handling fixed-size tensors
    if 'input_ids' in batch_items[0]:
        collated_batch['input_ids'] = torch.stack([item['input_ids'] for item in batch_items])
    if 'target_ids' in batch_items[0]:
        collated_batch['target_ids'] = torch.stack([item['target_ids'] for item in batch_items])

    # Handling the variable-length memory streams
    if 'memory_streams_ids' in batch_items[0]:
        # `all_memories_in_batch` is a list of lists of tensors.
        # e.g., [[item1_stream1, item1_stream2], [item2_stream1, item2_stream2], ...]
        all_memories_in_batch = [item['memory_streams_ids'] for item in batch_items]

        num_streams = len(all_memories_in_batch[0])
        padded_streams_for_stacking = []

        for i in range(num_streams):
            # Gathering all tensors for this stream type from across the batch
            # e.g., [item1_stream1, item2_stream1, item3_stream1, ...]
            streams_of_type_i = [streams[i] for streams in all_memories_in_batch]

            # Using pad_sequence to pad this specific set of streams to the max
            # length of the longest stream of this type *in this batch*.
            padded_stream_i = pad_sequence(streams_of_type_i, batch_first=True, padding_value=pad_id)
            padded_streams_for_stacking.append(padded_stream_i)

        max_seq_len_across_all_streams = max(s.shape[1] for s in padded_streams_for_stacking)

        final_padded_streams = []
        for stream_tensor in padded_streams_for_stacking:
            padding_needed = max_seq_len_across_all_streams - stream_tensor.shape[1]
            if padding_needed > 0:
                # Pad on the sequence length dimension (dim=1)
                padded_tensor = F.pad(stream_tensor, (0, padding_needed), 'constant', pad_id)
                final_padded_streams.append(padded_tensor)
            else:
                final_padded_streams.append(stream_tensor)

        # Creating the Final Tensor
        # `final_padded_streams` is now a list of 2D tensors of the exact same size.
        # We can stack them to create the final 3D tensor your training loop expects.
        final_memory_tensor = torch.stack(final_padded_streams, dim=1)
        collated_batch['memory_streams_ids'] = final_memory_tensor

    return collated_batch
