import random

import torch
from torch.utils.data import IterableDataset, Dataset

from src.lib.core.hf_tokenizer_wrapper import HFTokenizerWrapper
from src.loaders.pretrain_loader import prepare_single_pretrain_item
from src.loaders.text_loader import TextLoaderStream, AdvancedDataStreamer, IndexedJsonlDataset
from src.utils.prepare import prepare_single_instruction_item


class StreamLocalPretrainDataset(IterableDataset):
    """
    An IterableDataset for pre-training on a single, large, local .txt file.
    It uses AdvancedDataStreamer to handle memory-mapping and chunking,
    and then prepares the data into the multi-stream format.
    """

    def __init__(self, filepath: str, tokenizer: 'HFTokenizerWrapper', config: dict):
        super().__init__()
        self.filepath = filepath
        self.tokenizer = tokenizer
        self.config = config

    def __iter__(self):
        """
        The entry point for the DataLoader worker.
        Initializes streamers here to ensure compatibility with multiple workers.
        """
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        print(f"[Worker {worker_id}] Initializing local file streamer for {self.filepath}...")

        # Creating the stream from the local file
        text_stream = TextLoaderStream(self.filepath)

        # Using AdvancedDataStreamer
        data_streamer = AdvancedDataStreamer(
            text_stream=text_stream,
            tokenizer=self.tokenizer,
            seq_len=self.config['max_seq_len'],
            overlap_len_tokens=self.config.get('OVERLAP_LEN_TOKENS', 64)
        )

        # The `stream_data` method yields single examples of the OLD format:
        # e.g., {"source_ids": [...], "context_ids": [...]}
        # We need to adapt this to the NEW multi-stream format.

        # This buffer will hold recent context chunks to be used as memory
        num_mem_streams = self.config['model']['num_memory_streams']
        context_buffer = []

        for old_format_item in data_streamer.stream_data(shuffle=True):

            source_ids = old_format_item['source_ids']
            immediate_context = old_format_item['context_ids']

            # Creating the multi-stream item here
            memory_streams = [immediate_context]  # Stream 1 is always the immediate context

            # Sample other memory streams from our buffer of recent contexts
            if num_mem_streams > 1 and context_buffer:
                num_to_sample = min(len(context_buffer), num_mem_streams - 1)
                # Sample from the most recent contexts
                sampled_contexts = random.sample(context_buffer, k=num_to_sample)
                memory_streams.extend(sampled_contexts)

            # This is the new raw item format our prep function expects
            new_raw_item = {
                "source_ids": source_ids,
                "memory_streams": memory_streams
            }
            # Adding the current context to the buffer for future examples
            context_buffer.append(immediate_context)
            # Keeping the buffer from growing too large
            if len(context_buffer) > 100:  # Keep a buffer of 100 recent contexts
                context_buffer.pop(0)

            # Yielding the fully prepared item for the DataLoader
            yield prepare_single_pretrain_item(new_raw_item, self.tokenizer, self.config)


class FinetuneLocalDataset(Dataset):
    """
    A map-style Dataset for fine-tuning on structured .jsonl files.
    It uses an IndexedJsonlDataset for efficient random access.
    """

    def __init__(self, filepath: str, tokenizer: 'HFTokenizerWrapper', config: dict, special_tokens: dict):
        super().__init__()
        self.indexed_data = IndexedJsonlDataset(filepath)
        self.tokenizer = tokenizer
        self.config = config
        self.special_tokens = special_tokens

    def __len__(self) -> int:
        return len(self.indexed_data)

    def __getitem__(self, index: int) -> dict:
        raw_item = self.indexed_data[index]
        return prepare_single_instruction_item(raw_item, self.tokenizer, self.config, self.special_tokens)
