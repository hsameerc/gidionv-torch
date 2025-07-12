import itertools
import random
import traceback

import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset, Dataset

from src.lib.core.hf_tokenizer_wrapper import HFTokenizerWrapper
from src.loaders.text_loader import StreamingDatasetProcessor
from src.utils.prepare import prepare_single_pretrain_item


class PretrainDataset(IterableDataset):
    """
    A robust, streaming IterableDataset for language model pretraining.

    It interleaves high-quality datasets like RedPajama and C4 in specified proportions,
    processes documents on-the-fly, and yields tokenized training samples.
    """

    def __init__(self, tokenizer: 'HFTokenizerWrapper', config: dict):
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config

        self.data_sources = {
            "redpajama_commoncrawl": {
                "id": "togethercomputer/RedPajama-Data-1T", "name": "common_crawl",
                "weight": 0.67
            },
            "c4": {
                "id": "allenai/c4", "name": "en",
                "weight": 0.15
            },
            "github": {
                "id": "togethercomputer/RedPajama-Data-1T", "name": "github",
                "weight": 0.045
            },
            "arxiv": {
                "id": "togethercomputer/RedPajama-Data-1T", "name": "arxiv",
                "weight": 0.09
            },
            "wikipedia": {
                "id": "togethercomputer/RedPajama-Data-1T", "name": "wikipedia",
                "weight": 0.045
            },
        }

    @staticmethod
    def _process_and_filter(example: dict) -> dict:
        """
        Processes a single example from any source.
        - Extracts text.
        - Checks for English language in 'meta' field if present.
        - Adds a 'keep' flag for easy filtering.
        """
        text = example.get('text', '')
        keep = False

        if 'meta' in example and isinstance(example['meta'], dict):
            if example['meta'].get('language') == 'en':
                if isinstance(text, str) and len(text.strip()) > 100:
                    keep = True
        elif 'repo_name' in example:  # A heuristic to identify github dataset
            if isinstance(text, str) and len(text.strip()) > 50:
                keep = True
        else:
            if isinstance(text, str) and len(text.strip()) > 100:
                keep = True

        return {"text": text, "keep": keep}

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        print(f"[Worker {worker_id}] Initializing streaming dataset pipeline...")

        streams = []
        probabilities = []

        for source_name, source_info in self.data_sources.items():
            try:
                ds = load_dataset(
                    path=source_info["id"],
                    name=source_info["name"],
                    split="train",
                    streaming=True,
                )

                processed_stream = ds.map(self._process_and_filter)

                final_generator = (
                    example["text"]
                    for example in processed_stream
                    if example["keep"]
                )

                streams.append(final_generator)
                probabilities.append(source_info["weight"])
                print(f"[Worker {worker_id}] Initialized stream: {source_name}")

            except Exception as e:
                print(f"[Worker {worker_id}] FAILED to load stream {source_name}. Skipping. Error: {e}")
                traceback.print_exc()
                continue

        if not streams:
            raise RuntimeError("Fatal: No dataset streams could be initialized.")

        total_prob = sum(probabilities)
        probabilities = [p / total_prob for p in probabilities]

        interleaved_stream = self._efficient_weighted_round_robin(streams, probabilities)

        processor = StreamingDatasetProcessor(
            tokenizer=self.tokenizer,
            seq_len=self.config['max_seq_len'],
            overlap_len_tokens=self.config.get('OVERLAP_LEN_TOKENS', 64)
        )
        raw_example_stream = processor.process_stream(interleaved_stream)

        for raw_item in raw_example_stream:
            yield prepare_single_pretrain_item(raw_item, self.tokenizer, self.config)

    @staticmethod
    def _efficient_weighted_round_robin(streams, weights):
        """
         Weighted round-robin iterator.
        - Shuffles once.
        - Doesn't rebuild lists during iteration.
        """
        iterators = [iter(s) for s in streams]
        weight_counts = [int(w * 1000) for w in weights]
        pool = list(itertools.chain.from_iterable([[i] * wc for i, wc in enumerate(weight_counts)]))
        random.shuffle(pool)

        active_iterators = list(range(len(iterators)))

        while active_iterators:
            if not pool:
                pool = list(itertools.chain.from_iterable([[i] * weight_counts[i] for i in active_iterators]))
                if not pool: break
                random.shuffle(pool)

            idx_to_try = pool.pop()

            try:
                yield next(iterators[idx_to_try])
            except StopIteration:
                if idx_to_try in active_iterators:
                    active_iterators.remove(idx_to_try)
                pool = [i for i in pool if i != idx_to_try]
            except Exception as e:
                print(f"Warning: Skipping an example from stream {idx_to_try} due to error: {e}")
                continue


class PretrainValidationDataset(Dataset):
    """
    A map-style Dataset for language model validation.
    It loads a fixed subset of a dataset, tokenizes it once, and
    stores it in memory for reproducible and efficient evaluation.
    """

    def __init__(self, tokenizer: 'HFTokenizerWrapper', config: dict):
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        self.examples = []
        self.val_max_samples = config.get("val_max_samples", 10000)

        print("Preparing fixed validation dataset...")
        self._prepare_data()
        print(f"Validation dataset prepared with {len(self.examples)} examples.")

    def _prepare_data(self):
        val_source = self.config.get("val_dataset", {
            "path": "allenai/c4",
            "name": "en",
            "split": "validation"
        })

        try:
            val_dataset = load_dataset(
                val_source["path"],
                val_source.get("name"),
                split=val_source.get("split", "validation"),
                streaming=True,
                # trust_remote_code=True
            )
            val_dataset_subset = val_dataset.take(self.val_max_samples)
            print(f"Loading up to {self.val_max_samples} documents from the validation set...")

        except Exception as e:
            print(f"Could not load validation set. Error: {e}")
            raise e

        processor = StreamingDatasetProcessor(
            tokenizer=self.tokenizer,
            seq_len=self.config['max_seq_len'],
            overlap_len_tokens=0
        )

        text_stream = (
            example['text'] for example in val_dataset_subset  # Iterate over the limited subset
            if 'text' in example and isinstance(example['text'], str) and len(example['text'].strip()) > 100
        )

        processed_examples = list(processor.process_stream(text_stream, shuffle=False))

        for raw_item in processed_examples:
            try:
                item = prepare_single_pretrain_item(raw_item, self.tokenizer, self.config)
                self.examples.append(item)
            except Exception as e:
                print(f"[Validation] Skipped an example due to error: {e}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]
