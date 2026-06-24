import itertools
import random
import re
import traceback
from html import unescape

import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset, Dataset

from src.lib.core.hf_tokenizer_wrapper import HFTokenizerWrapper
from src.loaders.pretrain_loader import StreamingDatasetProcessor, prepare_single_pretrain_item


class PretrainDatasetStreamer(IterableDataset):
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
            "arxiv": {
                "id": "togethercomputer/RedPajama-Data-1T", "name": "arxiv",
                "weight": 0.09
            },
            "wikipedia": {
                "id": "togethercomputer/RedPajama-Data-1T", "name": "wikipedia",
                "weight": 0.09
            },
        }

    @staticmethod
    def _process_and_filter(example: dict) -> dict:
        """
        Processes and filters a single example to improve data quality.

        Quality Checks:
        1. Language: English only.
        2. HTML Stripping (optional, safe fallback).
        3. Minimum Length: 50 words.
        4. Alphabetic Ratio: ≥70% alpha characters.
        5. Line Repetition Check.
        6. N-gram Repetition Check.
        7. Boilerplate Phrase Filtering.
        8. "Lorem Ipsum" Filter.
        """
        text = example.get('text', '')
        keep = True

        # Language Check
        if 'meta' in example and isinstance(example['meta'], dict):
            if example['meta'].get('language') != 'en':
                return {"text": "", "keep": False}

        # Valid string?
        if not isinstance(text, str) or not text.strip():
            return {"text": "", "keep": False}

        # Optional HTML decode and strip
        cleaned_text = unescape(text.strip())
        cleaned_text = re.sub(r"<[^>]+>", "", cleaned_text)  # Strip HTML tags

        words = cleaned_text.split()
        num_words = len(words)
        if num_words < 50:
            return {"text": "", "keep": False}

        # Alphabetic ratio
        alpha_chars = sum(c.isalpha() for c in cleaned_text)
        total_chars = len(cleaned_text)
        if total_chars == 0 or (alpha_chars / total_chars) < 0.70:
            return {"text": "", "keep": False}

        # Line repetition check
        lines = [line.strip() for line in cleaned_text.split('\n') if line.strip()]
        if len(lines) > 10:
            unique_ratio = len(set(lines)) / len(lines)
            if unique_ratio < 0.5:
                return {"text": "", "keep": False}

        # 5-gram repetition check
        if num_words > 20:
            ngrams = set()
            dupes = 0
            for i in range(num_words - 4):
                ng = " ".join(words[i:i + 5])
                if ng in ngrams:
                    dupes += 1
                else:
                    ngrams.add(ng)
            if dupes / (num_words - 4) > 0.4:
                return {"text": "", "keep": False}

        # Boilerplate filters
        boilerplate_phrases = [
            "terms of use", "privacy policy", "cookie policy", "rights reserved",
            "log in", "sign up", "javascript is disabled", "enable javascript",
            "view our", "back to top", "copyright"
        ]
        text_lower = cleaned_text.lower()
        if any(phrase in text_lower for phrase in boilerplate_phrases):
            return {"text": "", "keep": False}

        # Lorem Ipsum check
        if "lorem ipsum" in text_lower:
            return {"text": "", "keep": False}

        return {"text": cleaned_text, "keep": True}


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
                    trust_remote_code=True,
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
            overlap_len_tokens=self.config.get('OVERLAP_LEN_TOKENS', 64),
            num_memory_streams=self.config['model']['num_memory_streams']
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
                trust_remote_code=True,
            )
            val_dataset_subset = val_dataset.take(self.val_max_samples)
            print(f"Loading up to {self.val_max_samples} documents from the validation set...")

        except Exception as e:
            print(f"Could not load validation set. Error: {e}")
            raise e

        processor = StreamingDatasetProcessor(
            tokenizer=self.tokenizer,
            seq_len=self.config['max_seq_len'],
            overlap_len_tokens=0,
            num_memory_streams=self.config['model']['num_memory_streams']
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
