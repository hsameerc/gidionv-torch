import itertools
import random

import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset

from src.lib.core.hf_tokenizer_wrapper import HFTokenizerWrapper
from src.loaders.text_loader import StreamingDatasetProcessor
from src.utils.prepare import prepare_single_pretrain_item


class PretrainDataset(IterableDataset):
    """
    Streaming IterableDataset for large-scale language model pretraining.

    It interleaves high-quality datasets like RedPajama and C4 in specified proportions,
    processes documents on-the-fly, and yields tokenized training samples.
    """

    def __init__(self, tokenizer: 'HFTokenizerWrapper', config: dict):
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config

        self.data_mix = {
            "redpajama_commoncrawl": {
                "dataset": "togethercomputer/RedPajama-Data-1T",
                "config": "common_crawl",
                "weight": 0.67
            },
            "c4": {
                "dataset": "togethercomputer/RedPajama-Data-1T",
                "config": "c4",
                "weight": 0.15
            },
            "redpajama_github": {
                "dataset": "togethercomputer/RedPajama-Data-1T",
                "config": "github",
                "weight": 0.045
            },
            "redpajama_arxiv": {
                "dataset": "togethercomputer/RedPajama-Data-1T",
                "config": "arxiv",
                "weight": 0.045
            },
            "redpajama_wikipedia": {
                "dataset": "togethercomputer/RedPajama-Data-1T",
                "config": "wikipedia",
                "weight": 0.045
            },
        }

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        print(f"[Worker {worker_id}] Initializing streaming dataset...")

        def stream_dataset(path, config_name):
            try:
                ds_raw = load_dataset(path, name=config_name, split="train", streaming=True, trust_remote_code=True)
                ds_text_only = ds_raw.select_columns("text")
                for example in ds_text_only:
                    text = example['text']
                    if isinstance(text, str) and len(text.strip()) > 100:
                        yield text
            except Exception as e:
                print(f"[Worker {worker_id}] ERROR during streaming of {path}/{config_name}. Details: {e}")

        def weighted_round_robin(streams, weights):
            iterators = [iter(s) for s in streams]
            weight_counts = [int(w * 1000) for w in weights]
            pool = list(itertools.chain.from_iterable([[i] * wc for i, wc in enumerate(weight_counts)]))
            random.shuffle(pool)
            active_iterators = list(range(len(iterators)))

            while active_iterators:
                if not pool:
                    active_weights = [weight_counts[i] for i in active_iterators]
                    pool = list(
                        itertools.chain.from_iterable([[i] * wc for i, wc in zip(active_iterators, active_weights)]))
                    random.shuffle(pool)
                idx_to_try = pool.pop()

                try:
                    yield next(iterators[idx_to_try])
                except StopIteration:
                    if idx_to_try in active_iterators:
                        active_iterators.remove(idx_to_try)
                    pool = [i for i in pool if i != idx_to_try]
                except Exception as e:
                    print(f"Error yielding from iterator {idx_to_try}: {e}")
                    continue

        datasets = []
        weights = []

        for name, entry in self.data_mix.items():
            gen = stream_dataset(entry["dataset"], entry["config"])
            datasets.append(gen)
            weights.append(entry["weight"])
            print(f"[Worker {worker_id}] Initialized stream: {name} ({entry['dataset']}/{entry['config']})")

        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        interleaved_stream = weighted_round_robin(datasets, normalized_weights)
        processor = StreamingDatasetProcessor(
            tokenizer=self.tokenizer,
            seq_len=self.config['max_seq_len'],
            overlap_len_tokens=self.config.get('OVERLAP_LEN_TOKENS', 64)
        )
        processed_stream = processor.process_stream(interleaved_stream)
        for raw_item in processed_stream:
            yield prepare_single_pretrain_item(raw_item, self.tokenizer, self.config)
