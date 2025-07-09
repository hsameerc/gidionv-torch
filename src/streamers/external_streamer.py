import itertools
import random
import traceback

import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset

from src.lib.core.hf_tokenizer_wrapper import HFTokenizerWrapper
from src.loaders.text_loader import StreamingDatasetProcessor
from src.utils.prepare import prepare_single_pretrain_item


class PretrainDataset(IterableDataset):
    """
    Streaming IterableDataset for our language model pretraining.

    It interleaves high-quality datasets like RedPajama C4, and codeparrot in specified proportions,
    processes documents on-the-fly, and yields tokenized training samples.
    """

    def __init__(self, tokenizer: 'HFTokenizerWrapper', config: dict):
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        self.data_mix = {
            "redpajama_commoncrawl": {
                "dataset": "togethercomputer/RedPajama-Data-1T", "config": "common_crawl", "weight": 0.67
            },
            "c4": {
                "dataset": "allenai/c4", "config": "en", "weight": 0.15
            },
            "redpajama_arxiv": {
                "dataset": "togethercomputer/RedPajama-Data-1T", "config": "arxiv", "weight": 0.045
            },
            "redpajama_wikipedia": {
                "dataset": "togethercomputer/RedPajama-Data-1T", "config": "wikipedia", "weight": 0.045
            },
            "codeparrot_github": {
                "dataset": "codeparrot/github-code", "config": None, "weight": 0.045
            },
        }

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        print(f"[Worker {worker_id}] Initializing streaming dataset...")

        def stream_dataset(path, config_name):
            try:
                ds_raw = load_dataset(path, name=config_name, split="train", streaming=True, trust_remote_code=True)

                def flatten_and_extract_lang(example):
                    if 'meta' in example and isinstance(example['meta'], dict):
                        lang = example['meta'].get('language', 'unknown')
                    else:
                        lang = 'en'
                    return {'text': example.get('text', ''), 'language': lang}

                processed_stream = ds_raw.map(flatten_and_extract_lang)
                for raw_example in processed_stream:
                    if raw_example['language'] == 'en':
                        text = raw_example['text']
                        if isinstance(text, str) and len(text.strip()) > 100:
                            yield text

            except Exception as e:
                print(f"[Worker {worker_id}] ERROR during streaming of {path}/{config_name}. Details: {e}")
                traceback.print_exc()

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
                    if not pool: break
                    random.shuffle(pool)

                idx_to_try = pool.pop()

                try:
                    yield next(iterators[idx_to_try])
                except StopIteration:
                    if idx_to_try in active_iterators:
                        active_iterators.remove(idx_to_try)
                    pool = [i for i in pool if i != idx_to_try]
                except (KeyError, ValueError) as e:
                    print(f"[Worker {worker_id}] Skipping example due to error: {e}")
                    continue
                except Exception as e:
                    print(f"[Worker {worker_id}] Unexpected error in iterator {idx_to_try}: {type(e).__name__}: {e}")
                    traceback.print_exc()
                    active_iterators.remove(idx_to_try)
                    pool = [i for i in pool if i != idx_to_try]

        datasets = []
        weights_data = []
        for name, entry in self.data_mix.items():
            gen = stream_dataset(entry["dataset"], entry["config"])
            datasets.append(gen)
            weights_data.append(entry["weight"])
            print(
                f"[Worker {worker_id}] Initialized stream with EN filter: {name} ({entry['dataset']}/{entry['config']})")

        total_weight = sum(weights_data)
        normalized_weights = [w / total_weight for w in weights_data]
        interleaved_stream = weighted_round_robin(datasets, normalized_weights)
        processor = StreamingDatasetProcessor(tokenizer=self.tokenizer, seq_len=self.config['max_seq_len'],
                                              overlap_len_tokens=self.config.get('OVERLAP_LEN_TOKENS', 64))
        raw_processed_stream = processor.process_stream(interleaved_stream)

        for raw_item in raw_processed_stream:
            yield prepare_single_pretrain_item(raw_item, self.tokenizer, self.config)
