import itertools
import random
import traceback
from typing import Iterable, Dict, Optional

import torch
from datasets import load_dataset, tqdm
from torch.utils.data import IterableDataset

from src.lib.core.hf_tokenizer_wrapper import HFTokenizerWrapper
from src.utils.prepare import prepare_single_instruction_item


class FinetuneDatasetStream(IterableDataset):
    """
    A comprehensive, streaming IterableDataset for instruction fine-tuning.

    It combines multiple datasets (Alpaca, SQuAD, Dolly, etc.), each with its own
    specialized on-the-fly processor. It handles complex cases like dialogue unrolling
    and 1-to-many sample generation, interleaves the sources by weight, and uses a
    standard preparation function for final tokenization.
    """

    def __init__(self, tokenizer: 'HFTokenizerWrapper', config: dict, special_tokens: dict):
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        self.special_tokens = special_tokens

        self.data_sources = {
            "web_questions": {
                "id": "web_questions", "split": "train",
                "weight": 0.10, "processor": self._process_web_questions
            },
            "natural_questions": {
                "id": "google-research-datasets/natural_questions", "split": "train",
                "weight": 0.30, "processor": self.process_natural_questions
            },
            "alpaca": {
                "id": "yahma/alpaca-cleaned", "split": "train",
                "weight": 0.10, "processor": self._process_alpaca
            },
            "dolly": {
                "id": "databricks/databricks-dolly-15k", "split": "train",
                "weight": 0.10, "processor": self._process_dolly
            },
            "squad_v2": {
                "id": "squad_v2", "split": "train",
                "weight": 0.10, "processor": self._process_squad
            },
            "daily_dialog": {
                "id": "daily_dialog", "split": "train",
                "weight": 0.10, "processor": self._process_daily_dialog
            },
            "trivia_qa": {
                "id": "trivia_qa", "name": "rc.nocontext", "split": "train",
                "weight": 0.10, "processor": self._process_trivia_qa
            },
            "chain_of_thought": {
                "id": "AlekseyKorshuk/chain-of-thoughts-chatml", "split": "train",
                "weight": 0.10, "processor": self._process_cot
            },
        }

    @staticmethod
    def _process_web_questions(example: dict) -> Optional[dict]:
        """Processes a single example from the web_questions dataset."""
        if not example.get('answers') or not example['answers']:
            return None

        return {
            "instruction": example['question'].strip(),
            "context": "",
            "output": example['answers'][0].strip()
        }

    @staticmethod
    def process_natural_questions(example: dict) -> Optional[dict]:
        """Processes a single example from the Natural Questions dataset."""
        try:
            short_answers = example['annotations']['short_answers']
            if len(short_answers) > 0 and len(short_answers[0]['text']) > 0:
                instruction = example['question']['text'].strip()
                output = short_answers[0]['text'][0].strip()

                if instruction and output:
                    return {"instruction": instruction, "context": "", "output": output}
        except (KeyError, IndexError):
            return None
        return None

    @staticmethod
    def _process_alpaca(example: dict) -> Iterable[Dict]:
        """Processes a single example from the Alpaca dataset."""
        instruction = example.get('instruction', '').strip()
        inp = example.get('input', '').strip()
        output = example.get('output', '').strip()
        if instruction and output:
            yield {"instruction": instruction, "context": inp, "output": output}

    @staticmethod
    def _process_dolly(example: dict) -> Iterable[Dict]:
        """Processes a single example from the Dolly-15k dataset."""
        instruction = example.get('instruction', '').strip()
        context = example.get('context', '').strip()
        output = example.get('response', '').strip()
        if instruction and output:
            yield {"instruction": instruction, "context": context, "output": output}

    @staticmethod
    def _process_squad(example: dict) -> Iterable[Dict]:
        """Processes a single example from the SQuAD v2 dataset."""
        context = example.get('context', '').strip()
        question = example.get('question', '').strip()
        answers = example.get('answers', {}).get('text', [])
        if context and question and answers:
            yield {"instruction": question, "context": context, "output": answers[0].strip()}

    @staticmethod
    def _process_trivia_qa(example: dict) -> Iterable[Dict]:
        """Processes a single example from the TriviaQA dataset."""
        question = example.get('question', '').strip()
        answer = example.get('answer', {}).get('value', '').strip()
        if question and answer:
            yield {"instruction": question, "context": "", "output": answer}

    @staticmethod
    def _process_daily_dialog(example: dict) -> Iterable[Dict]:
        """Unrolls a dialogue from the daily_dialog dataset into multiple samples."""
        dialog = example.get('dialog', [])
        if not dialog or len(dialog) < 2:
            return

        for i in range(1, len(dialog)):
            context_turns = dialog[:i]
            target_output = dialog[i].strip()
            instruction = "Continue the conversation naturally."
            context = "\n".join(f"Turn {t + 1}: {turn.strip()}" for t, turn in enumerate(context_turns))
            if target_output:
                yield {"instruction": instruction, "context": context, "output": target_output}

    @staticmethod
    def _process_cot(example: dict) -> Iterable[Dict]:
        """Processes a conversation from the Chain-of-Thoughts dataset."""
        conversation = example.get('conversation', [])
        if not conversation or len(conversation) < 2:
            return

        for i in range(0, len(conversation) - 1, 2):
            if conversation[i]['role'] == 'user' and conversation[i + 1]['role'] == 'assistant':
                instruction = conversation[i]['content'].strip()
                output = conversation[i + 1]['content'].strip()
                if instruction and output:
                    yield {"instruction": instruction, "context": "", "output": output}

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        print(f"[Worker {worker_id}] Initializing fine-tuning streaming pipeline...")

        streams = []
        probabilities = []

        for source_name, source_info in self.data_sources.items():
            try:
                ds = load_dataset(
                    path=source_info["id"],
                    name=source_info.get("name"),  # Use .get() for optional 'name'
                    split=source_info["split"],
                    streaming=True,
                    trust_remote_code=True
                )

                processed_generator = (
                    structured_item
                    for raw_example in ds
                    for structured_item in source_info["processor"](raw_example)
                )

                streams.append(processed_generator)
                probabilities.append(source_info["weight"])
                print(f"[Worker {worker_id}] Initialized stream: {source_name}")

            except Exception as e:
                print(f"[Worker {worker_id}] FAILED to load stream {source_name}. Skipping. Error: {e}")
                traceback.print_exc()
                continue

        if not streams:
            raise RuntimeError("Fatal: No fine-tuning dataset streams could be initialized.")

        total_prob = sum(probabilities)
        probabilities = [p / total_prob for p in probabilities]

        interleaved_stream = self._efficient_weighted_round_robin(streams, probabilities)

        for raw_item in interleaved_stream:
            try:
                tokenized_item = prepare_single_instruction_item(
                    raw_item, self.tokenizer, self.config, self.special_tokens
                )
                if tokenized_item and "input_ids" in tokenized_item and len(tokenized_item["input_ids"]) > 0:
                    yield tokenized_item
            except Exception as e:
                print(f"Skipping item due to preparation error: {e}. Item: {raw_item}")
                continue

    @staticmethod
    def _efficient_weighted_round_robin(streams, weights):
        """High-performance weighted round-robin iterator."""
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
            except (KeyError, ValueError) as e:
                print(f"[Skipping example due to error: {e}")
                continue
            except Exception as e:
                print(f"[Unexpected error in iterator {idx_to_try}: {type(e).__name__}: {e}")
                traceback.print_exc()
                active_iterators.remove(idx_to_try)
                pool = [i for i in pool if i != idx_to_try]


class FinetuneValidationDataset(IterableDataset):
    """
    An IterableDataset for fine-tuning validation that STREAMS data.

    It streams a fixed number of samples from a validation split without
    downloading the entire file, processes them on-the-fly, and yields
    them for evaluation. This is memory and disk efficient.
    """

    def __init__(self, tokenizer: 'HFTokenizerWrapper', config: dict, special_tokens: dict):
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        self.special_tokens = special_tokens
        self.val_max_samples = self.config.get("val_max_samples", 1000)
        self.val_source = self.config.get("finetune_val_dataset", {
            "id": "databricks/databricks-dolly-15k",
            "split": "train",
            "processor": FinetuneValidationDataset.process_dolly_sample
        })
        print("Validation dataset configured. It will be streamed on-the-fly.")

    @staticmethod
    def process_dolly_sample(sample: dict) -> list:
        """Takes one raw Dolly sample and returns a list with one structured dict."""
        instruction = sample.get('instruction', '').strip()
        context = sample.get('context', '').strip()
        output = sample.get('response', '').strip()
        if not instruction or not output:
            return []
        return [{"instruction": instruction, "context": context, "output": output}]

    def __iter__(self):
        """
        This method is called by the DataLoader for each epoch (or each run).
        It sets up the stream and yields the requested number of samples.
        """
        print(f"Streaming validation data from {self.val_source['id']}")

        try:
            dataset_stream = load_dataset(
                self.val_source["id"],
                split=self.val_source.get("split", "validation"),
                streaming=True,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Could not load validation stream. Error: {e}")
            raise e

        processor_func = self.val_source["processor"]

        count = 0
        for raw_sample in tqdm(dataset_stream, total=self.val_max_samples, desc="Streaming validation samples"):
            if count >= self.val_max_samples:
                break

            structured_items = processor_func(raw_sample)

            for structured_item in structured_items:
                try:
                    tokenized_item = prepare_single_instruction_item(
                        structured_item, self.tokenizer, self.config, self.special_tokens
                    )
                    if tokenized_item and "input_ids" in tokenized_item:
                        yield tokenized_item
                        count += 1
                except Exception as e:
                    print(f"[Validation] Warning: Skipped a sample due to processing error: {e}")
                    continue
