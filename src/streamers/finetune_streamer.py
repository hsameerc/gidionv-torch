import itertools
import random
import re
import traceback
from typing import Iterable, Dict

import torch
from datasets import load_dataset, tqdm
from math import ceil
from torch.utils.data import IterableDataset

from src.lib.core.hf_tokenizer_wrapper import HFTokenizerWrapper
from src.loaders.finetune_loader import prepare_single_instruction_item


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
                "weight": 0.10,
                "processor": self._process_web_questions,
                "slot": -1,
            },
            "natural_questions": {
                "id": "google-research-datasets/natural_questions",
                "split": "train",
                "weight": 0.20,
                "processor": self._process_natural_questions,
                "slot": -1,
            },
            "alpaca": {
                "id": "yahma/alpaca-cleaned",
                "split": "train",
                "weight": 0.10,
                "processor": self._process_alpaca,
                "slot": 0,
            },
            "dolly": {
                "id": "databricks/databricks-dolly-15k",
                "split": "train",
                "weight": 0.10,
                "processor": self._process_dolly,
                "slot": 0,
            },
            "squad_v2": {
                "id": "squad_v2",
                "split": "train",
                "weight": 0.5,
                "processor": self._process_squad,
                "slot": 0,
            },
            "trivia_qa": {
                "id": "trivia_qa", "name": "rc.nocontext",
                "split": "train",
                "weight": 0.5,
                "processor": self._process_trivia_qa,
                "slot": -1,
            },
            "chain_of_thought": {
                "id": "AlekseyKorshuk/chain-of-thoughts-chatml",
                "split": "train",
                "weight": 0.20,
                "processor": self._process_cot,
                "slot": -1,
            },
            "math_x_5m": {
                "id": "XenArcAI/MathX-5M",
                "split": "train",
                "weight": 0.10,
                "processor": self._process_math_sample,
                "slot": 2,
            },
            "hugging_face_tb": {
                "id": "HuggingFaceTB/everyday-conversations-llama3.1-2k",
                "split": "train_sft",
                "weight": 0.10,
                "processor": self._process_everyday_convo_from_messages,
                "slot": 1,
            },
        }

    @staticmethod
    def _process_everyday_convo_from_messages(example: dict, special_tokens:dict) -> Iterable[Dict]:
        """
        Processes multi-turn 'messages' format using special tokens.
        """

        messages = example.get("messages", [])
        if not messages or len(messages) < 2:
            return

        # Get the special tokens from the dictionary, with fallbacks
        user_token = special_tokens.get("USER", "<USER>")
        assistant_token = special_tokens.get("ASSISTANT", "<ASSISTANT>")

        for i in range(1, len(messages)):
            current_turn = messages[i]
            previous_turn = messages[i - 1]

            if current_turn.get("role") == "assistant" and previous_turn.get("role") == "user":

                instruction = previous_turn.get("content", "").strip()

                history_turns = messages[:i - 1]
                context_parts = []
                for msg in history_turns:
                    role = msg.get("role", "")
                    content = msg.get("content", "").strip()
                    if role and content:
                        # Use the special tokens instead of plain text
                        if role == "user":
                            prefix = user_token
                        elif role == "assistant":
                            prefix = assistant_token
                        else:
                            prefix = f"{role.title()}:"  # Fallback for other roles

                        context_parts.append(f"{prefix} {content}")

                context = "\n".join(context_parts).strip()
                output = current_turn.get("content", "").strip()

                if instruction and output:
                    yield {
                        "instruction": instruction,
                        "context": context,
                        "output": output
                    }

    @staticmethod
    def _process_math_sample(example: dict, special_tokens: dict) -> Iterable[Dict]:
        """
        Processes a sample from a math reasoning dataset.
        """
        inst_token = special_tokens.get("INST", "<INST>")
        end_inst_token = special_tokens.get("END_INST", "</INST>")

        problem = example.get('problem', '').strip()
        answer = example.get('expected_answer', '').strip()
        generated_solution = example.get('generated_solution', '').strip()

        if not problem or not answer:
            return

        clean_solution =  generated_solution.lstrip("<think>").strip()
        formatted_cot_context = f"{inst_token}{clean_solution}{end_inst_token}"
        instruction = problem
        context = formatted_cot_context
        output = answer

        yield {
            "instruction": instruction,
            "context": context,
            "output": output
        }

    @staticmethod
    def _process_web_questions(example: dict, special_tokens:dict) -> Iterable[Dict]:
        if example.get('answers') and example['answers']:
            instruction = example['question'].strip()
            output = example['answers'][0].strip()
            if instruction and output:
                yield {"instruction": instruction, "context": "", "output": output}

    @staticmethod
    def _process_natural_questions(example: dict, special_tokens:dict) -> Iterable[Dict]:
        try:
            short_answers = example['annotations']['short_answers']
            if short_answers and short_answers[0]['text']:
                instruction = example['question']['text'].strip()
                output = short_answers[0]['text'][0].strip()
                if instruction and output:
                    yield {"instruction": instruction, "context": "", "output": output}
        except (KeyError, IndexError):
            pass

    @staticmethod
    def _process_alpaca(example: dict, special_tokens:dict) -> Iterable[Dict]:
        inst_token = special_tokens.get("INST", "<INST>")
        end_inst_token = special_tokens.get("END_INST", "</INST>")

        instruction = example.get('instruction', '').strip()
        inp = example.get('input', '').strip()
        formatted_context = f"{inp}"
        output = example.get('output', '').strip()
        if instruction and output:
            yield {"instruction": instruction, "context": formatted_context, "output": output}

    @staticmethod
    def _process_dolly(example: dict, special_tokens:dict) -> Iterable[Dict]:
        inst_token = special_tokens.get("INST", "<INST>")
        end_inst_token = special_tokens.get("END_INST", "</INST>")

        instruction = example.get('instruction', '').strip()
        context = example.get('context', '').strip()
        formatted_context = f"{context}"
        output = example.get('response', '').strip()
        if instruction and output:
            yield {"instruction": instruction, "context": formatted_context, "output": output}

    @staticmethod
    def _process_squad(example: dict, special_tokens:dict) -> Iterable[Dict]:
        inst_token = special_tokens.get("INST", "<INST>")
        end_inst_token = special_tokens.get("END_INST", "</INST>")

        """Processes a single example from the SQuAD v2 dataset."""
        context = example.get('context', '').strip()
        formatted_context = f"{context}"
        question = example.get('question', '').strip()
        answers = example.get('answers', {}).get('text', [])
        if context and question and answers:
            yield {"instruction": question, "context": formatted_context, "output": answers[0].strip()}

    @staticmethod
    def _process_trivia_qa(example: dict, special_tokens:dict) -> Iterable[Dict]:
        """Processes a single example from the TriviaQA dataset."""
        question = example.get('question', '').strip()
        answer = example.get('answer', {}).get('value', '').strip()
        if question and answer:
            yield {"instruction": question, "context": "", "output": answer}

    @staticmethod
    def _process_cot(example: dict, special_tokens:dict) -> Iterable[Dict]:
        """Processes a conversation from the Chain-of-Thoughts dataset."""
        conversation = example.get('conversation', [])
        for i in range(0, len(conversation) - 1, 2):
            user_msg = conversation[i]
            assistant_msg = conversation[i + 1]
            if not assistant_msg.get('do_train', False):
                continue
            if (
                    user_msg.get('role', '').lower() == 'user' and
                    assistant_msg.get('role', '').lower() == 'assistant'
            ):
                instruction = user_msg.get('content', '').strip()
                output = assistant_msg.get('content', '').strip()

                if instruction and output:
                    yield {
                        "instruction": instruction,
                        "context": "",
                        "output": output
                    }

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        print(f"[Worker {worker_id}] Initializing fine-tuning streaming pipeline...")

        stream_factories = []
        probabilities = []

        for source_name, source_info in self.data_sources.items():
            try:
                def make_stream_fn(source_info=source_info):
                    ds = load_dataset(
                        path=source_info["id"],
                        name=source_info.get("name"),
                        split=source_info["split"],
                        streaming=True,
                    )
                    for structured_item in (item for raw  in ds for item in source_info["processor"](raw, self.special_tokens)):
                        structured_item['mem_slot'] = source_info.get('slot', -1)
                        yield structured_item

                stream_factories.append(make_stream_fn)
                probabilities.append(source_info["weight"])
                print(f"[Worker {worker_id}] Initialized stream: {source_name}")

            except Exception as e:
                print(f"[Worker {worker_id}] FAILED to load stream {source_name}. Skipping. Error: {e}")
                traceback.print_exc()
                continue

        if not stream_factories:
            raise RuntimeError("Fatal: No fine-tuning dataset streams could be initialized.")

        total_prob = sum(probabilities)
        probabilities = [p / total_prob for p in probabilities]

        interleaved_structured_stream = self._efficient_weighted_round_robin(
            stream_factories, probabilities, worker_info
        )

        for raw_item in interleaved_structured_stream:
            try:
                yield prepare_single_instruction_item(
                    raw_item, self.tokenizer, self.config, self.special_tokens
                )
            except Exception as e:
                print(f"Skipping item due to preparation error: {e}. Item: {raw_item}")
                continue

    @staticmethod
    def _efficient_weighted_round_robin(stream_factories, weights, worker_info):
        """A high-performance weighted round-robin iterator with restartable streams."""
        iterators = [iter(factory()) for factory in stream_factories]
        weight_counts = [max(1, ceil(w * 1000)) for w in weights]
        pool = list(itertools.chain.from_iterable([[i] * wc for i, wc in enumerate(weight_counts)]))

        if worker_info is not None:
            random.seed(worker_info.seed)

        active_iterators = list(range(len(iterators)))

        while active_iterators:
            if not pool:
                pool = list(itertools.chain.from_iterable([[i] * weight_counts[i] for i in active_iterators]))
                random.shuffle(pool)

            idx_to_try = pool.pop()
            try:
                yield next(iterators[idx_to_try])
            except StopIteration:
                print(f"[Worker {worker_info.id if worker_info else 0}] Stream {idx_to_try} exhausted. Restarting...")
                try:
                    iterators[idx_to_try] = iter(stream_factories[idx_to_try]())
                    yield next(iterators[idx_to_try])
                except Exception as e:
                    print(f"[Stream {idx_to_try}] Failed to restart. Removing. Reason: {e}")
                    active_iterators.remove(idx_to_try)
                    pool = [i for i in pool if i != idx_to_try]
            except Exception as e:
                print(f"[Stream {idx_to_try}] Skipping due to exception: {e}")
                continue


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
                # trust_remote_code=True
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
