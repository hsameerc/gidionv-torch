import json

from torch.utils.data import DataLoader

from src.config.config import get_config
from src.lib.core.hf_tokenizer_wrapper import HFTokenizerWrapper
from src.trainer.datafactory import get_data_components
import itertools

def collect_finetune_samples(config, num_desired_samples=20, output_file="finetune_samples.jsonl"):
    """
    Collects a unique set of fine-tuning examples from a streaming dataset.
    """
    print(" Initializing Dataloader for Verification ")
    tokenizer = HFTokenizerWrapper("bpe_v2_50200_vocab.json")
    dataset_factory = get_data_components(config, tokenizer=tokenizer)
    train_dataset = dataset_factory.create_training_dataset()
    # Using a small batch size for clear inspection
    data_loader = DataLoader(train_dataset, batch_size=4, num_workers=0)
    collected = []
    print(f"Collecting {num_desired_samples} unique samples from the stream...")
    for prepared_item in itertools.islice(train_dataset, num_desired_samples):
        # The `prepared_item` is a dictionary of tensors from your dataset's __iter__
        input_ids = prepared_item['input_ids']
        target_ids = prepared_item['target_ids']
        memory_ids = prepared_item['memory_streams_ids']
        # Decoding the tensors back to text
        input_text = tokenizer.decode([t for t in input_ids.tolist() if t != tokenizer.pad_token_id])
        target_tokens = [t for t in target_ids.tolist() if t != -100 and t != tokenizer.pad_token_id]
        target_text = tokenizer.decode(target_tokens)
        memory_texts = []
        # memory_ids has shape (num_streams, seq_len)
        for stream_ids in memory_ids:
            decoded = tokenizer.decode([t for t in stream_ids.tolist() if t != tokenizer.pad_token_id])
            if decoded.strip():
                memory_texts.append(decoded)
            else:
                memory_texts.append('N/A')

        # Adding the fully decoded sample to our collection
        collected.append({
            "input_prompt": input_text.strip(),
            "memory_context": memory_texts,
            "target_output": target_text.strip()
        })

    #  Saving to file (your logic is perfect)
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in collected:
            json.dump(sample, f, ensure_ascii=False, indent=2)  # Added indent for readability
            f.write("\n")

    print(f"\n✅ Collected {len(collected)} fine-tuning samples saved to '{output_file}'")


def verify_pretrain_dataloader(config, num_batches_to_check=50):
    """
    Loads multiple batches from the pre-training dataloader until it finds
    a sample with populated memory streams and prints a detailed analysis.
    """
    print(" Initializing Dataloader for Verification ")
    tokenizer = HFTokenizerWrapper("bpe_v2_50200_vocab.json")

    dataset_factory = get_data_components(config, tokenizer=tokenizer)
    train_dataset = dataset_factory.create_training_dataset()

    # Use a small batch size for clear inspection
    data_loader = DataLoader(train_dataset, batch_size=2, num_workers=0)

    print(f"\n Fetching up to {num_batches_to_check} batches to find a multi-stream sample... ")

    found_sample = None
    for i, batch in enumerate(data_loader):
        if i >= num_batches_to_check:
            print("\nCould not find a multi-stream sample within the first 50 batches.")
            # Use the last valid batch we found
            first_batch = batch
            break

        # Check the first sample in the batch
        memory_ids = batch['memory_streams_ids'][0]
        memory_masks = (memory_ids != tokenizer.pad_token_id)

        # Check if more than one stream has real tokens
        # .any(dim=1) checks if there is any True value along the sequence length dimension
        # .sum() counts how many streams are not empty
        if memory_masks.any(dim=1).sum().item() > 1:
            print(f"\nFound a multi-stream sample at batch #{i + 1}!")
            first_batch = batch
            break
    else:  # This 'else' belongs to the 'for' loop, runs if the loop finishes without break
        print("\nCould not find a multi-stream sample. Analyzing the last batch found.")
        first_batch = batch  # Fallback to the last batch

    print("\n" + "=" * 20 + " BATCH VERIFICATION REPORT " + "=" * 20)

    # Check Shapes
    print("\n[1. TENSOR SHAPES]")
    input_ids = first_batch['input_ids']
    target_ids = first_batch['target_ids']
    memory_ids = first_batch['memory_streams_ids']

    print(f"  - input_ids shape:          {input_ids.shape}")
    print(f"  - target_ids shape:         {target_ids.shape}")
    print(f"  - memory_streams_ids shape: {memory_ids.shape}")

    # Inspect a Single Sample (the first one in the batch)
    print("\n[2. ANALYSIS OF SAMPLE #0]")

    sample_input_ids = input_ids[0]
    sample_target_ids = target_ids[0]
    sample_memory_ids = memory_ids[0]

    # Decoding the main input and target
    decoded_input = tokenizer.decode([t for t in sample_input_ids.tolist() if t != tokenizer.pad_token_id])
    target_tokens = [t for t in sample_target_ids.tolist() if t != -100 and t != tokenizer.pad_token_id]
    decoded_target = tokenizer.decode(target_tokens)

    print("\n Main Input/Target ")
    print(f"  DECODED INPUT:  '{decoded_input[:150]}...'")
    print(f"  DECODED TARGET: '{decoded_target[:150]}...'")

    # Decoding ALL memory streams
    print("\n Memory Streams ")
    for i, stream_ids in enumerate(sample_memory_ids):
        decoded_stream = tokenizer.decode([t for t in stream_ids.tolist() if t != tokenizer.pad_token_id])
        if decoded_stream:
            print(f"  STREAM {i}: '{decoded_stream[:150]}...'")
        else:
            print(f"  STREAM {i}: [EMPTY/PADDING]")

    print("\n" + "=" * 20 + " VERIFICATION COMPLETE " + "=" * 20)


if __name__ == "__main__":
    cfg = get_config()
    with open("test-config.json", 'r') as f:
        cfg.update(json.load(f))
    cfg.update({"TRAINING_TYPE": "finetune"})
    cfg.update({"TRAINING_SOURCE": "online"})
    collect_finetune_samples(cfg, num_desired_samples=1000, output_file="finetune_samples.jsonl")
    # verify_pretrain_dataloader(cfg)
