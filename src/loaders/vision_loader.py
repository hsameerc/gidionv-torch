from pathlib import Path
from typing import List, Dict, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.lib.core.hf_tokenizer_wrapper import HFTokenizerWrapper
from src.loaders.text_loader import IndexedJsonlDataset


class VisionLanguageDataset(Dataset):
    """
    A PyTorch Dataset for loading and pre-processing (image, text) pairs.
    Each item in the dataset corresponds to one pair.
    """

    def __init__(self, annotations_path: str, image_dir: str, tokenizer: HFTokenizerWrapper, image_size: int = 224):

        self.image_dir = Path(image_dir)
        self.tokenizer = tokenizer

        # Load all annotations into memory. For very large datasets, you might
        # use a memory-mapped file or a database, but this is standard practice.
        print(f"Loading annotations from {annotations_path}...")
        self.annotations_dataset = IndexedJsonlDataset(annotations_path)
        print(f"Loaded {len(self.annotations_dataset)} samples.")

        #  Define the image transformation pipeline using torchvision
        self.transform = transforms.Compose(
            [transforms.Resize((image_size, image_size), antialias=True), transforms.ToTensor(),
             # Converts PIL image to (C, H, W) tensor and scales to [0, 1]
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalizes to [-1, 1]
             ])

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.annotations_dataset)

    def __getitem__(self, idx: int) -> Optional[Dict]:
        """
        Loads and processes one sample from the dataset.

        Returns:
            A dictionary containing the processed image and tokenized text,
            or None if the image file is not found.
        """
        item = self.annotations_dataset[idx]

        # Process Image
        image_path = self.image_dir / item['image_file']
        try:
            # Open image and ensure it's in RGB
            with Image.open(image_path).convert("RGB") as img:
                image_tensor = self.transform(img)
        except FileNotFoundError:
            print(f"Warning: Image file not found, skipping: {image_path}")
            # Returning None is one way to handle missing data; the collate_fn must filter these out.
            return None

        # Process Text
        # Tokenize the text but don't pad it yet. The collate_fn will handle padding.
        text_ids = self.tokenizer.encode(item['text'])

        return {"image_input": image_tensor, "text_input_ids": torch.tensor(text_ids, dtype=torch.long)}


def vision_language_collate_fn(batch: List[Optional[Dict]], pad_id: int, bos_id: int, config: dict) -> Dict:
    """
    Complete collate function for vision-language data, prepared for a multi-memory transformer.
    It places the vision context in the first memory slot and pads the rest.
    """
    # Filter out any samples that failed to load
    batch = [item for item in batch if item is not None]
    if not batch:
        return {}

    batch_size = len(batch)

    # Prepare Vision Input (The primary memory stream)
    image_tensors = torch.stack([item['image_input'] for item in batch], dim=0)

    # For a ViT, the sequence length is fixed. The mask is all True.
    num_patches_plus_one = image_tensors.shape[1]  # ViT output has shape (B, num_patches+1, D)
    vision_padding_mask = torch.ones(batch_size, num_patches_plus_one, dtype=torch.bool)

    # Create the list of memory streams and masks
    memory_streams_ids = [image_tensors]
    memory_padding_masks = [vision_padding_mask]

    num_total_mem_streams = config['model']['num_memory_streams']
    if num_total_mem_streams > 1:
        # Create an empty placeholder tensor for the other streams.
        # It needs batch_size and d_model, but seq_len can be 0.
        # The model's cross-attention should handle an empty sequence gracefully.
        d_model = image_tensors.shape[-1]
        empty_stream = torch.empty((batch_size, 0, d_model), dtype=image_tensors.dtype)
        empty_mask = torch.zeros((batch_size, 0), dtype=torch.bool)

        for _ in range(num_total_mem_streams - 1):
            memory_streams_ids.append(empty_stream)
            memory_padding_masks.append(empty_mask)

    # Prepare Text Input/Target for the Decoder
    text_sequences = [item['text_input_ids'] for item in batch]
    input_ids_list = [torch.cat([torch.tensor([bos_id]), seq]) for seq in text_sequences]
    target_ids_list = [seq for seq in text_sequences]

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
    target_ids = torch.nn.utils.rnn.pad_sequence(target_ids_list, batch_first=True, padding_value=pad_id)
    target_ids[target_ids == pad_id] = -100

    #  Create Final Dictionary
    return {"input_ids": input_ids, "target_ids": target_ids, "padding_mask": (input_ids != pad_id),
        "memory_streams_ids": memory_streams_ids, "memory_padding_masks": memory_padding_masks}

# Example Usage
# from functools import partial

# def main():
#     # ... setup config, tokenizer ...
#     pad_id = tokenizer.pad_token_id

#     # 1. Create the Dataset instance
#     train_dataset = VisionLanguageDataset(
#         annotations_path=config['train_annotations_path'],
#         image_dir=config['train_image_dir'],
#         tokenizer=tokenizer,
#         image_size=config['image_size']
#     )

#     # 2. Create the collate function with the pad_id baked in
#     collate_fn = partial(vision_language_collate_fn, pad_id=pad_id)

#     # 3. Create the DataLoader
#     # This replaces your entire VisionDataLoader class and stream_batches method.
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=config['BATCH_SIZE'],
#         shuffle=True,  # DataLoader can shuffle the data every epoch
#         num_workers=4, # Use multiple processes to load data in parallel
#         collate_fn=collate_fn
#     )

#     # 4. Use it in your training loop
#     for batch in train_loader:
#         images = batch['image_input'].to(device)
#         texts = batch['text_input'].to(device)
#         # ... proceed with training ...
