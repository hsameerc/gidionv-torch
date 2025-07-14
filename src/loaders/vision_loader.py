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


def vision_language_collate_fn(batch: List[Optional[Dict]], pad_id: int, bos_id: int, eos_id: int, config: dict) -> Dict:
    """
    Prepares and collates a batch of vision-language data for a transformer model.

    This function takes a list of samples (each a dictionary containing an image and tokenized text),
    and formats them into tensors ready for model training. It handles:
    - Stacking image tensors.
    - Creating encoder padding masks for vision features.
    - Preparing decoder `input_ids` and `target_ids` for teacher-forcing.
    - Padding all sequences to the same length within the batch.
    - Setting up placeholders for a multi-memory architecture.

    Args:
        batch: A list of dictionary samples from the Dataset. Each dict should have
               'image_input' (Tensor) and 'text_input_ids' (Tensor).
        pad_id: The token ID for padding.
        bos_id: The token ID for "beginning of sentence".
        eos_id: The token ID for "end of sentence".
        config: A configuration dictionary containing model and data parameters.

    Returns:
        A dictionary of tensor ready for the model's forward pass.
    """
    # Filtering out any samples that failed to load (e.g., due to a missing image file)
    batch = [item for item in batch if item is not None]
    if not batch:
        return {}

    batch_size = len(batch)

    # Preparing Vision Inputs
    image_tensors = torch.stack([item['image_input'] for item in batch], dim=0)

    # Calculating the sequence length of the ViT output (patches + CLS token)
    vision_config = config['vision_encoder']
    patch_size = vision_config['patch_size']
    image_size = vision_config['image_size']
    num_patches = (image_size // patch_size) ** 2
    vision_seq_len = num_patches + 1

    # Vision features are never padded, so their mask is all True.
    vision_padding_mask = torch.ones(batch_size, vision_seq_len, dtype=torch.bool)

    # Prepare Text Inputs and Targets for Teacher Forcing
    text_sequences = [item['text_input_ids'] for item in batch]

    input_ids_list = [torch.cat([torch.tensor([bos_id]), seq]) for seq in text_sequences]
    target_ids_list = [torch.cat([seq, torch.tensor([eos_id])]) for seq in text_sequences]

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
    target_ids = torch.nn.utils.rnn.pad_sequence(target_ids_list, batch_first=True, padding_value=pad_id)

    # The loss function should ignore padded tokens in the targets.
    # PyTorch's CrossEntropyLoss ignores targets with the value -100.
    target_ids[target_ids == pad_id] = -100

    # Creating the decoder's padding mask to ignore padded tokens in the input.
    decoder_padding_mask = (input_ids != pad_id)

    # Assemble Final Batch Dictionary
    final_res = {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "input_padding_mask": decoder_padding_mask,
        "image_input": image_tensors,
        "image_input_padding_masks": vision_padding_mask,
    }
    return final_res
