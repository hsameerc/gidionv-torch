from pathlib import Path
from typing import List, Dict, Optional

import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset

from src.lib.core.hf_tokenizer_wrapper import HFTokenizerWrapper
from src.loaders.text_loader import IndexedJsonlDataset


class AudioLanguageDataset(Dataset):
    """
    A PyTorch Dataset for loading and pre-processing (audio, text) pairs.
    Each item in the dataset corresponds to one pair.
    """

    def __init__(self, annotations_path: str, audio_dir: str, tokenizer: HFTokenizerWrapper, sample_rate: int = 16000,
                 n_fft: int = 400, hop_length: int = 160, n_mels: int = 80):

        self.audio_dir = Path(audio_dir)
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate

        # Load all annotations into memory
        print(f"Loading annotations from {annotations_path}...")
        self.annotations = IndexedJsonlDataset(annotations_path)
        print(f"Loaded {len(self.annotations)} samples.")

        # Define the audio transformation pipeline using torchaudio
        self.audio_transform = T.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
                                                n_mels=n_mels)
        self.db_transform = T.AmplitudeToDB(stype="power", top_db=80.0)

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Optional[Dict]:
        """
        Loads and processes one sample from the dataset.
        """
        item = self.annotations[idx]

        # Process Audio
        audio_path = self.audio_dir / item['audio_file']
        audio_path = str(audio_path)
        try:
            waveform, sr = torchaudio.load(audio_path)
            # Resample if necessary
            if sr != self.sample_rate:
                resampler = T.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)

            # Apply transforms
            mel_spec = self.audio_transform(waveform)
            log_mel_spec = self.db_transform(mel_spec)

            # Normalization
            log_mel_spec = (log_mel_spec - log_mel_spec.max() + 4.0) / 4.0

            # Squeeze to remove the channel dimension if it's mono
            audio_tensor = log_mel_spec.squeeze(0)

        except (FileNotFoundError, RuntimeError) as e:
            print(f"Warning: Failed to load/process audio, skipping: {audio_path}. Error: {e}")
            return None

        # Process Text
        text_ids = self.tokenizer.encode(item['transcript'])

        return {"audio_input": audio_tensor, "text_target_ids": torch.tensor(text_ids, dtype=torch.long)}


def audio_language_collate_fn(batch: List[Optional[Dict]], pad_id: int, bos_id: int, config: dict) -> Dict:
    """
    Complete collate function for audio-language pairs, prepared for a multi-memory transformer.
    It places the audio context in the first memory slot and pads the rest.
    """
    # Filter out any samples that failed to load
    batch = [item for item in batch if item is not None]
    if not batch:
        return {}

    batch_size = len(batch)

    # Prepare Audio Input (The primary memory stream)
    audio_tensors = [item['audio_input'] for item in batch]
    audio_tensors_t = [t.transpose(0, 1) for t in audio_tensors]
    padded_audio_t = torch.nn.utils.rnn.pad_sequence(audio_tensors_t, batch_first=True, padding_value=0.0)
    padded_audio = padded_audio_t.transpose(1, 2)  # (B, Freq, Time)

    # Create the padding mask for the audio context
    audio_lengths = torch.tensor([t.shape[1] for t in audio_tensors])
    audio_padding_mask = torch.arange(padded_audio.shape[2])[None, :] < audio_lengths[:, None]

    # The AudioEncoder will convert this to (B, Time, d_model). We'll treat that as our memory stream.
    # The `forward` pass of the main model will handle the encoding step.
    memory_streams_ids = [padded_audio]
    memory_padding_masks = [audio_padding_mask]

    num_total_mem_streams = config['model']['num_memory_streams']
    if num_total_mem_streams > 1:
        # Create an empty placeholder tensor for other streams.
        # This one is tricky because it's token IDs, not embeddings yet.
        # We can pass an empty tensor of shape (batch_size, 0).
        empty_stream = torch.empty((batch_size, 0), dtype=torch.long)
        empty_mask = torch.zeros((batch_size, 0), dtype=torch.bool)

        for _ in range(num_total_mem_streams - 1):
            memory_streams_ids.append(empty_stream)
            memory_padding_masks.append(empty_mask)

    # Prepare Text Input/Target for the Decoder
    text_sequences = [item['text_target_ids'] for item in batch]
    input_ids_list = [torch.cat([torch.tensor([bos_id]), seq]) for seq in text_sequences]
    target_ids_list = [seq for seq in text_sequences]

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
    target_ids = torch.nn.utils.rnn.pad_sequence(target_ids_list, batch_first=True, padding_value=pad_id)
    target_ids[target_ids == pad_id] = -100

    # Create Final Dictionary
    return {"input_ids": input_ids, "target_ids": target_ids, "padding_mask": (input_ids != pad_id),
            "memory_streams_ids": memory_streams_ids, "memory_padding_masks": memory_padding_masks}

# Example Usage
# from functools import partial

# def main():
#     # ... setup config, tokenizer ...
#     pad_id = tokenizer.pad_token_id

#     # 1. Create the Dataset instance
#     train_dataset = AudioLanguageDataset(
#         annotations_path=config['train_annotations_path'],
#         audio_dir=config['train_audio_dir'],
#         tokenizer=tokenizer,
#         sample_rate=config['sample_rate']
#     )

#     # 2. Create the collate function
#     collate_fn = partial(audio_language_collate_fn, pad_id=pad_id)

#     # 3. Create the DataLoader
#     # This replaces your AudioDataLoader class and stream_batches method.
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=config['BATCH_SIZE'],
#         shuffle=True,
#         num_workers=4,
#         collate_fn=collate_fn
#     )

#     # 4. Use it in your training loop
#     for batch in train_loader:
#         audio = batch['audio_input'].to(device)
#         texts = batch['text_target'].to(device)
#         # ... proceed with training ...
