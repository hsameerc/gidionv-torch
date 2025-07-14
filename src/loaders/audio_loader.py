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


def audio_language_collate_fn(batch: List[Optional[Dict]], pad_id: int, bos_id: int, eos_id: int, config: dict) -> Dict:
    """
    Prepares and collates a batch of audio-language data for a transformer model.

    This function takes a list of samples (each a dictionary containing an audio spectrogram and
    tokenized text), and formats them into tensors ready for model training. It handles:
    - Padding variable-length audio spectrograms to the same time dimension.
    - Creating encoder padding masks for the audio features.
    - Preparing decoder `input_ids` and `target_ids` for teacher-forcing.
    - Padding all text sequences to the same length within the batch.
    - Setting up placeholders for a multi-memory architecture.

    Args:
        batch: A list of dictionary samples from the Dataset. Each dict should have
               'audio_input' (Tensor of shape [Freq, Time]) and 'text_target_ids' (Tensor).
        pad_id: The token ID for padding.
        bos_id: The token ID for "beginning of sentence".
        eos_id: The token ID for "end of sentence".
        config: A configuration dictionary containing model and data parameters.

    Returns:
        A dictionary of tensors ready for the model's forward pass.
    """
    # Filtering out any samples that failed to load
    batch = [item for item in batch if item is not None]
    if not batch:
        return {}

    batch_size = len(batch)

    # Preparing Audio Inputs
    # Audio tensors have variable length in the time dimension.
    audio_tensors = [item['audio_input'] for item in batch]  # List of (Freq, Time) tensors
    audio_lengths = torch.tensor([t.shape[1] for t in audio_tensors])

    # To use pad_sequence on the time dimension, we must temporarily make it the first dimension.
    # (Freq, Time) -> (Time, Freq)
    audio_tensors_t = [t.transpose(0, 1) for t in audio_tensors]
    padded_audio_t = torch.nn.utils.rnn.pad_sequence(audio_tensors_t, batch_first=True, padding_value=0.0)

    # Transpose back to the standard (Batch, Freq, Time) format
    padded_audio = padded_audio_t.transpose(1, 2)

    # Creating a boolean mask for the audio encoder. True where there is real data, False where there is padding.
    max_audio_len = padded_audio.shape[2]
    audio_padding_mask = torch.arange(max_audio_len)[None, :] < audio_lengths[:, None]

    # Preparing Text Inputs and Targets for Teacher Forcing ---
    # The key 'text_target_ids' comes from your dataset implementation.
    text_sequences = [item['text_target_ids'] for item in batch]

    input_ids_list = [torch.cat([torch.tensor([bos_id]), seq]) for seq in text_sequences]
    target_ids_list = [torch.cat([seq, torch.tensor([eos_id])]) for seq in text_sequences]

    # Padding both lists of tensors to the maximum length in the batch.
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
    target_ids = torch.nn.utils.rnn.pad_sequence(target_ids_list, batch_first=True, padding_value=pad_id)

    # The loss function should ignore padded tokens in the targets.
    target_ids[target_ids == pad_id] = -100

    # Creating the decoder's padding mask to ignore padded tokens in attention.
    decoder_padding_mask = (input_ids != pad_id)

    # Assembling Final Batch Dictionary
    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "input_ids_padding_mask": decoder_padding_mask,
        "audio_input": padded_audio,
        "audio_input_padding_masks": audio_padding_mask,
    }
