from typing import Dict, Any

from torch.utils.data import Dataset

from src.streamers.external_streamer import PretrainDataset, PretrainValidationDataset
from src.streamers.finetune_external_streamer import FinetuneValidationDataset, FinetuneDatasetStream


class DatasetFactory:
    """A factory class to create and configure datasets based on training type."""

    def __init__(self, config: Dict[str, Any], tokenizer: Any):
        """
        Initializes the factory with all necessary components.

        Args:
            config: The main configuration dictionary.
            tokenizer: The tokenizer instance.
        """
        self.config = config
        self.tokenizer = tokenizer
        self.special_tokens = config.get('special_tokens',
                                         {"USER": "<USER>", "ASSISTANT": "<ASSISTANT>", "INST": "<INST>",
                                          "END_INST": "</INST>"})
        self.training_type = config.get('TRAINING_TYPE', 'pretrain').lower()

    def create_training_dataset(self) -> Dataset:
        """Creates an instance of the appropriate training dataset."""
        print(f"Factory: Creating TRAINING dataset for type '{self.training_type}'...")
        if self.training_type == 'pretrain':
            return PretrainDataset(tokenizer=self.tokenizer, config=self.config)

        elif self.training_type == 'finetune':
            return FinetuneDatasetStream(tokenizer=self.tokenizer, config=self.config,
                                         special_tokens=self.special_tokens)

        else:
            raise ValueError(f"Unknown TRAINING_TYPE for training dataset: '{self.training_type}'")

    def create_validation_dataset(self) -> Dataset:
        """Creates an instance of the appropriate validation dataset."""
        print(f"Factory: Creating VALIDATION dataset for type '{self.training_type}'...")
        if self.training_type == 'pretrain':
            return PretrainValidationDataset(tokenizer=self.tokenizer, config=self.config)

        elif self.training_type == 'finetune':
            return FinetuneValidationDataset(tokenizer=self.tokenizer, config=self.config,
                                             special_tokens=self.special_tokens)

        else:
            raise ValueError(f"Unknown TRAINING_TYPE for validation dataset: '{self.training_type}'")


def get_data_components(config: Dict[str, Any], tokenizer: Any) -> DatasetFactory:
    """
    A helper function to initialize the factory and get the batch processor.
    This is the only function your training script needs to import.

    Returns:
        DatasetFactory instance
    """
    factory = DatasetFactory(config, tokenizer)
    return factory
