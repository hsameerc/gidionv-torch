from typing import Dict, Any

from torch.utils.data import Dataset

from src.streamers.datasets import StreamLocalPretrainDataset, FinetuneLocalDataset
from src.streamers.finetune_streamer import FinetuneDatasetStream, FinetuneValidationDataset
from src.streamers.pretrain_streamer import PretrainDatasetStreamer, PretrainValidationDataset


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
        self.training_source = config.get('TRAINING_SOURCE', 'online').lower()

    def create_training_dataset(self) -> Dataset:
        """Creates an instance of the appropriate training dataset."""
        print(
            f"Factory: Creating TRAINING dataset for type '{self.training_type}' and source '{self.training_source}'...")

        if self.training_source == 'online':
            dataset_map = {
                'pretrain': lambda: PretrainDatasetStreamer(tokenizer=self.tokenizer, config=self.config),
                'finetune': lambda: FinetuneDatasetStream(tokenizer=self.tokenizer, config=self.config,
                                                          special_tokens=self.special_tokens)
            }

        elif self.training_source == 'local':
            dataset_map = {
                'pretrain': lambda: StreamLocalPretrainDataset(filepath=self.config["TRAIN_FILE_PATH"],
                                                               config=self.config, tokenizer=self.tokenizer),
                'finetune': lambda: FinetuneLocalDataset(filepath=self.config["TRAIN_FILE_PATH"], config=self.config,
                                                         tokenizer=self.tokenizer,
                                                         special_tokens=self.special_tokens)
            }

        else:
            raise ValueError(f"Unknown TRAINING_SOURCE: '{self.training_source}'")

        if self.training_type not in dataset_map:
            raise ValueError(f"Unknown TRAINING_TYPE: '{self.training_type}'")

        return dataset_map[self.training_type]()

    def create_validation_dataset(self) -> Dataset:
        """Creates an instance of the appropriate validation dataset."""
        print(
            f"Factory: Creating VALIDATION dataset for type '{self.training_type}' and source '{self.training_source}'...")

        if self.training_source == 'online':
            dataset_map = {
                'pretrain': lambda: PretrainValidationDataset(tokenizer=self.tokenizer, config=self.config),
                'finetune': lambda: FinetuneValidationDataset(tokenizer=self.tokenizer, config=self.config,
                                                              special_tokens=self.special_tokens)
            }

        elif self.training_source == 'local':
            dataset_map = {
                'pretrain': lambda: StreamLocalPretrainDataset(filepath=self.config["VAL_FILE_PATH"],
                                                               config=self.config, tokenizer=self.tokenizer),
                'finetune': lambda: FinetuneLocalDataset(filepath=self.config["VAL_FILE_PATH"], config=self.config,
                                                         tokenizer=self.tokenizer,
                                                         special_tokens=self.special_tokens)
            }
        else:
            raise ValueError(f"Unknown TRAINING_SOURCE: '{self.training_source}'")

        if self.training_type not in dataset_map:
            raise ValueError(f"Unknown TRAINING_TYPE for validation dataset: '{self.training_type}'")

        return dataset_map[self.training_type]()


def get_data_components(config: Dict[str, Any], tokenizer: Any) -> DatasetFactory:
    """
    A helper function to initialize the factory and get the batch processor.
    This is the only function your training script needs to import.

    Returns:
        DatasetFactory instance
    """
    factory = DatasetFactory(config, tokenizer)
    return factory
