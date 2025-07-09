from torch.utils.data import IterableDataset, Dataset

from src.lib.core.hf_tokenizer_wrapper import HFTokenizerWrapper
from src.loaders.text_loader import TextLoaderStream, AdvancedDataStreamer, IndexedJsonlDataset
from src.utils.prepare import prepare_single_pretrain_item, prepare_single_instruction_item


class StreamerDataset(IterableDataset):
    def __init__(self, config, tokenizer: HFTokenizerWrapper, validation_stream: bool = False):
        self.config = config
        self.tokenizer = tokenizer
        self.validation_stream = validation_stream

    def __iter__(self):
        if self.validation_stream:
            loader_path = self.config['VAL_FILE_PATH']
            overlap_length = 0
        else:
            loader_path = self.config['TRAIN_FILE_PATH']
            overlap_length = self.config.get('OVERLAP_LEN_TOKENS', 64)
        stream = TextLoaderStream(loader_path)
        streamer = AdvancedDataStreamer(text_stream=stream, tokenizer=self.tokenizer,
                                        seq_len=self.config['max_seq_len'], overlap_len_tokens=overlap_length, )
        for item in streamer.stream_data(shuffle=True):
            yield prepare_single_pretrain_item(item, self.tokenizer, self.config)


class FinetuneDataset(Dataset):
    """
    A map-style Dataset for fine-tuning on structured .jsonl files.
    It uses an IndexedJsonlDataset for efficient random access.
    """

    def __init__(self, filepath: str, tokenizer: 'HFTokenizerWrapper', config: dict, special_tokens: dict):
        super().__init__()
        self.indexed_data = IndexedJsonlDataset(filepath)
        self.tokenizer = tokenizer
        self.config = config
        self.special_tokens = special_tokens

    def __len__(self) -> int:
        return len(self.indexed_data)

    def __getitem__(self, index: int) -> dict:
        raw_item = self.indexed_data[index]
        return prepare_single_instruction_item(raw_item, self.tokenizer, self.config, self.special_tokens)
