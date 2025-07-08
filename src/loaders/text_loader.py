import atexit
import csv
import json
import mmap
import os
import random
from pathlib import Path
from typing import Generator
from typing import List, Dict, Optional

from torch.utils.data import Dataset

from src.lib.core.hf_tokenizer_wrapper import HFTokenizerWrapper


def log_to_csv(writer: csv.DictWriter, data_dict: Dict):
    """
    Helper to write a dictionary row using a DictWriter.
    """
    writer.writerow(data_dict)


def split_large_text_file(input_path, train_path, val_path, split_ratio=0.99):
    """
    Splits a large text file into training and validation sets without
    loading the whole file into memory.
    """
    print(f"Splitting '{input_path}'...")
    total_size = os.path.getsize(input_path)
    split_point = int(total_size * split_ratio)

    buffer_size = 10 * 1024 * 1024  # Read in 10MB chunks

    # Write training file
    print(f"Writing training data to '{train_path}'...")
    with open(input_path, 'rb') as fin, open(train_path, 'wb') as fout:
        bytes_written = 0
        while bytes_written < split_point:
            chunk = fin.read(min(buffer_size, split_point - bytes_written))
            if not chunk:
                break
            fout.write(chunk)
            bytes_written += len(chunk)

    # Write validation file
    print(f"Writing validation data to '{val_path}'...")
    with open(input_path, 'rb') as fin, open(val_path, 'wb') as fout:
        fin.seek(split_point)
        while True:
            chunk = fin.read(buffer_size)
            if not chunk:
                break
            fout.write(chunk)

    print("File split complete.")


class TextLoaderStream:
    """
    Handles loading large text files efficiently using memory-mapping.
    This allows treating a large file on disk as if it were a string in memory
    without actually loading the entire file into RAM. The operating system
    manages paging the data from disk as needed.
    """

    def __init__(self, filename: str, encoding: str = 'utf-8'):
        self.filename = filename
        self.encoding = encoding
        self._file = None
        self._mm = None

        if not os.path.exists(filename):
            raise FileNotFoundError(f"The specified file does not exist: {filename}")

        try:
            self._file = open(self.filename, "r", encoding=self.encoding)
            self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
            print(f"Successfully memory-mapped file: {self.filename} ({len(self._mm) / 1e9:.2f} GB)")
        except Exception as e:
            if self._mm: self._mm.close()
            if self._file: self._file.close()
            raise IOError(f"Failed to memory-map file {filename}: {e}") from e

    @property
    def raw_text(self):
        """
        Returns the memory-map object directly.
        This object behaves like a byte string but is not loaded into RAM.
        Slicing this object reads directly from the disk-backed map.
        """
        if self._mm is None:
            raise RuntimeError("File is not open or memory-map is closed.")
        return self._mm

    def __len__(self):
        """Returns the total number of bytes in the file."""
        return len(self.raw_text) if self._mm else 0

    def close(self):
        """Closes the memory-map and the file."""
        if self._mm: self._mm.close()
        if self._file: self._file.close()
        self._mm, self._file = None, None

    def __del__(self):
        """Ensure resources are cleaned up when the object is garbage-collected."""
        self.close()


class IndexedJsonlDataset(Dataset):
    """
    A high-performance, memory-efficient PyTorch Dataset for very large .jsonl files.

    This class creates an index of byte offsets for each line in the file, allowing for
    fast, random access to any data point. It is designed to be used with a
    PyTorch `DataLoader` and multiple workers, where each worker will keep its own
    file handle open to avoid repeated open/close overhead.
    """

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")

        self._line_offsets: List[int] = []

        # File handle, one per worker process
        # This will be initialized lazily in __getitem__ to be compatible
        # with PyTorch's multiprocessing DataLoader.
        self._file_handle = None

        print(f"Indexing large JSONL file: {self.filepath}...")
        self._build_index()
        print(f"Indexing complete. Found {len(self)} lines.")

        # Ensure file handles are closed when the main Python process exits
        atexit.register(self.close)

    def _build_index(self):
        """
        Scans the file once to build a byte offset index for each line.
        This is a more efficient implementation.
        """
        with self.filepath.open('rb') as f:
            self._line_offsets.append(0)
            while True:
                line = f.readline()
                if not line:
                    break
                self._line_offsets.append(f.tell())

        self._line_offsets.pop()

    def __len__(self) -> int:
        """Returns the total number of lines (samples) in the file."""
        return len(self._line_offsets)

    def __getitem__(self, index: int) -> Dict:
        """
        Retrieves and parses a single JSON object by its line index.
        It lazily opens a file handle for each worker process.
        """
        if not 0 <= index < len(self):
            raise IndexError(f"Index {index} is out of range for file with {len(self)} lines.")

        # Each worker process gets its own file handle, which stays open.
        if self._file_handle is None:
            self._file_handle = self.filepath.open('rb')

        # Seek to the pre-computed byte offset and read the line
        self._file_handle.seek(self._line_offsets[index])
        line_bytes = self._file_handle.readline()

        # Decode and parse the JSON line
        return json.loads(line_bytes.decode('utf-8'))

    def close(self):
        """Closes the file handle."""
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None

    def __del__(self):
        """Destructor to ensure file handle is closed when the object is destroyed."""
        self.close()


class AdvancedDataStreamer:
    """
    Processes a stream of text from a large, memory-mapped file into tokenized
    {"source_ids": ..., "context_ids": ...} pairs suitable for training.

    This is the core logic engine for a multi-worker data loading pipeline,
    replacing the flawed logic in the original PretrainingDataLoader.
    """

    def __init__(self, text_stream: 'TextLoaderStream', tokenizer: 'HFTokenizerWrapper', seq_len: int,
                 overlap_len_tokens: Optional[int] = 0, chunk_size_bytes: Optional[int] = 10 * 1024 * 1024):
        """
        Args:
            text_stream: instance of TextLoaderStream
            tokenizer: A tokenizer instance with `.encode()` and `.eos_token_id`.
            seq_len: The sequence length for BOTH the context and source parts.
            overlap_len_tokens: Number of tokens to overlap between full examples.
                                A value of 0 means non-overlapping examples.
            chunk_size_bytes: The size of byte chunks to read from the file.
        """
        self.text_stream = text_stream
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.example_len = 2 * seq_len
        self.step_len = self.example_len - overlap_len_tokens

        if self.step_len <= 0:
            raise ValueError("overlap_len_tokens must be smaller than the full example length (2 * seq_len).")

        if not hasattr(self.tokenizer, 'eos_token_id') or self.tokenizer.eos_token_id is None:
            raise ValueError("The provided tokenizer must have a defined 'eos_token_id' attribute.")
        self.eos_token_id = self.tokenizer.eos_token_id
        self.chunk_size_bytes = chunk_size_bytes

    def _stream_safe_text_chunks(self, start_byte: int, end_byte: int) -> Generator[str, None, None]:
        """
        Yields UTF-8 decoded text chunks from the byte stream, safely handling
        character boundaries.
        """
        byte_buffer = b''
        for i in range(start_byte, end_byte, self.chunk_size_bytes):
            chunk = self.text_stream.raw_text[i:min(i + self.chunk_size_bytes, end_byte)]
            if not chunk:
                continue
            byte_buffer += chunk
            last_valid_byte_pos = len(byte_buffer)
            while last_valid_byte_pos > 0 and 0x80 <= byte_buffer[last_valid_byte_pos - 1] <= 0xBF:
                last_valid_byte_pos -= 1
            if last_valid_byte_pos > 0:
                first_byte = byte_buffer[last_valid_byte_pos - 1]
                if first_byte >= 0xC0:
                    valid_text = byte_buffer[:last_valid_byte_pos - 1].decode('utf-8', errors='ignore')
                    yield valid_text
                    byte_buffer = byte_buffer[last_valid_byte_pos - 1:]
                else:
                    valid_text = byte_buffer[:last_valid_byte_pos].decode('utf-8', errors='ignore')
                    yield valid_text
                    byte_buffer = byte_buffer[last_valid_byte_pos:]
        if byte_buffer:
            yield byte_buffer.decode('utf-8', errors='ignore')

    def stream_from_partition(self, start_byte: int, end_byte: int, shuffle: bool = True, buffer_size: int = 20000) -> \
            Generator[Dict[str, List[int]], None, None]:
        """
        Generator that yields tokenized examples from a specific byte partition of the file.
        This is the main workhorse method intended for use by parallel workers.
        """
        buffer = []

        def flush_buffer():
            if shuffle:
                random.shuffle(buffer)
            for item in buffer:
                yield item
            buffer.clear()

        token_remainder = []
        for text_segment in self._stream_safe_text_chunks(start_byte, end_byte):
            if not text_segment:
                continue
            all_tokens = token_remainder + self.tokenizer.encode(text_segment)
            documents = []
            current_doc = []
            for token in all_tokens:
                if token == self.eos_token_id:
                    if current_doc:
                        documents.append(current_doc)
                    current_doc = []
                else:
                    current_doc.append(token)

            token_remainder = current_doc

            for doc_tokens in documents:
                if len(doc_tokens) < self.example_len:
                    continue

                for j in range(0, len(doc_tokens) - self.example_len + 1, self.step_len):
                    context_indices = doc_tokens[j: j + self.seq_len]
                    source_indices = doc_tokens[j + self.seq_len: j + self.example_len]
                    buffer.append({"source_ids": source_indices, "context_ids": context_indices})

                    if len(buffer) >= buffer_size:
                        yield from flush_buffer()

        if len(token_remainder) >= self.example_len:
            for j in range(0, len(token_remainder) - self.example_len + 1, self.step_len):
                context_indices = token_remainder[j: j + self.seq_len]
                source_indices = token_remainder[j + self.seq_len: j + self.example_len]
                buffer.append({"source_ids": source_indices, "context_ids": context_indices})

        if buffer:
            yield from flush_buffer()

    def stream_data(self, shuffle: bool = True, buffer_size: int = 20000) -> Generator[
        Dict[str, List[int]], None, None]:
        """
        A public method to stream from the ENTIRE file for single-threaded use.
        """
        total_file_size = len(self.text_stream.raw_text)
        return self.stream_from_partition(start_byte=0, end_byte=total_file_size, shuffle=shuffle,
                                          buffer_size=buffer_size)

    def stream_batches(self, batch_size: int, shuffle: bool = True, buffer_size: int = 20000) -> Generator[
        List[Dict[str, List[int]]], None, None]:
        """
        A convenient public method that yields BATCHES of examples from the entire file.
        """
        current_batch = []
        for item in self.stream_data(shuffle=shuffle, buffer_size=buffer_size):
            current_batch.append(item)
            if len(current_batch) == batch_size:
                yield current_batch
                current_batch = []
        if current_batch:
            yield current_batch
