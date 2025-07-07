import unittest
from typing import List

from src.loaders.text_loader import AdvancedDataStreamer


class DummyTokenizer:
    """A predictable tokenizer for testing."""

    def __init__(self):
        self.vocab = {'<eos>': 99}
        self.next_id = 1
        self.eos_token_id = 99

    def encode(self, text: str) -> List[int]:
        tokens = text.split()
        ids = []
        for token in tokens:
            if token not in self.vocab:
                if token != '<eos>':
                    self.vocab[token] = self.next_id
                    self.next_id += 1
            ids.append(self.vocab[token])
        return ids


class DummyTextLoaderStream:
    """A mock text stream that mimics a memory-mapped file from bytes."""

    def __init__(self, text: str):
        self.raw_text = text.encode('utf-8')

    def __len__(self):
        return len(self.raw_text)


class TestAdvanceDataStreamer(unittest.TestCase):

    def test_utf8_safety_at_chunk_boundaries(self):
        """
        CRITICAL: Ensures multibyte UTF-8 characters are not corrupted
        when a chunk boundary splits them.
        """
        original_text = "hello world € and some more text 🚀 this is a test"
        stream = DummyTextLoaderStream(original_text)
        tokenizer = DummyTokenizer()

        data_streamer = AdvancedDataStreamer(
            stream, tokenizer, seq_len=10, chunk_size_bytes=13
        )

        text_chunks = list(data_streamer._stream_safe_text_chunks(0, len(stream)))
        reassembled_text = "".join(text_chunks)
        assert reassembled_text == original_text


    def test_non_overlapping_example_generation(self):
        """
        Tests if correct source/context pairs are created with no overlap.
        """
        text = "t1 t2 t3 t4 t5 t6 <eos>"
        stream = DummyTextLoaderStream(text)
        tokenizer = DummyTokenizer()
        data_streamer = AdvancedDataStreamer(
            stream, tokenizer, seq_len=3, overlap_len_tokens=0
        )
        examples = list(data_streamer.stream_data(shuffle=False))
        assert len(examples) == 1
        expected_ids = tokenizer.encode("t1 t2 t3 t4 t5 t6")
        assert examples[0]['context_ids'] == expected_ids[0:3]  # [t1, t2, t3]
        assert examples[0]['source_ids'] == expected_ids[3:6]  # [t4, t5, t6]

    def test_overlapping_example_generation(self):
        """
        Tests if the sliding window correctly overlaps examples.
        """
        text = "t1 t2 t3 t4 t5 t6 t7 t8 <eos>"
        stream = DummyTextLoaderStream(text)
        tokenizer = DummyTokenizer()
        data_streamer = AdvancedDataStreamer(
            stream, tokenizer, seq_len=2, overlap_len_tokens=2
        )

        examples = list(data_streamer.stream_data(shuffle=False))
        all_ids = tokenizer.encode("t1 t2 t3 t4 t5 t6 t7 t8")
        assert len(examples) == 3
        assert examples[0]['context_ids'] == all_ids[0:2]
        assert examples[0]['source_ids'] == all_ids[2:4]
        assert examples[1]['context_ids'] == all_ids[2:4]
        assert examples[1]['source_ids'] == all_ids[4:6]
        assert examples[2]['context_ids'] == all_ids[4:6]
        assert examples[2]['source_ids'] == all_ids[6:8]

    def test_document_splitting_and_skipping(self):
        """
        Ensures text is split by <eos> and short documents are ignored.
        """
        text = "doc1_a doc1_b doc1_c doc1_d <eos> short doc <eos> doc2_a doc2_b doc2_c doc2_d <eos>"
        stream = DummyTextLoaderStream(text)
        tokenizer = DummyTokenizer()
        data_streamer = AdvancedDataStreamer(
            stream, tokenizer, seq_len=2, overlap_len_tokens=0
        )

        examples = list(data_streamer.stream_data(shuffle=False))

        assert len(examples) == 2

        doc1_ids = tokenizer.encode("doc1_a doc1_b doc1_c doc1_d")
        doc2_ids = tokenizer.encode("doc2_a doc2_b doc2_c doc2_d")

        assert examples[0]['context_ids'] == doc1_ids[0:2]
        assert examples[1]['context_ids'] == doc2_ids[0:2]

    def test_final_remainder_processing_without_eos(self):
        """
        Tests that data at the end of the file is processed even without a
        trailing <eos> token.
        """
        text = "t1 t2 t3 t4"
        stream = DummyTextLoaderStream(text)
        tokenizer = DummyTokenizer()
        data_streamer = AdvancedDataStreamer(
            stream, tokenizer, seq_len=2, overlap_len_tokens=0
        )
        examples = list(data_streamer.stream_data(shuffle=False))
        assert len(examples) == 1
        all_ids = tokenizer.encode(text)
        assert examples[0]['source_ids'] == all_ids[2:4]

    def test_stream_batches(self):
        """
        Verifies the batching wrapper works as intended.
        """
        text = "t1 t2 t3 t4 t5 t6 t7 t8 t9 t10 t11 t12 <eos>"
        stream = DummyTextLoaderStream(text)
        tokenizer = DummyTokenizer()
        data_streamer = AdvancedDataStreamer(
            stream, tokenizer, seq_len=1, overlap_len_tokens=0
        )
        batches = list(data_streamer.stream_batches(batch_size=2, shuffle=False))
        assert len(batches) == 3
        assert all(len(batch) == 2 for batch in batches)
        batches = list(data_streamer.stream_batches(batch_size=4, shuffle=False))
        assert len(batches) == 2
        assert len(batches[0]) == 4
        assert len(batches[1]) == 2