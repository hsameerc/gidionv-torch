import os
from typing import List

from tokenizers import Tokenizer, normalizers, decoders, processors
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


def train_bpe_tokenizer_if_needed(raw_text_corpus_path: str, save_dir: str, tokenizer_filename_base: str,
                                  vocab_size: int, min_freq: int, special_tokens_list: List[str]) -> str:
    """Trains a ByteLevel BPE tokenizer and saves it as tokenizer.json."""
    tokenizer_save_path = os.path.join(save_dir, tokenizer_filename_base + ".json")
    os.makedirs(save_dir, exist_ok=True)
    if os.path.exists(tokenizer_save_path):
        print(f"BPE Tokenizer already found at '{tokenizer_save_path}'. Skipping training.")
        return tokenizer_save_path
    print(f"Training BPE tokenizer (ByteLevel) with vocab_size={vocab_size} from '{raw_text_corpus_path}'...")
    # 1. Init model
    bpe_tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    # 2. Setup tokenizer pipeline
    bpe_tokenizer.normalizer = normalizers.NFC()
    bpe_tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
    bpe_tokenizer.decoder = decoders.ByteLevel()
    # 3. Post-processing template
    bpe_tokenizer.post_processor = processors.TemplateProcessing(single="<s> $A </s>", pair="<s> $A </s> </s> $B </s>",
                                                                 special_tokens=[
                                                                     ("<s>", special_tokens_list.index("<s>")),
                                                                     ("</s>", special_tokens_list.index("</s>")), ])
    # 4. Train
    trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=min_freq, special_tokens=special_tokens_list,
                         initial_alphabet=ByteLevel.alphabet(), show_progress=True)
    try:
        bpe_tokenizer.train([raw_text_corpus_path], trainer=trainer)
        bpe_tokenizer.save(tokenizer_save_path)
        print(f"BPE Tokenizer trained and saved to '{tokenizer_save_path}'")
    except Exception as e:
        print(f"ERROR during BPE tokenizer training: {e}")
        raise

    return tokenizer_save_path
