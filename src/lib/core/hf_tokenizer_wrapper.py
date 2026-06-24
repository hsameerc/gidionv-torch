from typing import Any, List, Optional, Sequence

from tokenizers import Tokenizer


class HFTokenizerWrapper:
    def __init__(self, tokenizer_path_or_instance: Any):
        print(tokenizer_path_or_instance)
        if isinstance(tokenizer_path_or_instance, str):
            self.tokenizer: Tokenizer = Tokenizer.from_file(tokenizer_path_or_instance)
        elif isinstance(tokenizer_path_or_instance, Tokenizer):
            self.tokenizer: Tokenizer = tokenizer_path_or_instance
        else:
            raise TypeError("tokenizer_path_or_instance must be a path string or a Tokenizer object.")

        self.vocab_size: int = self.tokenizer.get_vocab_size()
        self.pad_token_id: Optional[int] = self._get_special_token_id("<pad>", "padding")
        self.bos_token_id: Optional[int] = self._get_special_token_id("<s>", "BOS")
        self.eos_token_id: Optional[int] = self._get_special_token_id("</s>", "EOS")
        self.unk_token_id: Optional[int] = self._get_special_token_id("<unk>", "unknown")

        if self.pad_token_id is None:
            print("Warning: '<pad>' token not found. This might be an issue if padding is required.")

    def _get_special_token_id(self, token_str: str, token_name: str) -> Optional[int]:
        token_id = self.tokenizer.token_to_id(token_str)
        if token_id is None:
            print(f"Info: Special token '{token_str}' ({token_name}) not found in vocab for this tokenizer instance.")
        return token_id

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encodes text into a list of token IDs.
        If add_special_tokens is True, prepends BOS and appends EOS if they are defined.
        """
        encoding = self.tokenizer.encode(text, add_special_tokens=False)
        token_ids = encoding.ids

        if add_special_tokens:
            final_ids = []
            if self.bos_token_id is not None:
                final_ids.append(self.bos_token_id)
            final_ids.extend(token_ids)
            if self.eos_token_id is not None:
                final_ids.append(self.eos_token_id)
            return final_ids
        else:
            return token_ids

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        """
        Decodes a list of token IDs back to a string.
        `skip_special_tokens` will remove BOS, EOS, PAD etc. from the decoded string.
        """
        if not isinstance(ids, list):
            if hasattr(ids, 'tolist'):
                ids_list = ids.tolist()
            else:
                ids_list = list(ids)
        else:
            ids_list = ids

        return self.tokenizer.decode(ids_list, skip_special_tokens=skip_special_tokens)

    def tokenize_to_subwords(self, text: str) -> List[str]:
        """
        Tokenizes text into a list of subword strings (not IDs).
        This does not add special BOS/EOS tokens by default.
        """
        encoding = self.tokenizer.encode(text, add_special_tokens=False)
        return encoding.tokens