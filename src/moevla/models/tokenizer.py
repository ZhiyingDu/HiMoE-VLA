import logging

import numpy as np
import sentencepiece
import re
import moevla.shared.download as download


class PaligemmaTokenizer:
    def __init__(self, max_len: int = 48):
        self._max_len = max_len

        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    def _preserve_special_tokens(self, text: str) -> str:
        specials = re.findall(r"\[.*?:.*?\]", text)
        segments = re.split(r"(\[.*?:.*?\])", text)
        processed = []
        for seg in segments:
            if seg in specials:
                processed.append(seg) 
            else:
                seg = seg.replace("_", " ").replace("\n", " ").lower()
                processed.append(seg)
        return "".join(processed).strip()
    
    def tokenize(self, prompt: str, preserve_special_tokens: bool = True) -> tuple[np.ndarray, np.ndarray]:
        if preserve_special_tokens:
            cleaned_text = self._preserve_special_tokens(prompt)
        else:
            cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ").lower()
        if not cleaned_text.endswith(('.', '!', '?')):
            cleaned_text += '.'
        # tokenize "\n" separately as the "start of answer" token
        tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + self._tokenizer.encode("\n")
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len
        return np.asarray(tokens), np.asarray(mask)