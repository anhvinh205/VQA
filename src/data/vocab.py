from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

_TOKEN_RE = re.compile(r"[A-Za-z']+|[?.,!]")

PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"
SPECIALS = [PAD, SOS, EOS, UNK]


def tokenize_words(text: str) -> list[str]:
    return [tok.lower() for tok in _TOKEN_RE.findall(text)]


class Vocab:
    def __init__(self, token_to_idx: dict[str, int]):
        self.token_to_idx = token_to_idx
        self.idx_to_token = {idx: tok for tok, idx in token_to_idx.items()}
        self.default_index = token_to_idx[UNK]

    def __len__(self) -> int:
        return len(self.token_to_idx)

    def __getitem__(self, token: str) -> int:
        return self.token_to_idx.get(token, self.default_index)

    def encode(self, text: str, max_seq_len: int) -> list[int]:
        ids = [self[tok] for tok in tokenize_words(text)]
        if len(ids) < max_seq_len:
            ids = ids + [self[PAD]] * (max_seq_len - len(ids))
        else:
            ids = ids[:max_seq_len]
        return ids

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.token_to_idx, f)

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        with open(path, "r", encoding="utf-8") as f:
            token_to_idx = json.load(f)
        return cls(token_to_idx)

    @classmethod
    def build(
        cls,
        questions: Iterable[str],
        min_freq: int = 2,
        specials: list[str] | None = None,
    ) -> Vocab:
        specials = specials or SPECIALS
        counter: Counter[str] = Counter()
        for q in questions:
            counter.update(tokenize_words(q))
        token_to_idx = {tok: idx for idx, tok in enumerate(specials)}
        for tok, freq in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
            if freq >= min_freq and tok not in token_to_idx:
                token_to_idx[tok] = len(token_to_idx)
        return cls(token_to_idx)