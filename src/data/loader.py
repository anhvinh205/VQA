from __future__ import annotations

from pathlib import Path
from typing import TypedDict


class VQASample(TypedDict):
    image_path: str
    question: str
    answer: str

def load_split(path: Path) -> list[VQASample]:
    samples: list[VQASample] = []
    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            image_tag, qa = line.split("\t", maxsplit=1)
            question_text, answer = qa.rsplit("?", maxsplit=1)
            samples.append(
                VQASample(
                    image_path=image_tag.removesuffix("#0"),
                    question=question_text.strip() + "?",
                    answer=answer.strip(),
                )
            )
    return samples