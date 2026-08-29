"""Parses the vaq2.0.*Images.txt split files into structured samples."""
from __future__ import annotations

from pathlib import Path
from typing import TypedDict


class VQASample(TypedDict):
    image_path: str
    question: str
    answer: str


def load_split(path: Path) -> list[VQASample]:
    """Read one of the vaq2.0.*Images.txt files.

    Each line looks like:
        COCO_val2014_000000393225.jpg#0\tIs this a creamy soup ? no
    """
    samples: list[VQASample] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            image_tag, qa = line.split("\t")
            qa_parts = qa.split("?")
            if len(qa_parts) == 3:
                answer = qa_parts[2].strip()
            else:
                answer = qa_parts[1].strip()
            samples.append(
                VQASample(
                    image_path=image_tag[:-2],  # strip the "#0" suffix
                    question=qa_parts[0].strip() + "?",
                    answer=answer,
                )
            )
    return samples
