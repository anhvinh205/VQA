from pathlib import Path

from src.data.loader import load_split

def test_load_split_parses_lines(tmp_path: Path):
    split_file = tmp_path / "split.txt"
    split_file.write_text(
        "COCO_val2014_000000393225.jpg#0\tIs this a creamy soup ? no\n"
        "COCO_val2014_000000393243.jpg#0\tIs this person wearing a tie ? no\n",
        encoding="utf-8",
    )

    samples = load_split(split_file)

    assert len(samples) == 2
    assert samples[0]["image_path"] == "COCO_val2014_000000393225.jpg"
    assert samples[0]["question"] == "Is this a creamy soup?"
    assert samples[0]["answer"] == "no"


def test_load_split_handles_question_mark_in_answer_split(tmp_path: Path):
    split_file = tmp_path / "split.txt"
    split_file.write_text(
        "img.jpg#0\tIs this a ? weird question ? yes\n",
        encoding="utf-8",
    )
    samples = load_split(split_file)
    assert samples[0]["answer"] == "yes"