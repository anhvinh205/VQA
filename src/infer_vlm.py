"""Zero-shot VQA inference with a pretrained LVLM (LLaVA-1.5-7B).

This is the third approach from the project (part II.3): no
training, just prompting a pretrained vision-language model. Requires
a GPU with bitsandbytes 4-bit quantization support and the
`transformers`/`bitsandbytes` extras. Not used by the FastAPI service
(too heavy for a small deployment) — run standalone for exploration:

    python -m src.infer_vlm --index 0
"""
from __future__ import annotations

import argparse
import os

import torch
from PIL import Image

from src import config
from src.data.loader import load_split


def create_prompt(question: str) -> str:
    return (
        "### INSTRUCTION:\n"
        "Your task is to answer the question based on the given image. "
        "You can only answer 'yes' or 'no'.\n"
        "### USER: <image>\n"
        f"{question}\n"
        "### ASSISTANT:"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "LLaVA 4-bit inference requires a CUDA GPU with bitsandbytes support; "
            "the CPU fallback is not supported for this 7B model."
        )

    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        GenerationConfig,
        LlavaForConditionalGeneration,
    )

    test_data = load_split(config.TEST_SPLIT)
    sample = test_data[args.index]
    image = Image.open(config.IMAGE_DIR / sample["image_path"])

    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    device = "cuda"
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_id, quantization_config=quantization_config, device_map=device
    )

    generation_config = GenerationConfig(
        max_new_tokens=10,
        do_sample=True,
        temperature=0.1,
        top_p=0.95,
        top_k=50,
        eos_token_id=model.config.eos_token_id,
        pad_token=model.config.pad_token_id,
    )

    prompt = create_prompt(sample["question"])
    inputs = processor(prompt, image, padding=True, return_tensors="pt").to(device)
    output = model.generate(**inputs, generation_config=generation_config)
    generated_text = processor.decode(output[0], skip_special_tokens=True)

    prediction = generated_text.split("### ASSISTANT:")[-1].strip()
    print(f"Question: {sample['question']}")
    print(f"Label: {sample['answer']}")
    print(f"Prediction: {prediction}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
