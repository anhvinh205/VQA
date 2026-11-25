"""Inference predictor for ViT + RoBERTa VQA model."""
from __future__ import annotations

import io
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer

from src import config
from src.models.vit_roberta import Classifier, TextEncoder, VisualEncoder, VQAModel


class VitRobertaPredictor:
    def __init__(
        self,
        checkpoint_path: Path = config.ARTIFACTS_DIR / "vit_roberta_model.pt",
        vit_model_name: str = "google/vit-base-patch16-224",
        roberta_model_name: str = "roberta-base",
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(roberta_model_name)
        self.image_processor = AutoImageProcessor.from_pretrained(vit_model_name)

        visual_encoder = VisualEncoder(vit_model_name)
        text_encoder = TextEncoder(roberta_model_name)
        classifier = Classifier(hidden_size=512, dropout_prob=0.0, n_classes=2)
        self.model = VQAModel(visual_encoder, text_encoder, classifier)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
        self.model.eval()

        self.idx2label = {0: "no", 1: "yes"}
        if "label2idx" in checkpoint:
            self.idx2label = {v: k for k, v in checkpoint["label2idx"].items()}

    @torch.no_grad()
    def predict(self, image_bytes: bytes, question: str) -> dict:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_inputs = self.image_processor(images=image, return_tensors="pt")
        pixel_values = img_inputs["pixel_values"].to(self.device)

        text_inputs = self.tokenizer(
            question,
            max_length=32,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs["input_ids"].to(self.device)
        attention_mask = text_inputs["attention_mask"].to(self.device)

        img_dict = {"pixel_values": pixel_values}
        text_dict = {"input_ids": input_ids, "attention_mask": attention_mask}

        logits = self.model(img_dict, text_dict)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(probs.argmax().item())

        return {
            "answer": self.idx2label[pred_idx],
            "confidence": round(float(probs[pred_idx].item()), 4),
            "probabilities": {self.idx2label[i]: round(float(p.item()), 4) for i, p in enumerate(probs)},
        }
