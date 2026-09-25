"""Load the trained CNN+LSTM checkpoint and expose a single predict() call."""
from __future__ import annotations

import hashlib
import io
import json
import logging
from pathlib import Path

import torch
from PIL import Image

from src import config
from src.data.dataset import build_transforms
from src.data.vocab import PAD, Vocab
from src.models.cnn_lstm import VQAModel

logger = logging.getLogger(__name__)


def _verify_checkpoint_integrity(checkpoint_path: Path) -> None:
    manifest_path = checkpoint_path.parent / "manifest.json"
    if not manifest_path.exists():
        return

    with manifest_path.open(encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    expected_hash = manifest.get("sha256", {}).get(checkpoint_path.name)
    if not expected_hash:
        return

    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest().upper() != expected_hash.upper():
        raise ValueError(f"Checkpoint integrity verification failed for {checkpoint_path}")


class VQAPredictor:
    def __init__(
        self,
        checkpoint_path: Path = config.CNN_LSTM_CHECKPOINT,
        vocab_path: Path = config.VOCAB_PATH,
        labels_path: Path = config.LABELS_PATH,
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.vocab = Vocab.load(vocab_path)
        with open(labels_path, encoding="utf-8") as f:
            label2idx = json.load(f)
        self.idx2label = {v: k for k, v in label2idx.items()}

        _verify_checkpoint_integrity(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        cfg = checkpoint["config"]
        self.max_seq_len = cfg["max_seq_len"]

        self.model = VQAModel(
            vocab_size=checkpoint["vocab_size"],
            n_classes=len(checkpoint["classes"]),
            img_model_name=cfg["img_model_name"],
            embedding_dim=cfg["embedding_dim"],
            n_layers=cfg["n_layers"],
            hidden_size=cfg["hidden_size"],
            drop_p=cfg["dropout"],
            pad_idx=self.vocab[PAD],
            pretrained_backbone=cfg.get("pretrained_backbone", False),
        )
        model_state = checkpoint.get("model_state") or checkpoint["model_state_dict"]
        self.model.load_state_dict(model_state)
        self.model.to(self.device)
        self.model.eval()

        self.transform = build_transforms(cfg["image_size"])["eval"]
        logger.info("loaded VQA predictor on %s", self.device)
        
    @torch.no_grad()
    def predict(self, image_bytes: bytes, question: str) -> dict:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        question_ids = self.vocab.encode(question, self.max_seq_len)
        question_tensor = torch.tensor([question_ids], dtype=torch.long).to(self.device)

        logits = self.model(img_tensor, question_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(probs.argmax().item())

        return {
            "answer": self.idx2label[pred_idx],
            "confidence": round(float(probs[pred_idx].item()), 4),
            "probabilities": {self.idx2label[i]: round(float(p.item()), 4) for i, p in enumerate(probs)},
        }
