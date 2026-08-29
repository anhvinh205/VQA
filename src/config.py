"""Central configuration for the VQA (Yes/No) project.

All paths and hyperparameters are collected here so that training,
inference and the API all agree on the same values.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("VQA_DATA_DIR", ROOT_DIR / "data"))
ARTIFACTS_DIR = Path(os.environ.get("VQA_ARTIFACTS_DIR", ROOT_DIR / "artifacts"))

TRAIN_SPLIT = DATA_DIR / "vaq2.0.TrainImages.txt"
DEV_SPLIT = DATA_DIR / "vaq2.0.DevImages.txt"
TEST_SPLIT = DATA_DIR / "vaq2.0.TestImages.txt"
IMAGE_DIR = DATA_DIR / "val2014-resised"

VOCAB_PATH = ARTIFACTS_DIR / "vocab.json"
LABELS_PATH = ARTIFACTS_DIR / "labels.json"
CNN_LSTM_CHECKPOINT = ARTIFACTS_DIR / "cnn_lstm_model.pt"


@dataclass
class CNNLSTMConfig:
    """Hyperparameters for the CNN + BiLSTM baseline (see project part II.1)."""

    img_model_name: str = "resnet18"
    pretrained_backbone: bool = os.environ.get("VQA_PRETRAINED", "0") == "1"
    embedding_dim: int = 128
    hidden_size: int = 256
    n_layers: int = 2
    dropout: float = 0.2
    max_seq_len: int = 20
    image_size: int = 128
    train_batch_size: int = 32
    eval_batch_size: int = 32
    lr: float = 1e-3
    epochs: int = 5
    seed: int = 59
    min_freq: int = 2
    specials: list = field(default_factory=lambda: ["<pad>", "<sos>", "<eos>", "<unk>"])


CNN_LSTM_CFG = CNNLSTMConfig()
