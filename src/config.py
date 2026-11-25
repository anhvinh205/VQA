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
    img_model_name: str = "efficientnet_b0"   # Phase 2: mạnh hơn resnet18
    embedding_dim: int = 128
    hidden_size: int = 256
    n_layers: int = 2
    dropout: float = 0.3                      # Phase 2: tăng nhẹ để tránh overfit
    max_seq_len: int = 20
    image_size: int = 224                     # Phase 1: 128 → 224 (chuẩn ImageNet)
    train_batch_size: int = 32
    eval_batch_size: int = 64
    lr: float = 3e-4                          # Phase 2: giảm lr phù hợp AdamW
    epochs: int = 20                          # Phase 1: 5 → 20
    seed: int = 59
    min_freq: int = 2
    specials: list = field(default_factory=lambda: ["<pad>", "<sos>", "<eos>", "<unk>"])
    pretrained_backbone: bool = os.environ.get("VQA_PRETRAINED", "0") == "1"
    early_stopping_patience: int = 5          # Phase 1: 3 → 5
    early_stopping_min_delta: float = 1e-4


CNN_LSTM_CFG = CNNLSTMConfig()
