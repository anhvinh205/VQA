"""CNN + BiLSTM baseline for Yes/No VQA (project part II.1).

The image branch can use a `timm` pretrained backbone (as in the
original course material) or a small from-scratch CNN when no
internet access is available to download pretrained weights (e.g.
in an offline CI runner). Toggle with `pretrained_backbone`.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SmallCNNEncoder(nn.Module):
    """A tiny from-scratch conv encoder, used when pretrained weights
    can't be downloaded (offline environments)."""

    def __init__(self, out_dim: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.proj(x)


def build_image_encoder(img_model_name: str, hidden_size: int, pretrained: bool) -> nn.Module:
    if not pretrained:
        return SmallCNNEncoder(hidden_size)

    import timm  # imported lazily: only required when using pretrained backbones

    return timm.create_model(img_model_name, pretrained=True, num_classes=hidden_size)


class VQAModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_classes: int,
        img_model_name: str = "resnet18",
        embedding_dim: int = 128,
        n_layers: int = 2,
        hidden_size: int = 256,
        drop_p: float = 0.2,
        pretrained_backbone: bool = False,
    ):
        super().__init__()
        self.image_encoder = build_image_encoder(img_model_name, hidden_size, pretrained_backbone)

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=drop_p if n_layers > 1 else 0.0,
        )

        self.fc1 = nn.Linear(hidden_size * 3, hidden_size)
        self.dropout = nn.Dropout(drop_p)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_size, n_classes)

    def forward(self, img: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        img_features = self.image_encoder(img)

        text_emb = self.embedding(text)
        lstm_out, _ = self.lstm(text_emb)
        lstm_out = lstm_out[:, -1, :]

        combined = torch.cat((img_features, lstm_out), dim=1)
        x = self.fc1(combined)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
