"""CNN + BiLSTM baseline for binary visual question answering."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class SmallCNNEncoder(nn.Module):
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
        return self.proj(torch.flatten(self.features(x), 1))


def build_image_encoder(img_model_name: str, hidden_size: int, pretrained: bool) -> nn.Module:
    if not pretrained:
        return SmallCNNEncoder(hidden_size)

    import timm

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
        pad_idx: int = 0,
        pretrained_backbone: bool = False,
    ):
        super().__init__()
        self.image_encoder = build_image_encoder(img_model_name, hidden_size, pretrained_backbone)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
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
        lengths = text.ne(self.embedding.padding_idx).sum(dim=1).clamp_min(1).cpu()
        packed = pack_padded_sequence(text_emb, lengths, batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        lstm_out = torch.cat((hidden[-2], hidden[-1]), dim=1)

        combined = torch.cat((img_features, lstm_out), dim=1)
        x = self.fc1(combined)
        x = self.gelu(x)
        x = self.dropout(x)
        return self.fc2(x)
    