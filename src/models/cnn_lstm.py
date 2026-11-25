"""CNN + BiLSTM with Visual Cross-Attention for binary visual question answering."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class CrossAttention(nn.Module):
    """Question representation acts as query to attend over spatial image features."""

    def __init__(self, query_dim: int, key_dim: int, hidden_dim: int):
        super().__init__()
        self.q_proj = nn.Linear(query_dim, hidden_dim)
        self.k_proj = nn.Linear(key_dim, hidden_dim)
        self.v_proj = nn.Linear(key_dim, hidden_dim)
        self.scale = hidden_dim**0.5

    def forward(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        # query: (B, query_dim) -> (B, 1, hidden_dim)
        # keys: (B, N, key_dim) -> (B, N, hidden_dim)
        q = self.q_proj(query).unsqueeze(1)
        k = self.k_proj(keys)
        v = self.v_proj(keys)

        scores = torch.bmm(q, k.transpose(1, 2)) / self.scale  # (B, 1, N)
        attn = torch.softmax(scores, dim=-1)  # (B, 1, N)
        attended = torch.bmm(attn, v).squeeze(1)  # (B, hidden_dim)
        return attended


class SmallCNNSpatialEncoder(nn.Module):
    """CNN encoder outputting spatial feature tokens (B, N, C)."""

    def __init__(self):
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
        )
        self.num_features = 128

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)  # (B, 128, H', W')
        b, c, _, _ = feat.shape
        return feat.view(b, c, -1).transpose(1, 2)  # (B, N, 128)


class SpatialTimmEncoder(nn.Module):
    """Pretrained vision backbone outputting spatial feature tokens (B, N, C)."""

    def __init__(self, img_model_name: str, pretrained: bool = True):
        super().__init__()
        import timm

        self.backbone = timm.create_model(img_model_name, pretrained=pretrained, num_classes=0)
        self.num_features = self.backbone.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone.forward_features(x)
        if feat.ndim == 4:
            b, c, _, _ = feat.shape
            return feat.view(b, c, -1).transpose(1, 2)  # (B, H*W, C)
        if feat.ndim == 3:
            return feat
        return feat.unsqueeze(1)


class VQAModel(nn.Module):
    """Visual Question Answering model with Spatial Cross-Attention."""

    def __init__(
        self,
        vocab_size: int,
        n_classes: int,
        img_model_name: str = "efficientnet_b0",
        embedding_dim: int = 128,
        n_layers: int = 2,
        hidden_size: int = 256,
        drop_p: float = 0.3,
        pad_idx: int = 0,
        pretrained_backbone: bool = False,
    ):
        super().__init__()
        if pretrained_backbone:
            self.image_encoder = SpatialTimmEncoder(img_model_name, pretrained=True)
        else:
            self.image_encoder = SmallCNNSpatialEncoder()

        img_feat_dim = self.image_encoder.num_features

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=drop_p if n_layers > 1 else 0.0,
        )

        lstm_dim = hidden_size * 2  # Bidirectional
        self.cross_attention = CrossAttention(
            query_dim=lstm_dim,
            key_dim=img_feat_dim,
            hidden_dim=hidden_size,
        )

        # Fusion: attended visual features (hidden_size) +
        # question features (hidden_size * 2) = hidden_size * 3
        self.fc1 = nn.Linear(hidden_size + lstm_dim, hidden_size)
        self.dropout = nn.Dropout(drop_p)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_size, n_classes)

    def forward(self, img: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        spatial_img = self.image_encoder(img)  # (B, N, C)
        text_emb = self.embedding(text)
        lengths = text.ne(self.embedding.padding_idx).sum(dim=1).clamp_min(1).cpu()
        packed = pack_padded_sequence(text_emb, lengths, batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        lstm_out = torch.cat((hidden[-2], hidden[-1]), dim=1)  # (B, hidden_size * 2)

        # Cross-Attention: Question attends to spatial regions of the image
        attended_visual = self.cross_attention(query=lstm_out, keys=spatial_img)  # (B, hidden_size)

        combined = torch.cat((attended_visual, lstm_out), dim=1)
        x = self.fc1(combined)
        x = self.gelu(x)
        x = self.dropout(x)
        return self.fc2(x)
