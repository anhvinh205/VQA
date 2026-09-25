from __future__ import annotations

import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    def __init__(self, model_name: str = "roberta-base"):
        super().__init__()
        from transformers import RobertaModel

        self.model = RobertaModel.from_pretrained(model_name)

    def forward(self, inputs: dict) -> torch.Tensor:
        return self.model(**inputs).pooler_output


class VisualEncoder(nn.Module):
    def __init__(self, model_name: str = "google/vit-base-patch16-224"):
        super().__init__()
        from transformers import ViTModel

        self.model = ViTModel.from_pretrained(model_name)

    def forward(self, inputs: dict) -> torch.Tensor:
        return self.model(**inputs).pooler_output


class Classifier(nn.Module):
    def __init__(self, hidden_size: int = 512, dropout_prob: float = 0.2, n_classes: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(768 * 2, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        return self.fc2(x)


class VQAModel(nn.Module):
    def __init__(self, visual_encoder: VisualEncoder, text_encoder: TextEncoder, classifier: Classifier):
        super().__init__()
        self.visual_encoder = visual_encoder
        self.text_encoder = text_encoder
        self.classifier = classifier

    def forward(self, image: dict, question: dict) -> torch.Tensor:
        text_out = self.text_encoder(question)
        image_out = self.visual_encoder(image)
        x = torch.cat((image_out, text_out), dim=1)
        return self.classifier(x)

    def freeze(self, visual: bool = True, textual: bool = True, classifier: bool = False) -> None:
        if visual:
            for p in self.visual_encoder.parameters():
                p.requires_grad = False
        if textual:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
        if classifier:
            for p in self.classifier.parameters():
                p.requires_grad = False