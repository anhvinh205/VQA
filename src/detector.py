from __future__ import annotations

import io
import os
import re
import unicodedata

import torch
from PIL import Image
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
)
from torchvision.transforms.functional import to_tensor


class UnsupportedQuestionError(ValueError):
    pass


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    without_accents = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", without_accents).strip()


class ObjectDetectionPredictor:
    """COCO object detector that uses CUDA when available and otherwise runs on CPU."""

    model_name = "Faster R-CNN MobileNetV3 320 (COCO)"

    def __init__(
        self,
        model: torch.nn.Module | None = None,
        categories: list[str] | None = None,
        threshold: float | None = None,
        device: str | torch.device | None = None,
    ):
        weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        self.categories = categories or weights.meta["categories"]
        self.category_to_id = {name.casefold(): index for index, name in enumerate(self.categories)}
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.threshold = threshold if threshold is not None else float(
            os.environ.get("VQA_DETECTION_THRESHOLD", "0.5")
        )
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("VQA_DETECTION_THRESHOLD must be between 0 and 1")

        self.model = model or fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
        self.model.to(self.device)
        self.model.eval()

    def _resolve_target(self, question: str) -> tuple[str, int]:
        normalized_question = _normalize(question)
        aliases = {
            "people": "person",
            "person": "person",
            "human": "person",
            "bike": "bicycle",
            "plane": "airplane",
            "motorbike": "motorcycle",
            "sofa": "couch",
            "television": "tv",
            "cellphone": "cell phone",
            "mobile phone": "cell phone",
            "cho": "dog",
            "meo": "cat",
            "nguoi": "person",
            "xe dap": "bicycle",
            "xe hoi": "car",
            "oto": "car",
            "xe may": "motorcycle",
            "xe buyt": "bus",
            "may bay": "airplane",
        }
        aliases.update({name.casefold(): name.casefold() for name in self.categories})

        for phrase in sorted(aliases, key=len, reverse=True):
            normalized_phrase = _normalize(phrase)
            pattern = rf"(?<![a-z0-9]){re.escape(normalized_phrase)}(?:s|es)?(?![a-z0-9])"
            if re.search(pattern, normalized_question):
                target = aliases[phrase]
                category_id = self.category_to_id.get(target)
                if category_id is not None:
                    return target, category_id

        supported = ", ".join(self.categories[1:])
        raise UnsupportedQuestionError(
            f"Could not find a supported object category in the question. Supported categories: {supported}"
        )

    @torch.inference_mode()
    def predict(self, image_bytes: bytes, question: str) -> dict:
        target, target_id = self._resolve_target(question)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = to_tensor(image).to(self.device)
        prediction = self.model([image_tensor])[0]

        target_scores = [
            float(score)
            for label, score in zip(prediction["labels"], prediction["scores"], strict=True)
            if int(label) == target_id
        ]
        detection_score = max(target_scores, default=0.0)
        detections = [
            {
                "label": self.categories[int(label)],
                "score": round(float(score), 4),
            }
            for label, score in zip(prediction["labels"], prediction["scores"], strict=True)
            if float(score) >= self.threshold
        ]

        return {
            "answer": "yes" if detection_score >= self.threshold else "no",
            "target": target,
            "detection_score": round(detection_score, 4),
            "threshold": self.threshold,
            "model": self.model_name,
            "detected_objects": detections,
        }