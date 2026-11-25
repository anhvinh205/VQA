import io

import pytest
import torch
from PIL import Image

from src.detector import ObjectDetectionPredictor, UnsupportedQuestionError


class _FakeDetector(torch.nn.Module):
    def forward(self, images):
        return [
            {
                "labels": torch.tensor([18, 17]),
                "scores": torch.tensor([0.91, 0.83]),
                "boxes": torch.tensor([[1.0, 2.0, 10.0, 12.0], [2.0, 3.0, 8.0, 9.0]]),
            }
            for _ in images
        ]


class _DeviceTrackingDetector(_FakeDetector):
    def to(self, device):
        self.selected_device = torch.device(device)
        return self


def _image_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (24, 24), color="white").save(buffer, format="JPEG")
    return buffer.getvalue()


def test_detector_answers_yes_for_detected_target():
    predictor = ObjectDetectionPredictor(model=_FakeDetector(), threshold=0.5)

    result = predictor.predict(_image_bytes(), "Is there a dog in this image?")

    assert result["answer"] == "yes"
    assert result["target"] == "dog"
    assert result["detection_score"] == pytest.approx(0.91)
    assert result["detected_objects"][0]["label"] == "dog"


def test_detector_resolves_vietnamese_object_alias():
    predictor = ObjectDetectionPredictor(model=_FakeDetector())

    assert predictor._resolve_target("Có chó trong ảnh không?") == ("dog", 18)


def test_detector_rejects_questions_without_supported_object_category():
    predictor = ObjectDetectionPredictor(model=_FakeDetector())

    with pytest.raises(UnsupportedQuestionError, match="Supported categories"):
        predictor.predict(_image_bytes(), "Is the sky blue?")


def test_detector_answers_no_when_target_score_is_below_threshold():
    predictor = ObjectDetectionPredictor(model=_FakeDetector(), threshold=0.95)

    result = predictor.predict(_image_bytes(), "Is there a dog?")

    assert result["answer"] == "no"
    assert result["detection_score"] == pytest.approx(0.91)
    assert result["detected_objects"] == []


@pytest.mark.parametrize(("cuda_available", "expected_device"), [(True, "cuda"), (False, "cpu")])
def test_detector_selects_available_device(monkeypatch, cuda_available: bool, expected_device: str):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    model = _DeviceTrackingDetector()

    predictor = ObjectDetectionPredictor(model=model)

    assert predictor.device.type == expected_device
    assert model.selected_device.type == expected_device