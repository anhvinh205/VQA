import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app, predictor_state
from app.schemas import PredictionResponse
from src.detector import ObjectDetectionPredictor
from src.predictor import VQAPredictor


class _FakePredictor:
    def predict(self, image_bytes: bytes, question: str) -> dict:
        return {
            "answer": "yes",
            "target": "dog",
            "detection_score": 0.87,
            "threshold": 0.5,
            "model": ObjectDetectionPredictor.model_name,
            "detected_objects": [{"label": "dog", "score": 0.87}],
        }


class _FakeClassifierPredictor(VQAPredictor):
    def predict(self, image_bytes: bytes, question: str) -> dict:
        return {"answer": "no", "confidence": 0.7, "probabilities": {"no": 0.7, "yes": 0.3}}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr("app.main.create_predictor", _FakePredictor)
    with TestClient(app) as c:
        yield c


def _sample_image_bytes() -> bytes:
    img = Image.new("RGB", (32, 32), color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert "model_loaded" in resp.json()


def test_ready_returns_503_without_model(client):
    predictor_state["predictor"] = None
    resp = client.get("/ready")
    assert resp.status_code == 503
    assert resp.json()["status"] == "not_ready"


def test_predict_without_model_returns_503(client):
    predictor_state["predictor"] = None
    resp = client.post(
        "/predict",
        files={"image": ("test.jpg", _sample_image_bytes(), "image/jpeg")},
        data={"question": "Is this red?"},
    )
    assert resp.status_code == 503


def test_predict_with_fake_model_succeeds(client):
    predictor_state["predictor"] = _FakePredictor()
    resp = client.post(
        "/predict",
        files={"image": ("test.jpg", _sample_image_bytes(), "image/jpeg")},
        data={"question": "Is this red?"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "yes"
    assert body["target"] == "dog"
    assert body["detection_score"] == 0.87
    assert body["detected_objects"] == [{"label": "dog", "score": 0.87}]


def test_predict_with_cnn_baseline_keeps_legacy_response(client):
    predictor_state["predictor"] = _FakeClassifierPredictor()

    response = client.post(
        "/predict",
        files={"image": ("test.jpg", _sample_image_bytes(), "image/jpeg")},
        data={"question": "Is this a dog?"},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "no"
    assert response.json()["probabilities"] == {"no": 0.7, "yes": 0.3}


def test_predict_rejects_empty_question(client):
    predictor_state["predictor"] = _FakePredictor()
    resp = client.post(
        "/predict",
        files={"image": ("test.jpg", _sample_image_bytes(), "image/jpeg")},
        data={"question": "  "},
    )
    assert resp.status_code == 422


def test_predict_rejects_bad_content_type(client):
    predictor_state["predictor"] = _FakePredictor()
    resp = client.post(
        "/predict",
        files={"image": ("test.txt", b"not an image", "text/plain")},
        data={"question": "Is this red?"},
    )
    assert resp.status_code == 415


def test_prediction_response_exposes_detector_score_not_probabilities():
    response = PredictionResponse(
        answer="yes",
        target="dog",
        detection_score=0.87,
        threshold=0.5,
        model=ObjectDetectionPredictor.model_name,
        detected_objects=[{"label": "dog", "score": 0.87}],
    )
    assert response.answer == "yes"
    assert response.detection_score == 0.87
