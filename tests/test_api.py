import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app, predictor_state


class _FakePredictor:
    def predict(self, image_bytes: bytes, question: str) -> dict:
        return {"answer": "yes", "confidence": 0.87, "probabilities": {"yes": 0.87, "no": 0.13}}


@pytest.fixture
def client():
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
    assert body["confidence"] == 0.87


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
