"""HTTP integration tests: inject a fake detector, no real model weights needed."""

from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.main as main
import app.core.lifecycle as lifecycle
from app.services.inference_service import (
    BoundingBox,
    Detection,
    InferenceService,
)


def make_image(image_format="PNG", mode="RGB"):
    output = BytesIO()
    Image.new(mode, (80, 60), "red" if mode == "RGB" else 128).save(
        output, format=image_format
    )
    return output.getvalue()


def upload(client, content=None, mime="image/png", threshold="0.6"):
    return client.post(
        "/v1/detect",
        files={"image": ("example.png", make_image() if content is None else content, mime)},
        data={"confidence_threshold": threshold},
    )


class FakeDetector:
    def __init__(self):
        self.calls = []

    def predict(self, image, confidence_threshold):
        self.calls.append((image.mode, image.size, confidence_threshold))
        return [
            Detection(
                class_id=0,
                class_name="person",
                confidence=0.93,
                bbox=BoundingBox(x1=10, y1=12, x2=40, y2=50),
            )
        ]


@pytest.fixture
def client(monkeypatch):
    """TestClient as context manager runs startup/shutdown (lifespan)."""
    monkeypatch.delenv("MODEL_PATH", raising=False)
    main.limiter.reset()
    with TestClient(main.app) as client:
        yield client
    main.app.dependency_overrides.clear()


def test_liveness_vs_readiness_without_a_model(client):
    assert client.get("/health").json() == {"status": "ok"}
    ready = client.get("/ready")
    assert ready.status_code == 503
    assert "not ready" in ready.json()["detail"].lower()


def test_detection_returns_503_when_model_not_configured(client):
    response = upload(client)
    assert response.status_code == 503


def test_detection_success_with_injected_detector(client):
    detector = FakeDetector()
    main.app.dependency_overrides[main.get_inference_service] = (
        lambda: InferenceService(detector)
    )
    response = upload(client)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "success"
    assert body["format"] == "PNG"
    assert (body["width"], body["height"]) == (80, 60)
    assert body["filename"] == "example.png"
    assert body["model_name"] is None  # Only fake dependency loaded.
    assert body["inference_time_ms"] >= 0
    assert body["detections"] == [{
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.93,
        "bbox": {"x1": 10, "y1": 12, "x2": 40, "y2": 50},
    }]
    assert detector.calls == [("RGB", (80, 60), 0.6)]


def test_detection_converts_grayscale_to_rgb(client):
    detector = FakeDetector()
    main.app.dependency_overrides[main.get_inference_service] = (
        lambda: InferenceService(detector)
    )
    response = upload(client, make_image(mode="L"))
    assert response.status_code == 200
    assert detector.calls[0][0] == "RGB"


def test_detection_rejects_invalid_image_before_detector(client):
    detector = FakeDetector()
    main.app.dependency_overrides[main.get_inference_service] = (
        lambda: InferenceService(detector)
    )
    response = upload(client, b"not an image")
    assert response.status_code == 400
    assert detector.calls == []


def test_detection_rejects_unsupported_content_type(client):
    main.app.dependency_overrides[main.get_inference_service] = (
        lambda: InferenceService(FakeDetector())
    )
    response = upload(client, b"some text", mime="text/plain")
    assert response.status_code == 415


def test_detection_rejects_oversize_upload(client):
    main.app.dependency_overrides[main.get_inference_service] = (
        lambda: InferenceService(FakeDetector())
    )
    response = upload(client, b"x" * (10 * 1024 * 1024 + 1))
    assert response.status_code == 413


def test_detection_rejects_invalid_threshold(client):
    main.app.dependency_overrides[main.get_inference_service] = (
        lambda: InferenceService(FakeDetector())
    )
    response = upload(client, threshold="1.2")
    assert response.status_code == 422


def test_detection_hides_unexpected_model_error(client):
    class BrokenDetector:
        def predict(self, image, confidence_threshold):
            raise RuntimeError("secret internal error")

    main.app.dependency_overrides[main.get_inference_service] = (
        lambda: InferenceService(BrokenDetector())
    )
    response = upload(client)
    assert response.status_code == 500
    assert response.json()["detail"] == "Model inference failed."
    assert "secret internal error" not in response.text


def test_model_is_loaded_once_per_lifespan(monkeypatch, tmp_path):
    fake_weights = tmp_path / "fake-weights.pt"
    fake_weights.write_bytes(b"not actually read")
    monkeypatch.setenv("MODEL_PATH", str(fake_weights))
    monkeypatch.setenv("MODEL_DEVICE", "cpu")
    # This test checks one model per lifespan, not the multi-model pool.
    # Override any MODEL_POOL_WORKERS value inherited from PowerShell or .env.
    monkeypatch.setenv("MODEL_POOL_WORKERS", "0")
    main.limiter.reset()

    creations = []

    class StubYOLODetector(FakeDetector):
        def __init__(self, model_path, *, device):
            super().__init__()
            creations.append((str(model_path), device))

    monkeypatch.setattr(lifecycle, "YOLODetector", StubYOLODetector)

    with TestClient(main.app) as client:
        ready = client.get("/ready")
        assert ready.status_code == 200
        assert ready.json()["model_name"] == "fake-weights.pt"
        assert upload(client).status_code == 200
        assert upload(client).status_code == 200
        assert len(creations) == 1
        assert len(main.app.state.inference_service.detector.calls) == 2

    assert main.app.state.inference_service is None
