"""Profile HTTP opt-in: no real YOLO or torch needed."""

from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image
import pytest

import app.main as main
from app.services.inference_service import InferenceService


class FakeDetector:
    def __init__(self):
        self.calls = 0

    def predict(self, image, confidence_threshold):
        self.calls += 1
        return []


def make_image():
    stream = BytesIO()
    Image.new("RGB", (60, 40), "red").save(stream, format="PNG")
    return stream.getvalue()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("MODEL_PATH", raising=False)
    monkeypatch.delenv("ENABLE_INFERENCE_PROFILING", raising=False)
    main.limiter.reset()
    with TestClient(main.app) as c:
        yield c
    main.app.dependency_overrides.clear()


def request(client, *, profile=False):
    return client.post(
        "/v1/detect",
        files={"image": ("test.png", make_image(), "image/png")},
        data={"profile": str(profile).lower()},
    )


def test_disabled_profiling_rejects_access_without_running_detector(client):
    detector = FakeDetector()
    main.app.dependency_overrides[main.get_inference_service] = lambda: InferenceService(detector)
    response = request(client, profile=True)
    assert response.status_code == 403
    assert "disabled" in response.json()["detail"]
    assert detector.calls == 0


def test_opt_in_profiling_fallback_works_with_plain_detector(client, monkeypatch):
    detector = FakeDetector()
    main.app.dependency_overrides[main.get_inference_service] = lambda: InferenceService(detector)
    main.app.state.settings.enable_inference_profiling = True
    response = request(client, profile=True)
    assert response.status_code == 200, response.text
    body = response.json()
    assert detector.calls == 1
    assert body["profiling"]["yolo"] is None
    assert body["profiling"]["validation_ms"] >= 0
    assert body["profiling"]["decode_rgb_ms"] >= 0
    assert body["profiling"]["service_total_ms"] >= 0


def test_regular_endpoint_does_not_include_profiling_by_default(client, monkeypatch):
    detector = FakeDetector()
    main.app.dependency_overrides[main.get_inference_service] = lambda: InferenceService(detector)
    main.app.state.settings.enable_inference_profiling = True
    response = request(client, profile=False)
    assert response.status_code == 200, response.text
    body = response.json()
    assert "profiling" not in body
    assert "model_name" in body
    assert "inference_time_ms" in body
    assert detector.calls == 1
