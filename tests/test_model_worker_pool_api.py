"""Verify ASGI lifespan with dedicated model threads and response compatibility."""

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import threading
import time

from fastapi.testclient import TestClient
from PIL import Image
import pytest

import app.main as main
import app.core.lifecycle as lifecycle


def image_bytes():
    buffer = BytesIO()
    Image.new("RGB", (45, 32), color="green").save(buffer, format="PNG")
    return buffer.getvalue()


def post(client, profile=True):
    return client.post(
        "/v1/detect",
        files={"image": ("sample.png", image_bytes(), "image/png")},
        data={"profile": str(profile).lower()},
    )


def test_two_model_workers_have_isolated_instances(monkeypatch, tmp_path):
    fake_weights = tmp_path / "two-models.pt"
    fake_weights.write_bytes(b"unused by fake")
    monkeypatch.setenv("MODEL_PATH", str(fake_weights))
    monkeypatch.setenv("MODEL_POOL_WORKERS", "2")
    monkeypatch.setenv("ENABLE_INFERENCE_PROFILING", "1")
    main.limiter.reset()
    constructed = []
    guard = threading.Lock()

    class FakeYOLODetector:
        def __init__(self, model_path, *, device):
            self.constructor_thread = threading.get_ident()
            with guard:
                constructed.append(self)

        def predict(self, image, confidence_threshold):
            assert threading.get_ident() == self.constructor_thread
            time.sleep(0.025)
            return []

    monkeypatch.setattr(lifecycle, "YOLODetector", FakeYOLODetector)
    with TestClient(main.app) as client:
        assert client.get("/ready").json() == {
            "status": "ready", "model_name": "two-models.pt", "model_pool_workers": 2
        }
        with ThreadPoolExecutor(max_workers=2) as ex:
            futures = [ex.submit(post, client, True) for _ in range(2)]
            replies = [future.result(timeout=5) for future in futures]
        assert all(r.status_code == 200 for r in replies), [r.text for r in replies]
        assert {r.json()["model_worker_id"] for r in replies} == {0, 1}
        assert all(r.json()["profiling"]["service_total_ms"] >= 0 for r in replies)
        assert "model_worker_id" not in post(client, False).json()
        assert len(constructed) == 2
        assert len({item.constructor_thread for item in constructed}) == 2

    assert main.app.state.inference_service is None


def test_invalid_config_rejected_before_model_loading(monkeypatch):
    monkeypatch.delenv("MODEL_PATH", raising=False)
    monkeypatch.setenv("MODEL_POOL_WORKERS", "100")
    with pytest.raises(ValueError, match="MODEL_POOL_WORKERS"):
        with TestClient(main.app):
            pass
