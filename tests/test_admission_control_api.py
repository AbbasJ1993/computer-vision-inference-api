"""HTTP 503 overload/timeout behavior with fake thread-owned models (no CUDA)."""

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import threading
import time

from fastapi.testclient import TestClient
from PIL import Image
import pytest

import app.main as main
import app.core.lifecycle as lifecycle
from app.services.inference_worker_pool import InferenceWorkerPool


def image_bytes():
    stream = BytesIO()
    Image.new("RGB", (40, 30), "red").save(stream, format="PNG")
    return stream.getvalue()


def post(client, content=None, profile=False):
    return client.post(
        "/v1/detect",
        files={"image": ("test.png", image_bytes() if content is None else content, "image/png")},
        data={"profile": str(profile).lower()},
    )


class FakeYOLODetector:
    started = threading.Event()
    release = threading.Event()
    calls = 0
    lock = threading.Lock()

    def __init__(self, model_path, *, device):
        self.owner = threading.get_ident()

    def predict(self, image, confidence_threshold):
        assert self.owner == threading.get_ident()
        with self.lock:
            type(self).calls += 1
        self.started.set()
        assert self.release.wait(3), "forgot to unblock fake detector"
        return []


def setup_pool_env(monkeypatch, tmp_path, *, max_queue_size, timeout):
    fake_weights = tmp_path / "fake.pt"
    fake_weights.write_bytes(b"fake")
    monkeypatch.setenv("MODEL_PATH", str(fake_weights))
    monkeypatch.setenv("MODEL_POOL_WORKERS", "1")
    monkeypatch.setenv("MAX_QUEUE_SIZE", str(max_queue_size))
    monkeypatch.setenv("QUEUE_TIMEOUT_SECONDS", str(timeout))
    monkeypatch.setenv("ENABLE_INFERENCE_PROFILING", "1")
    monkeypatch.setattr(lifecycle, "YOLODetector", FakeYOLODetector)
    main.limiter.reset()
    FakeYOLODetector.started.clear()
    FakeYOLODetector.release.clear()
    FakeYOLODetector.calls = 0


def await_count(pool, target, timeout=2):
    deadline = time.monotonic() + timeout
    while pool.admitted_count != target and time.monotonic() < deadline:
        time.sleep(0.003)
    assert pool.admitted_count == target


def test_overload_returns_503_with_retry_after_and_running_worker_is_preserved(monkeypatch, tmp_path):
    setup_pool_env(monkeypatch, tmp_path, max_queue_size=1, timeout=1)
    try:
        with TestClient(main.app) as client:
            pool = main.app.state.inference_service
            assert isinstance(pool, InferenceWorkerPool)
            with ThreadPoolExecutor(max_workers=2) as ex:
                first = ex.submit(post, client)
                assert FakeYOLODetector.started.wait(2)
                second = ex.submit(post, client)
                await_count(pool, 2)
                third = post(client)
                assert third.status_code == 503, third.text
                assert third.headers["retry-after"] == "1"
                assert third.json()["detail"]["code"] == "INFERENCE_OVERLOADED"
                assert pool.admitted_count == 2
                FakeYOLODetector.release.set()
                assert first.result(timeout=4).status_code == 200
                assert second.result(timeout=4).status_code == 200
            await_count(pool, 0)
        assert FakeYOLODetector.calls == 2
    finally:
        FakeYOLODetector.release.set()


def test_waiting_request_times_out_with_503_then_capacity_returns(monkeypatch, tmp_path):
    setup_pool_env(monkeypatch, tmp_path, max_queue_size=1, timeout=0.05)
    try:
        with TestClient(main.app) as client:
            pool = main.app.state.inference_service
            with ThreadPoolExecutor(max_workers=2) as ex:
                first = ex.submit(post, client)
                assert FakeYOLODetector.started.wait(2)
                second = ex.submit(post, client)
                timeout_reply = second.result(timeout=3)
                assert timeout_reply.status_code == 503, timeout_reply.text
                assert timeout_reply.headers["retry-after"] == "1"
                assert timeout_reply.json()["detail"]["code"] == "INFERENCE_QUEUE_TIMEOUT"
                assert pool.admitted_count == 1
                FakeYOLODetector.release.set()
                assert first.result(timeout=3).status_code == 200
            await_count(pool, 0)
            assert post(client).status_code == 200
        assert FakeYOLODetector.calls == 2
    finally:
        FakeYOLODetector.release.set()


def test_bad_image_releases_admission_capacity(monkeypatch, tmp_path):
    setup_pool_env(monkeypatch, tmp_path, max_queue_size=0, timeout=0.2)
    FakeYOLODetector.release.set()
    with TestClient(main.app) as client:
        pool = main.app.state.inference_service
        assert post(client, content=b"not an image").status_code == 400
        await_count(pool, 0)
        assert post(client).status_code == 200
        await_count(pool, 0)
    assert FakeYOLODetector.calls == 1


@pytest.mark.parametrize("key,value", [
    ("MAX_QUEUE_SIZE", "-1"),
    ("MAX_QUEUE_SIZE", "not-a-number"),
    ("QUEUE_TIMEOUT_SECONDS", "0"),
    ("QUEUE_TIMEOUT_SECONDS", "nan"),
])
def test_invalid_capacity_config_fails_at_startup(monkeypatch, tmp_path, key, value):
    setup_pool_env(monkeypatch, tmp_path, max_queue_size=1, timeout=1)
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError):
        with TestClient(main.app):
            pass
