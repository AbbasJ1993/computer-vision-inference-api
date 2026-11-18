"""Queue instrumentation and serialization of access to the shared detector."""

from concurrent.futures import ThreadPoolExecutor
import asyncio
from io import BytesIO
import threading
import time

from fastapi.testclient import TestClient
from PIL import Image
import pytest

import app.main as main
from app.services.inference_service import InferenceService


def create_image() -> bytes:
    stream = BytesIO()
    Image.new("RGB", (48, 36), "red").save(stream, format="PNG")
    return stream.getvalue()


class TrackingSemaphore(asyncio.Semaphore):
    """Expose when the second request really reaches the semaphore."""

    def __init__(self):
        super().__init__(1)
        self.queued = threading.Event()

    async def acquire(self):
        if self.locked():
            self.queued.set()
        return await super().acquire()


class SlowDetector:
    def __init__(self, release=None):
        self.release = release
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.calls = 0
        self.started = threading.Event()

    def predict(self, image, confidence_threshold):
        with self.lock:
            self.active += 1
            self.calls += 1
            self.peak = max(self.peak, self.active)
            self.started.set()
        if self.release is not None:
            assert self.release.wait(timeout=5), "Detector was not released in test"
        time.sleep(0.04)
        with self.lock:
            self.active -= 1
        return []


def post(client, *, profile):
    return client.post(
        "/v1/detect",
        files={"image": ("test.png", create_image(), "image/png")},
        data={"profile": str(profile).lower()},
    )


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("MODEL_PATH", raising=False)
    monkeypatch.setenv("ENABLE_INFERENCE_PROFILING", "1")
    main.limiter.reset()
    with TestClient(main.app) as instance:
        yield instance
    main.app.dependency_overrides.clear()


def test_queue_metric_only_present_when_profile_is_requested(client):
    detector = SlowDetector()
    main.app.dependency_overrides[main.get_inference_service] = lambda: InferenceService(detector)
    ordinary = post(client, profile=False)
    profiled = post(client, profile=True)
    assert ordinary.status_code == 200, ordinary.text
    assert profiled.status_code == 200, profiled.text
    assert "queue_wait_ms" not in ordinary.json()
    assert "profiling" not in ordinary.json()
    assert profiled.json()["queue_wait_ms"] >= 0
    assert profiled.json()["profiling"]["service_total_ms"] >= 0
    assert profiled.json()["inference_time_ms"] >= 0


def test_concurrent_requests_wait_for_same_semaphore(client):
    release = threading.Event()
    detector = SlowDetector(release=release)
    queue = TrackingSemaphore()
    main.app.state.inference_semaphore = queue
    main.app.dependency_overrides[main.get_inference_service] = lambda: InferenceService(detector)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(post, client, profile=True)
        assert detector.started.wait(timeout=5), "First detector never started"
        second = executor.submit(post, client, profile=True)
        try:
            assert queue.queued.wait(timeout=5), "Second request never reached the semaphore"
            time.sleep(0.04)  # Generate a measurable, controlled queue delay.
        finally:
            release.set()
        responses = [first.result(timeout=5), second.result(timeout=5)]

    assert [r.status_code for r in responses] == [200, 200]
    assert detector.calls == 2
    assert detector.peak == 1
    waits = sorted(r.json()["queue_wait_ms"] for r in responses)
    assert waits[0] >= 0
    assert waits[1] >= 20, f"Expected second request to queue, got {waits}"
    assert all(r.json()["inference_time_ms"] >= 30 for r in responses)
