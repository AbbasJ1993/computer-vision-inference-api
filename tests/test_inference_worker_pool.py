"""Thread ownership, parallelism, cancellation, and fault recovery (no CUDA)."""

import asyncio
from io import BytesIO
import threading
import time

from PIL import Image
import pytest

from app.services.inference_service import InferenceService
from app.services.inference_worker_pool import InferenceWorkerPool


def valid_image():
    buffer = BytesIO()
    Image.new("RGB", (48, 32), color="blue").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_each_model_is_constructed_and_executed_on_one_own_thread():
    created = []

    class Detector:
        def __init__(self):
            self.creation_thread = threading.get_ident()
            created.append(self)

        def predict(self, image, confidence_threshold):
            assert threading.get_ident() == self.creation_thread
            return []

    pool = InferenceWorkerPool([lambda: InferenceService(Detector())] * 2)
    await pool.start()
    try:
        first = await pool.infer(valid_image(), 0.5, profile=True)
        second = await pool.infer(valid_image(), 0.5, profile=True)
        assert len(created) == 2
        assert created[0] is not created[1]
        assert created[0].creation_thread != created[1].creation_thread
        assert {first.model_worker_id, second.model_worker_id} == {0, 1}
        assert first.profiling is not None
        assert first.queue_wait_ms >= 0
        assert second.inference_time_ms >= 0
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_two_models_can_infer_concurrently():
    lock = threading.Lock()
    state = {"active": 0, "peak": 0}

    class Detector:
        def predict(self, image, confidence_threshold):
            with lock:
                state["active"] += 1
                state["peak"] = max(state["active"], state["peak"])
            time.sleep(0.07)
            with lock:
                state["active"] -= 1
            return []

    pool = InferenceWorkerPool([lambda: InferenceService(Detector())] * 2)
    await pool.start()
    try:
        outputs = await asyncio.gather(
            pool.infer(valid_image(), 0.5, profile=False),
            pool.infer(valid_image(), 0.5, profile=False),
        )
        assert {x.model_worker_id for x in outputs} == {0, 1}
        assert state["peak"] == 2
        assert all(x.profiling is None for x in outputs)
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_cancelled_request_does_not_release_busy_model_early():
    started = threading.Event()
    unblock = threading.Event()
    lock = threading.Lock()
    state = {"active": 0, "peak": 0}

    class Detector:
        def predict(self, image, confidence_threshold):
            with lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            started.set()
            if not unblock.is_set():
                assert unblock.wait(timeout=4)
            with lock:
                state["active"] -= 1
            return []

    pool = InferenceWorkerPool([lambda: InferenceService(Detector())])
    await pool.start()
    try:
        first = asyncio.create_task(pool.infer(valid_image(), 0.5, profile=False))
        assert await asyncio.to_thread(started.wait, 2)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        second = asyncio.create_task(pool.infer(valid_image(), 0.5, profile=False))
        await asyncio.sleep(0.03)
        assert not second.done(), "A cancelled request released the busy model prematurely"
        unblock.set()
        result = await asyncio.wait_for(second, 3)
        assert result.queue_wait_ms >= 20
        assert state["peak"] == 1
    finally:
        unblock.set()
        await pool.close()


@pytest.mark.asyncio
async def test_worker_slot_is_released_after_model_exception():
    class Detector:
        def __init__(self):
            self.calls = 0

        def predict(self, image, confidence_threshold):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("test model failed")
            return []

    pool = InferenceWorkerPool([lambda: InferenceService(Detector())])
    await pool.start()
    try:
        with pytest.raises(RuntimeError, match="test model failed"):
            await pool.infer(valid_image(), 0.5, profile=True)
        result = await asyncio.wait_for(pool.infer(valid_image(), 0.5, profile=True), 3)
        assert result.result.detections == []
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_factory_failure_aborts_startup_and_closes_pool():
    def broken():
        raise RuntimeError("cannot load model")

    pool = InferenceWorkerPool([lambda: InferenceService(object()), broken])
    with pytest.raises(RuntimeError, match="cannot load model"):
        await pool.start()
    with pytest.raises(RuntimeError, match="not running"):
        await pool.infer(valid_image(), 0.5, profile=False)


def test_invalid_pool_size_rejected():
    with pytest.raises(ValueError):
        InferenceWorkerPool([])
    with pytest.raises(ValueError):
        InferenceWorkerPool([lambda: None] * 3)
