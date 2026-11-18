"""Deterministic admission, waiting, timeout and cancellation tests (no CUDA)."""

import asyncio
from io import BytesIO
import threading

from PIL import Image
import pytest

from app.services.inference_service import InferenceService
from app.services.inference_worker_pool import (
    InferenceWorkerPool,
    InferenceOverloadedError,
    InferenceQueueTimeoutError,
)


def image_bytes():
    buf = BytesIO()
    Image.new("RGB", (32, 32), "purple").save(buf, format="PNG")
    return buf.getvalue()


class BlockedDetector:
    def __init__(self, began: threading.Event, release: threading.Event):
        self.began, self.release = began, release
        self.calls = 0

    def predict(self, image, confidence_threshold):
        self.calls += 1
        self.began.set()
        assert self.release.wait(3), "test timed out waiting for detector release"
        return []


async def wait_until(predicate, timeout=2.0):
    async def poll():
        while not predicate():
            await asyncio.sleep(0.002)
    await asyncio.wait_for(poll(), timeout)


def make_pool(began, release, *, max_queue_size=1, timeout=0.5):
    return InferenceWorkerPool(
        [lambda: InferenceService(BlockedDetector(began, release))],
        max_queue_size=max_queue_size,
        queue_timeout_seconds=timeout,
    )


def test_reject_bad_capacity_and_timeout():
    factory = lambda: None
    with pytest.raises(ValueError, match="max_queue_size"):
        InferenceWorkerPool([factory], max_queue_size=-1)
    with pytest.raises(ValueError, match="max_queue_size"):
        InferenceWorkerPool([factory], max_queue_size=True)
    for timeout in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="queue_timeout_seconds"):
            InferenceWorkerPool([factory], queue_timeout_seconds=timeout)


@pytest.mark.asyncio
async def test_admission_limit_rejects_without_waiting_and_releases_on_cancel():
    pool = make_pool(threading.Event(), threading.Event(), max_queue_size=1)
    await pool.start()
    try:
        assert pool.max_admitted == 2
        first, second = pool.reserve(), pool.reserve()
        assert pool.admitted_count == 2
        with pytest.raises(InferenceOverloadedError):
            pool.reserve()
        first.release_if_pending()
        first.release_if_pending()  # idempotent
        assert pool.admitted_count == 1
        third = pool.reserve()
        assert pool.admitted_count == 2
        second.release_if_pending()
        third.release_if_pending()
        assert pool.admitted_count == 0
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_running_plus_waiting_fills_capacity_and_both_complete():
    began, release = threading.Event(), threading.Event()
    pool = make_pool(began, release)
    await pool.start()
    try:
        first = asyncio.create_task(pool.infer(image_bytes(), 0.5, profile=False))
        assert await asyncio.to_thread(began.wait, 2)
        second = asyncio.create_task(pool.infer(image_bytes(), 0.5, profile=False))
        await wait_until(lambda: pool.admitted_count == 2)
        with pytest.raises(InferenceOverloadedError):
            await pool.infer(image_bytes(), 0.5, profile=False)
        assert not second.done()
        release.set()
        first_result, second_result = await asyncio.gather(first, second)
        assert first_result.queue_wait_ms >= 0
        assert second_result.queue_wait_ms > 0
        await wait_until(lambda: pool.admitted_count == 0)
    finally:
        release.set()
        await pool.close()


@pytest.mark.asyncio
async def test_queue_timeout_releases_waiter_capacity_and_worker_continues():
    began, release = threading.Event(), threading.Event()
    pool = make_pool(began, release, timeout=0.04)
    await pool.start()
    try:
        first = asyncio.create_task(pool.infer(image_bytes(), 0.5, profile=False))
        assert await asyncio.to_thread(began.wait, 2)
        with pytest.raises(InferenceQueueTimeoutError):
            await pool.infer(image_bytes(), 0.5, profile=False)
        assert pool.admitted_count == 1
        # The timed-out request freed exactly one waiting slot.
        replacement = pool.reserve()
        assert pool.admitted_count == 2
        with pytest.raises(InferenceOverloadedError):
            pool.reserve()
        replacement.release_if_pending()
        release.set()
        await first
        await wait_until(lambda: pool.admitted_count == 0)
    finally:
        release.set()
        await pool.close()


@pytest.mark.asyncio
async def test_cancelled_queued_caller_releases_capacity():
    began, release = threading.Event(), threading.Event()
    pool = make_pool(began, release, timeout=1.0)
    await pool.start()
    try:
        first = asyncio.create_task(pool.infer(image_bytes(), 0.5, profile=False))
        assert await asyncio.to_thread(began.wait, 2)
        waiting = asyncio.create_task(pool.infer(image_bytes(), 0.5, profile=False))
        await wait_until(lambda: pool.admitted_count == 2)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        assert pool.admitted_count == 1
        replacement = asyncio.create_task(pool.infer(image_bytes(), 0.5, profile=False))
        await wait_until(lambda: pool.admitted_count == 2)
        release.set()
        await asyncio.wait_for(asyncio.gather(first, replacement), 3)
        await wait_until(lambda: pool.admitted_count == 0)
    finally:
        release.set()
        await pool.close()


@pytest.mark.asyncio
async def test_cancelled_running_caller_still_occupies_capacity_until_thread_finishes():
    began, release = threading.Event(), threading.Event()
    pool = make_pool(began, release, max_queue_size=0)
    await pool.start()
    try:
        first = asyncio.create_task(pool.infer(image_bytes(), 0.5, profile=False))
        assert await asyncio.to_thread(began.wait, 2)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert pool.admitted_count == 1
        with pytest.raises(InferenceOverloadedError):
            pool.reserve()
        release.set()
        await wait_until(lambda: pool.admitted_count == 0)
        result = await pool.infer(image_bytes(), 0.5, profile=False)
        assert result.model_worker_id == 0
    finally:
        release.set()
        await pool.close()


@pytest.mark.asyncio
async def test_reservation_is_single_use_and_belongs_to_one_pool():
    release = threading.Event(); release.set()
    p1 = make_pool(threading.Event(), release)
    p2 = make_pool(threading.Event(), release)
    await p1.start(); await p2.start()
    try:
        ticket = p1.reserve()
        with pytest.raises(ValueError, match="Invalid"):
            await p2.infer(image_bytes(), 0.5, profile=False, admission=ticket)
        await p1.infer(image_bytes(), 0.5, profile=False, admission=ticket)
        with pytest.raises(ValueError, match="Invalid"):
            await p1.infer(image_bytes(), 0.5, profile=False, admission=ticket)
        await wait_until(lambda: p1.admitted_count == 0)
    finally:
        await p1.close(); await p2.close()


@pytest.mark.asyncio
async def test_shutdown_does_not_start_queued_inference():
    began, release = threading.Event(), threading.Event()
    pool = make_pool(began, release, timeout=1.0)
    await pool.start()
    try:
        first = asyncio.create_task(pool.infer(image_bytes(), 0.5, profile=False))
        assert await asyncio.to_thread(began.wait, 2)
        pending = asyncio.create_task(pool.infer(image_bytes(), 0.5, profile=False))
        await wait_until(lambda: pool.admitted_count == 2)
        shutdown = asyncio.create_task(pool.close())
        await wait_until(lambda: pool._closed)
        release.set()
        await asyncio.wait_for(first, 3)
        with pytest.raises(InferenceOverloadedError, match="shutting down"):
            await asyncio.wait_for(pending, 3)
        await asyncio.wait_for(shutdown, 3)
        assert pool.admitted_count == 0
    finally:
        release.set()
        await pool.close()
