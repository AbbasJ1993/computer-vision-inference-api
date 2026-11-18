"""Deterministic tests for the concurrent HTTP benchmarking client."""

import asyncio
import httpx
import pytest

from scripts.benchmark_concurrent_inference import (
    measure_case, post_image, summarize_case,
)


@pytest.mark.asyncio
async def test_concurrent_client_enforces_concurrency_and_counts_requests():
    active = 0
    peak = 0
    lock = asyncio.Lock()

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, peak
        assert request.url.path == "/v1/detect"
        assert b'name="profile"' in request.content
        assert b"true" in request.content
        async with lock:
            active += 1
            peak = max(peak, active)
        await asyncio.sleep(0.02)
        async with lock:
            active -= 1
        return httpx.Response(200, json={
            "status": "success", "queue_wait_ms": 5.0,
            "inference_time_ms": 10.0,
            "profiling": {"service_total_ms": 9.5},
            "detections": [],
        })

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://test"
    ) as client:
        records, elapsed = await measure_case(
            client, image_name="test.jpg", image_bytes=b"image", threshold=0.5,
            concurrency=4, requests=11,
        )

    assert len(records) == 11
    assert [r["request_index"] for r in records] == list(range(1, 12))
    assert all(r["status_code"] == 200 for r in records)
    assert all(r["queue_wait_ms"] == 5.0 for r in records)
    assert peak == 4
    assert elapsed > 0
    summary = summarize_case(records, label="test", repeat=1, concurrency=4, wall_seconds=elapsed)
    assert summary["requests"] == 11
    assert summary["successes"] == 11
    assert summary["failures"] == 0
    assert summary["client_p95_ms"] >= 15
    assert summary["success_rps"] > 0


@pytest.mark.asyncio
async def test_http_errors_are_recorded_not_silently_ignored():
    count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal count
        count += 1
        if count == 1:
            return httpx.Response(429, json={"detail": "Too Many Requests"})
        return httpx.Response(503, json={"detail": "Model unavailable"})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://test"
    ) as client:
        records, seconds = await measure_case(
            client, image_name="test.jpg", image_bytes=b"image", threshold=0.5,
            concurrency=1, requests=2,
        )

    summary = summarize_case(records, label="test", repeat=1, concurrency=1, wall_seconds=seconds)
    assert summary["http_429"] == 1
    assert summary["http_503"] == 1
    assert summary["successes"] == 0
    assert summary["failures"] == 2
    assert summary["client_mean_ms"] is None
    assert summary["queue_p95_ms"] is None


@pytest.mark.asyncio
async def test_preflight_rejects_missing_queue_instrumentation():
    def handler(request):
        return httpx.Response(200, json={"status": "success", "inference_time_ms": 2.0})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://test"
    ) as client:
        with pytest.raises(ValueError, match="Missing server timings"):
            await post_image(client, "image.jpg", b"test", 0.5)


def test_empty_record_summary_does_not_divide_by_zero():
    summary = summarize_case([], label="test", repeat="all", concurrency=1, wall_seconds=0)
    assert summary["success_rps"] is None
    assert summary["failures"] == 0
    assert summary["client_p99_ms"] is None
