from pathlib import Path

import httpx
import pytest

from scripts.benchmark_inference_api import describe, percentile, single_request


def test_percentiles_are_interpolated_and_order_independent():
    assert percentile([4, 1, 3, 2], 0) == 1
    assert percentile([4, 1, 3, 2], 50) == 2.5
    assert percentile([4, 1, 3, 2], 100) == 4
    assert percentile([1, 2, 3, 4], 95) == pytest.approx(3.85)


def test_percentile_rejects_empty_values():
    with pytest.raises(ValueError):
        percentile([], 50)


def test_single_request_uses_response_server_time(tmp_path):
    path = tmp_path / "image.jpg"
    path.write_bytes(b"example")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/detect"
        assert b"confidence_threshold" in request.content
        return httpx.Response(
            200,
            json={
                "status": "success",
                "inference_time_ms": 12.75,
                "detections": [{"class_name": "car"}],
                "model_name": "test.pt",
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test") as client:
        result = single_request(client, path, path.read_bytes(), 0.5)

    assert result["server_ms"] == 12.75
    assert result["client_ms"] >= 0
    assert result["detections"] == 1
    assert result["model"] == "test.pt"
    assert "median=" in describe([1.0, 2.0, 3.0])
