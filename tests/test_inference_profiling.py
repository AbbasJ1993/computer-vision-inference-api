"""Profiling is opt-in and independent from any real weights/GPU."""

from io import BytesIO

import httpx
import pytest
from PIL import Image

from app.services.inference_service import (
    BoundingBox, Detection, InferenceService, YOLOProfile,
)
from app.services.yolo_detector import YOLODetector
from scripts.compare_inference_profiles import compare
from scripts.profile_inference_api import flatten_profile, measured_summary, single_profiled_request


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FakeBoxes:
    def __init__(self):
        self.xyxy = FakeTensor([[12.4, 20.6, 80.3, 99.9]])
        self.conf = FakeTensor([0.91])
        self.cls = FakeTensor([2])


class FakeResult:
    def __init__(self, *, speed):
        self.speed = speed
        self.boxes = FakeBoxes()
        self.names = {2: "car"}


class FakeYOLO:
    def __init__(self, speed=None):
        self.speed = speed
        self.calls = 0

    def predict(self, **kwargs):
        self.calls += 1
        return [FakeResult(speed=self.speed)]


def sample_image() -> bytes:
    stream = BytesIO()
    Image.new("RGB", (128, 112), "red").save(stream, "PNG")
    return stream.getvalue()


def test_profiled_yolo_returns_internal_speed_and_detection_without_duplicate_prediction():
    fake_model = FakeYOLO({"preprocess": 1.2, "inference": 7.6, "postprocess": 0.4})
    detections, metrics = YOLODetector(model=fake_model).predict_profiled(Image.new("RGB", (128, 112)), 0.5)
    assert fake_model.calls == 1
    assert detections[0].bbox.model_dump() == {"x1": 12, "y1": 21, "x2": 80, "y2": 100}
    assert metrics.inference_ms == 7.6
    assert metrics.preprocess_ms == 1.2
    assert metrics.postprocess_ms == 0.4
    assert metrics.predict_wall_ms >= 0
    assert metrics.adapter_conversion_ms >= 0


def test_yolo_missing_speed_is_not_falsely_reported_as_zero():
    model = FakeYOLO(speed=None)
    _, metrics = YOLODetector(model=model).predict_profiled(Image.new("RGB", (128, 112)), 0.5)
    assert metrics.preprocess_ms is None
    assert metrics.inference_ms is None
    assert metrics.postprocess_ms is None


def test_service_profile_separates_validation_decode_detector_and_yolo():
    fake_model = FakeYOLO({"preprocess": 0.3, "inference": 0.5, "postprocess": 0.1})
    service = InferenceService(YOLODetector(model=fake_model))
    result, profile = service.infer_profiled(sample_image(), 0.6)
    assert result.width == 128
    assert len(result.detections) == 1
    assert fake_model.calls == 1
    assert profile.yolo.inference_ms == 0.5
    assert profile.service_total_ms >= 0
    assert profile.validation_ms >= 0
    assert profile.decode_rgb_ms >= 0
    assert profile.detector_call_ms >= 0
    assert profile.result_build_ms >= 0


def test_service_profile_works_for_plain_protocol_detector_without_fake_internal_metrics():
    class PlainDetector:
        def predict(self, image, confidence_threshold):
            return [Detection(
                class_id=0, class_name="person", confidence=0.9,
                bbox=BoundingBox(x1=1, y1=2, x2=20, y2=30),
            )]

    result, profile = InferenceService(PlainDetector()).infer_profiled(sample_image())
    assert len(result.detections) == 1
    assert profile.yolo is None


def test_profile_script_posts_true_and_flattens_timings(tmp_path):
    image_file = tmp_path / "sample.png"
    image_file.write_bytes(sample_image())

    def responder(request: httpx.Request) -> httpx.Response:
        assert b'name="profile"' in request.content
        assert b"true" in request.content
        return httpx.Response(200, json={
            "status": "success", "model_name": "yolo11n.pt",
            "inference_time_ms": 20, "detections": [],
            "profiling": {
                "service_total_ms": 19.5, "validation_ms": 1.5,
                "decode_rgb_ms": 2, "detector_call_ms": 15,
                "result_build_ms": 0.01,
                "yolo": {
                    "predict_wall_ms": 14.5, "preprocess_ms": 2.0,
                    "inference_ms": 9, "postprocess_ms": 1.0,
                    "adapter_conversion_ms": 0.25,
                },
            },
        })

    with httpx.Client(transport=httpx.MockTransport(responder), base_url="http://test") as client:
        row = single_profiled_request(client, image_file, image_file.read_bytes(), .5)
    assert row["yolo_inference_ms"] == 9
    assert row["service_total_ms"] == 19.5
    assert row["client_ms"] >= 0


def test_profile_csv_preserves_missing_fields_and_excludes_warmups():
    result = flatten_profile({
        "profiling": {
            "service_total_ms": 2, "validation_ms": 1, "decode_rgb_ms": 1,
            "detector_call_ms": 0.3, "result_build_ms": 0.1,
            "yolo": None,
        },
    })
    assert result["yolo_inference_ms"] is None
    rows = [
        {"phase": "initial", "server_ms": 200, "client_ms": 210},
        {"phase": "measured", "server_ms": 10, "client_ms": 12},
        {"phase": "measured", "server_ms": 20, "client_ms": 22},
    ]
    summary = measured_summary(rows)
    assert "mean=15.00 ms" in summary["server_ms"]
    assert summary["yolo_inference_ms"] == "no measurements"


def test_cpu_gpu_comparison_handles_missing_metrics():
    cpu = [{"phase": "measured", "server_ms": "40", "yolo_inference_ms": "30"}]
    gpu = [{"phase": "measured", "server_ms": "20", "yolo_inference_ms": "5"}]
    result = compare(cpu, gpu)
    by_name = {r["metric"]: r for r in result}
    assert by_name["server_ms"]["speedup_cpu_over_gpu"] == 2.0
    assert by_name["yolo_inference_ms"]["speedup_cpu_over_gpu"] == 6.0
    assert by_name["adapter_conversion_ms"]["speedup_cpu_over_gpu"] is None
