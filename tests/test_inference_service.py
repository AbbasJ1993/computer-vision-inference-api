"""Step 2: exercise inference orchestration without model weights."""

from io import BytesIO

import pytest
from PIL import Image

from app.image_service import InvalidImageError
from app.services.inference_service import (
    BoundingBox,
    Detection,
    InferenceService,
)


class FakeDetector:
    def __init__(self, detections=None):
        self.detections = detections if detections is not None else []
        self.observed_mode = None
        self.observed_size = None
        self.observed_threshold = None
        self.calls = 0

    def predict(self, image, confidence_threshold):
        self.calls += 1
        self.observed_mode = image.mode
        self.observed_size = image.size
        self.observed_threshold = confidence_threshold
        return self.detections


def make_image_bytes(mode="L", image_format="PNG"):
    buffer = BytesIO()
    Image.new(mode, (80, 60), color=128).save(buffer, format=image_format)
    return buffer.getvalue()


def test_inference_service_passes_rgb_image_and_threshold_to_detector():
    detector = FakeDetector([
        Detection(
            class_id=0,
            class_name="person",
            confidence=0.92,
            bbox=BoundingBox(x1=5, y1=6, x2=30, y2=40),
        )
    ])
    service = InferenceService(detector)

    result = service.infer(make_image_bytes(), confidence_threshold=0.7)

    assert result.format == "PNG"
    assert (result.width, result.height) == (80, 60)
    assert len(result.detections) == 1
    assert result.detections[0].class_name == "person"
    assert detector.observed_mode == "RGB"
    assert detector.observed_size == (80, 60)
    assert detector.observed_threshold == 0.7
    assert detector.calls == 1


def test_inference_service_accepts_no_detections():
    result = InferenceService(FakeDetector()).infer(make_image_bytes())
    assert result.detections == []


def test_inference_service_rejects_corrupted_image_before_model():
    detector = FakeDetector()
    with pytest.raises(InvalidImageError):
        InferenceService(detector).infer(b"not an image")
    assert detector.calls == 0


def test_inference_service_rejects_bad_threshold_before_model():
    detector = FakeDetector()
    with pytest.raises(ValueError):
        InferenceService(detector).infer(make_image_bytes(), confidence_threshold=1.5)
    assert detector.calls == 0
