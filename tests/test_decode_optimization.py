"""Regression tests for step 2.2.5: no repeated pixel decoding."""
from io import BytesIO

import pytest
from PIL import Image

import app.image_service as image_service
from app.image_service import InvalidImageError, inspect_image
from app.services.inference_service import InferenceService


class SpyDetector:
    def __init__(self):
        self.calls = 0
        self.image_mode = None

    def predict(self, image, confidence_threshold):
        self.calls += 1
        self.image_mode = image.mode
        return []


def make_image(image_format: str) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (100, 80), color="red").save(buffer, format=image_format)
    return buffer.getvalue()


@pytest.mark.parametrize("image_format", ["JPEG", "PNG"])
def test_inference_reopens_encoded_bytes_only_twice(monkeypatch, image_format):
    """One open+verify and one open+full RGB decode, not three opens."""
    original_open = image_service.Image.open
    seen = []

    def tracked_open(*args, **kwargs):
        seen.append(1)
        return original_open(*args, **kwargs)

    monkeypatch.setattr(image_service.Image, "open", tracked_open)
    detector = SpyDetector()
    output, profile = InferenceService(detector).infer_profiled(make_image(image_format))
    assert len(seen) == 2
    assert output.format == image_format
    assert (output.width, output.height) == (100, 80)
    assert detector.calls == 1
    assert detector.image_mode == "RGB"
    assert profile.validation_ms >= 0
    assert profile.decode_rgb_ms >= 0


def test_truncated_jpeg_rejected_before_detector_call():
    detector = SpyDetector()
    damaged = make_image("JPEG")[:-2]
    with pytest.raises(InvalidImageError):
        InferenceService(detector).infer(damaged)
    assert detector.calls == 0


def test_metadata_inspection_remains_strict():
    valid = make_image("JPEG")
    assert inspect_image(valid)["format"] == "JPEG"
    with pytest.raises(InvalidImageError):
        inspect_image(valid[:-2])
