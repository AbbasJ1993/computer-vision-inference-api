"""Model-independent inference orchestration and optional timing instrumentation.

The normal ``infer`` path preserves the API used in previous steps. Call
``infer_profiled`` explicitly only for local profiling experiments.
"""

from time import perf_counter
from typing import Protocol

from PIL import Image
from pydantic import BaseModel, Field

from app.image_service import decode_rgb_image, validate_image_structure


class BoundingBox(BaseModel):
    """Pixel coordinates in xyxy format (left, top, right, bottom)."""

    x1: int
    y1: int
    x2: int
    y2: int


class Detection(BaseModel):
    class_id: int = Field(ge=0)
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox


class InferenceResult(BaseModel):
    format: str
    width: int
    height: int
    size_bytes: int
    detections: list[Detection]


class YOLOProfile(BaseModel):
    """One-image timings for the Ultralytics adapter, in milliseconds.

    ``predict_wall_ms`` covers the complete call to Ultralytics ``predict``;
    internal ``result.speed`` values cover its measured processing stages and
    need not sum to the wall time. Missing speed fields remain ``None``.
    """

    predict_wall_ms: float = Field(ge=0)
    preprocess_ms: float | None = Field(default=None, ge=0)
    inference_ms: float | None = Field(default=None, ge=0)
    postprocess_ms: float | None = Field(default=None, ge=0)
    adapter_conversion_ms: float = Field(ge=0)


class InferenceProfile(BaseModel):
    """Service-level timings, in milliseconds (not HTTP or queueing time)."""

    validation_ms: float = Field(ge=0)
    decode_rgb_ms: float = Field(ge=0)
    detector_call_ms: float = Field(ge=0)
    result_build_ms: float = Field(ge=0)
    service_total_ms: float = Field(ge=0)
    yolo: YOLOProfile | None = None


class Detector(Protocol):
    """Any detector that accepts a Pillow RGB image and returns detections."""

    def predict(
        self,
        image: Image.Image,
        confidence_threshold: float,
    ) -> list[Detection]:
        ...


def _ms(seconds_start: float) -> float:
    return round((perf_counter() - seconds_start) * 1000, 3)


class InferenceService:
    """Validate, decode and delegate prediction to an injected detector."""

    def __init__(self, detector: Detector) -> None:
        self.detector = detector

    def infer(
        self,
        image_bytes: bytes,
        confidence_threshold: float = 0.5,
    ) -> InferenceResult:
        result, _ = self._run(image_bytes, confidence_threshold, profile=False)
        return result

    def infer_profiled(
        self,
        image_bytes: bytes,
        confidence_threshold: float = 0.5,
    ) -> tuple[InferenceResult, InferenceProfile]:
        """Run inference once and capture timing for the same request."""
        result, timings = self._run(image_bytes, confidence_threshold, profile=True)
        assert timings is not None
        return result, timings

    def _run(
        self,
        image_bytes: bytes,
        confidence_threshold: float,
        *,
        profile: bool,
    ) -> tuple[InferenceResult, InferenceProfile | None]:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1.")

        started = perf_counter() if profile else 0.0
        segment = perf_counter() if profile else 0.0
        metadata = validate_image_structure(image_bytes)
        validation_ms = _ms(segment) if profile else 0.0

        segment = perf_counter() if profile else 0.0
        # Full pixel decoding happens ONCE, here. Validation above performs
        # structural checks only. Corrupt/truncated image errors remain 400s.
        rgb_image = decode_rgb_image(image_bytes)
        decode_ms = _ms(segment) if profile else 0.0

        segment = perf_counter() if profile else 0.0
        yolo_profile: YOLOProfile | None = None
        if profile:
            # Optional capability: ordinary fake/custom Detectors still work.
            profiled_predict = getattr(self.detector, "predict_profiled", None)
            if callable(profiled_predict):
                detections, yolo_profile = profiled_predict(
                    rgb_image, confidence_threshold
                )
            else:
                detections = self.detector.predict(rgb_image, confidence_threshold)
        else:
            detections = self.detector.predict(rgb_image, confidence_threshold)
        detector_ms = _ms(segment) if profile else 0.0

        segment = perf_counter() if profile else 0.0
        result = InferenceResult(
            format=metadata["format"],
            width=metadata["width"],
            height=metadata["height"],
            size_bytes=metadata["size_bytes"],
            detections=detections,
        )
        result_build_ms = _ms(segment) if profile else 0.0

        if not profile:
            return result, None

        timing = InferenceProfile(
            validation_ms=validation_ms,
            decode_rgb_ms=decode_ms,
            detector_call_ms=detector_ms,
            result_build_ms=result_build_ms,
            service_total_ms=_ms(started),
            yolo=yolo_profile,
        )
        return result, timing
