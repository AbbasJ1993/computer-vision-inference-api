"""Ultralytics-to-Detector adapter with opt-in, single-pass timing capture."""

from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image

from app.services.inference_service import (
    BoundingBox,
    Detection,
    YOLOProfile,
)


def _elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000, 3)


def _speed_ms(speed: Any, key: str) -> float | None:
    """Preserve absent backend timings as null rather than inventing zeros."""
    if not isinstance(speed, dict) or speed.get(key) is None:
        return None
    return float(speed[key])


class YOLODetector:
    """Object detector backed by Ultralytics YOLO."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        device: str = "cpu",
        model: Any | None = None,
    ) -> None:
        if model is None:
            if model_path is None:
                raise ValueError("model_path is required when no model is injected.")
            from ultralytics import YOLO

            model = YOLO(str(model_path))

        self.model = model
        self.device = device

    def predict(
        self,
        image: Image.Image,
        confidence_threshold: float,
    ) -> list[Detection]:
        """Ordinary request: output only model-independent detections."""
        detections, _ = self._predict(image, confidence_threshold, profile=False)
        return detections

    def predict_profiled(
        self,
        image: Image.Image,
        confidence_threshold: float,
    ) -> tuple[list[Detection], YOLOProfile]:
        """Run exactly one prediction and collect optional internal timing."""
        detections, metrics = self._predict(image, confidence_threshold, profile=True)
        assert metrics is not None
        return detections, metrics

    def _predict(
        self,
        image: Image.Image,
        confidence_threshold: float,
        *,
        profile: bool,
    ) -> tuple[list[Detection], YOLOProfile | None]:
        started = perf_counter() if profile else 0.0
        results = self.model.predict(
            source=image,
            conf=confidence_threshold,
            device=self.device,
            verbose=False,
        )
        predict_ms = _elapsed_ms(started) if profile else 0.0

        started = perf_counter() if profile else 0.0
        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue

            # CPU transfer may synchronize CUDA; it is accounted for here.
            xyxy = result.boxes.xyxy.cpu().tolist()
            scores = result.boxes.conf.cpu().tolist()
            class_ids = result.boxes.cls.cpu().tolist()

            for coords, score, class_id_value in zip(
                xyxy, scores, class_ids, strict=True
            ):
                class_id = int(class_id_value)
                x1, y1, x2, y2 = coords
                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=str(result.names[class_id]),
                        confidence=float(score),
                        bbox=BoundingBox(
                            x1=round(x1),
                            y1=round(y1),
                            x2=round(x2),
                            y2=round(y2),
                        ),
                    )
                )

        if not profile:
            return detections, None

        # This API receives one Pillow image. Ultralytics normally produces
        # precisely one Results object. Never imply valid per-image timing for
        # an unexpected multi-image response.
        speed = getattr(results[0], "speed", None) if len(results) == 1 else None
        profile_metrics = YOLOProfile(
            predict_wall_ms=predict_ms,
            preprocess_ms=_speed_ms(speed, "preprocess"),
            inference_ms=_speed_ms(speed, "inference"),
            postprocess_ms=_speed_ms(speed, "postprocess"),
            adapter_conversion_ms=_elapsed_ms(started),
        )
        return detections, profile_metrics
