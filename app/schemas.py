from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    status: str
    filename: str | None
    format: str
    width: int
    height: int
    size_bytes: int
    confidence_threshold: float

from typing import Literal

from app.services.inference_service import InferenceResult, InferenceProfile


class DetectionResponse(InferenceResult):
    """HTTP response for the new real-inference endpoint."""

    status: Literal["success"]
    filename: str | None
    model_name: str | None
    inference_time_ms: float = Field(ge=0)

    profiling: InferenceProfile | None = None
    # Only returned for profile=true. Measured immediately before/after the
    # per-worker inference semaphore. This is not total request queueing time.
    queue_wait_ms: float | None = Field(default=None, ge=0)
    # Experimental pool mode only; present with profile=true.
    model_worker_id: int | None = Field(default=None, ge=0)
