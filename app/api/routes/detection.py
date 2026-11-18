"""Production-facing inference HTTP boundary; model logic lives in services."""

import asyncio
import logging
import time

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from app.api.dependencies import get_inference_service
from app.api.upload import ALLOWED_CONTENT_TYPES, read_upload_limited
from app.core.config import load_settings
from app.core.limiter import limiter
from app.image_service import ImageDimensionsTooLargeError, ImageTooLargeError, InvalidImageError
from app.schemas import DetectionResponse
from app.services.inference_service import InferenceService
from app.services.inference_worker_pool import (
    InferenceWorkerPool, InferenceOverloadedError, InferenceQueueTimeoutError,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["inference"])
# SlowAPI binds limits at route-definition time. Change env before Uvicorn starts.
DETECTION_RATE_LIMIT = load_settings().detection_rate_limit


@router.post("/detect", response_model=DetectionResponse, response_model_exclude_unset=True)
@limiter.limit(DETECTION_RATE_LIMIT)
async def detect(
    request: Request,
    image: UploadFile = File(...),
    confidence_threshold: float = Form(0.5, ge=0.0, le=1.0),
    profile: bool = Form(False, description="Include timing breakdown (local profiling only)."),
    service: InferenceService | InferenceWorkerPool = Depends(get_inference_service),
):
    admission = None
    try:
        # Settings are snapped at startup to avoid changing server policy mid-flight.
        settings = getattr(request.app.state, "settings", None)
        if profile and not (
            settings.enable_inference_profiling if settings is not None else False
        ):
            raise HTTPException(status_code=403, detail="Inference profiling is disabled.")
        if image.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(status_code=415, detail="Unsupported media type.")
        if isinstance(service, InferenceWorkerPool):
            admission = service.reserve()
        image_bytes = await read_upload_limited(image)

        model_worker_id = None
        if isinstance(service, InferenceWorkerPool):
            execution = await service.infer(
                image_bytes, confidence_threshold, profile=profile,
                admission=admission,
            )
            result = execution.result
            timings = execution.profiling
            queue_wait_ms = execution.queue_wait_ms if profile else None
            elapsed_ms = execution.inference_time_ms
            model_worker_id = execution.model_worker_id
        else:
            queued_at = time.perf_counter() if profile else None
            async with request.app.state.inference_semaphore:
                queue_wait_ms = (
                    round((time.perf_counter() - queued_at) * 1000, 3)
                    if queued_at is not None else None
                )
                started = time.perf_counter()
                if profile:
                    result, timings = await asyncio.to_thread(
                        service.infer_profiled, image_bytes, confidence_threshold,
                    )
                else:
                    result = await asyncio.to_thread(
                        service.infer, image_bytes, confidence_threshold,
                    )
                elapsed_ms = (time.perf_counter() - started) * 1000

        payload = {
            "status": "success",
            "filename": image.filename,
            "model_name": getattr(request.app.state, "model_name", None),
            "inference_time_ms": round(elapsed_ms, 3),
            **result.model_dump(),
        }
        if profile:
            payload["profiling"] = timings.model_dump()
            payload["queue_wait_ms"] = queue_wait_ms
            if model_worker_id is not None:
                payload["model_worker_id"] = model_worker_id
        return DetectionResponse(**payload)
    except (InferenceOverloadedError, InferenceQueueTimeoutError) as exc:
        code = (
            "INFERENCE_OVERLOADED" if isinstance(exc, InferenceOverloadedError)
            else "INFERENCE_QUEUE_TIMEOUT"
        )
        raise HTTPException(
            status_code=503,
            detail={"code": code, "message": str(exc)},
            headers={"Retry-After": "1"},
        ) from exc
    except (ImageTooLargeError, ImageDimensionsTooLargeError) as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected inference failure")
        raise HTTPException(status_code=500, detail="Model inference failed.") from None
    finally:
        if admission is not None:
            admission.release_if_pending()
        await image.close()
