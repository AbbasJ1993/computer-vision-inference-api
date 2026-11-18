"""Inversion-of-control point for the detector used by HTTP endpoints."""

from fastapi import HTTPException, Request

from app.services.inference_service import InferenceService
from app.services.inference_worker_pool import InferenceWorkerPool


def get_inference_service(request: Request) -> InferenceService | InferenceWorkerPool:
    service = getattr(request.app.state, "inference_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Inference model is not ready.")
    return service
