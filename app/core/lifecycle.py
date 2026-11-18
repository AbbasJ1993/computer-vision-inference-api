"""ASGI lifespan: model loading, owned worker threads, and resource cleanup."""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.core.config import load_settings
from app.services.inference_service import InferenceService
from app.services.inference_worker_pool import InferenceWorkerPool
from app.services.yolo_detector import YOLODetector

logger = logging.getLogger(__name__)
MAX_CONCURRENT_INFERENCE = 1


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Parse/validate all environment values BEFORE loading any model.
    settings = load_settings()
    app.state.settings = settings
    app.state.inference_service = None
    app.state.model_name = None
    app.state.model_pool_workers = 0
    app.state.inference_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INFERENCE)

    pool: InferenceWorkerPool | None = None
    try:
        if settings.model_path.strip():
            model_path = Path(settings.model_path).expanduser()
            if not model_path.is_file():
                raise FileNotFoundError(
                    f"MODEL_PATH does not point to an existing file: {model_path}"
                )
            if settings.model_pool_workers:
                def make_service() -> InferenceService:
                    # Factory called on the worker's dedicated owning thread.
                    return InferenceService(
                        YOLODetector(model_path=model_path, device=settings.model_device)
                    )

                pool = InferenceWorkerPool(
                    [make_service] * settings.model_pool_workers,
                    max_queue_size=settings.max_queue_size,
                    queue_timeout_seconds=settings.queue_timeout_seconds,
                )
                await pool.start()
                app.state.inference_service = pool
            else:
                app.state.inference_service = InferenceService(
                    YOLODetector(model_path=model_path, device=settings.model_device)
                )
            app.state.model_pool_workers = settings.model_pool_workers
            app.state.model_name = model_path.name
            logger.info(
                "Model loaded: %s device=%s pool_workers=%d",
                model_path.name, settings.model_device, settings.model_pool_workers,
            )
        else:
            logger.info("MODEL_PATH unset: /v1/detect will return HTTP 503")

        yield
    finally:
        app.state.inference_service = None
        app.state.model_name = None
        app.state.model_pool_workers = 0
        if pool is not None:
            await pool.close()
