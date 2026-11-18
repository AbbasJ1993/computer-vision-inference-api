"""Legacy validation endpoint (no model inference)."""

import asyncio

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app.api.upload import ALLOWED_CONTENT_TYPES, read_upload_limited
from app.core.limiter import limiter
from app.image_service import (
    ImageDimensionsTooLargeError, ImageTooLargeError, InvalidImageError, inspect_image,
)
from app.schemas import PredictionResponse

router = APIRouter(tags=["validation"])
MAX_CONCURRENT_PREDICTIONS = 3
prediction_semaphore = asyncio.Semaphore(MAX_CONCURRENT_PREDICTIONS)


@router.post("/predict", response_model=PredictionResponse)
@limiter.limit("10/minute")
async def predict(
    request: Request,
    image: UploadFile = File(...),
    confidence_threshold: float = Form(0.5, ge=0.0, le=1.0),
):
    try:
        async with prediction_semaphore:
            if image.content_type not in ALLOWED_CONTENT_TYPES:
                raise HTTPException(status_code=415, detail="Unsupported media type.")
            image_bytes = await read_upload_limited(image)
            metadata = inspect_image(image_bytes)
            return {
                "status": "valid",
                "filename": image.filename,
                **metadata,
                "confidence_threshold": confidence_threshold,
            }
    except (ImageTooLargeError, ImageDimensionsTooLargeError) as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        await image.close()
