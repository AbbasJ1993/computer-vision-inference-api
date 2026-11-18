"""FastAPI composition root: assemble routers, shared dependencies and lifespan."""

from fastapi import FastAPI
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api.dependencies import get_inference_service
from app.api.routes import detection, health, validation
from app.core.lifecycle import lifespan
from app.core.limiter import limiter


def create_app() -> FastAPI:
    app = FastAPI(
        title="Computer Vision Inference API",
        description="An API for running computer vision models.",
        version="0.9.0",
        lifespan=lifespan,
    )
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.include_router(health.router)
    app.include_router(validation.router)
    app.include_router(detection.router)
    return app


app = create_app()
