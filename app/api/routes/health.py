from fastapi import APIRouter, HTTPException, Request

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.get("/ready")
def readiness_check(request: Request):
    if getattr(request.app.state, "inference_service", None) is None:
        raise HTTPException(status_code=503, detail="Inference model is not ready.")
    ready = {"status": "ready", "model_name": request.app.state.model_name}
    pool_workers = getattr(request.app.state, "model_pool_workers", 0)
    if pool_workers:
        ready["model_pool_workers"] = pool_workers
    return ready
