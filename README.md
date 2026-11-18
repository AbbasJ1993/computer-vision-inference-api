# Computer Vision Inference API

[![CI](https://github.com/AbbasJ1993/computer-vision-inference-api/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/AbbasJ1993/computer-vision-inference-api/actions/workflows/ci.yml)

A modular FastAPI service for validated image uploads and YOLO object detection, with a pluggable `Detector` protocol, controlled GPU concurrency, optional profiling, and bounded admission.

> **Current step: 2.2.10 (Docker + CI).** The repository includes a CPU-only container and automated Linux/Windows tests. This is a demonstrator and benchmark project, not a fully hardened public-facing deployment.

## API

| Route | Purpose |
|---|---|
| `GET /health` | Process liveness, does not require a model |
| `GET /ready` | Readiness, 503 when model not configured |
| `POST /predict` | Legacy input validation only; no AI inference |
| `POST /v1/detect` | Object detection through the configured model |

`POST /v1/detect` accepts multipart `image`, optional `confidence_threshold` (0–1, default 0.5) and optional `profile` (default false). The response returns image metadata, `detections` with `xyxy` boxes and confidence, plus the total inference-service timing. Profiling adds internal timing and, in pool mode, `model_worker_id`.

Input controls include JPEG/PNG/WEBP, a 10 MiB encoded-image limit, dimensions up to 4096 × 4096, a 16-megapixel maximum, and corrupt-image checks. Pool mode includes backpressure (HTTP 503) and queue timeouts; a separate 429 rate limiter is available.

## Architecture

```text
app/main.py                   # composition root: creates FastAPI and registers routers
app/api/routes/health.py      # /health and /ready
app/api/routes/validation.py  # /predict (validation only)
app/api/routes/detection.py   # /v1/detect
app/api/upload.py             # limited upload reader
app/api/dependencies.py       # dependency injection
app/core/config.py            # typed Settings via pydantic-settings
app/core/lifecycle.py         # startup/shutdown, model loading
app/core/limiter.py           # SlowAPI limiter
app/image_service.py          # image structure validation and decode
app/services/                 # inference service, YOLO Adapter, worker pool
scripts/                      # real CPU/GPU profiling and load-testing scripts
```

A dedicated thread owns each YOLO instance when `MODEL_POOL_WORKERS=1` or `2`; both instances share the same GPU. The `MODEL_POOL_WORKERS=0` legacy semaphore path is retained for comparison. There is **no** model multiprocessing in this project. Synthetic CPU benchmark routes were moved out of scope; see the separate `fastapi-concurrency-benchmark` repository.

## Quick start (Windows, Python 3.12)

Run these commands from the repository root in PowerShell. Model files are
**not included** in Git because `weights/` is gitignored. The application
checks that `MODEL_PATH` exists at startup and does **not** download model
weights itself.

### 1. Create a virtual environment and install dependencies

```powershell
py -3.12 -m venv cv-inference-api-env
.\cv-inference-api-env\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-model.txt
```

If PowerShell blocks activation, use
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force`
in that terminal and retry. To run the tests without installing model
packages, use `python -m pip install -r requirements.txt` instead.

**GPU option (Windows/NVIDIA, previously tested configuration):** after
installing the model requirements, install the CUDA 12.1 PyTorch wheels:

```powershell
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
python -m pip check
```

Only set `MODEL_DEVICE=0` when CUDA is available. For CPU-only operation,
set `MODEL_DEVICE=cpu`; a CUDA-specific PyTorch build is not required.
See [PyTorch version-specific install commands](https://pytorch.org/get-started/previous-versions/)
for alternative hardware and environments.

### 2. Download the YOLO11n weights

With Internet access, Ultralytics can download the official pretrained
`yolo11n.pt` model on first use. Create the target directory **first** and
explicitly load the model into the location expected by this project:

```powershell
New-Item -ItemType Directory -Force '.\weights' | Out-Null
python -c "from ultralytics import YOLO; YOLO('weights/yolo11n.pt')"
Test-Path '.\weights\yolo11n.pt'  # Must return True
```

Source: [official Ultralytics YOLO11 models](https://github.com/ultralytics/yolo11).
If you already have a legally obtained, compatible YOLO11 detection `.pt`
file, copy it to `weights/yolo11n.pt` instead of downloading it again.
**Do not commit model weights to this repository.** Keeping weights out of
Git does not override their third-party license terms.

### 3. Start the API with a real model

For CPU inference:

```powershell
$env:MODEL_PATH = (Resolve-Path '.\weights\yolo11n.pt').Path
$env:MODEL_DEVICE = 'cpu'
$env:MODEL_POOL_WORKERS = '1'
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

For a CUDA-enabled GPU, use `$env:MODEL_DEVICE = '0'` instead. The model
pool supports 0 (legacy semaphore), 1, or 2 independent thread-owned
model instances. Start with 1; 2 increases GPU memory use.

In a **second** PowerShell window, verify readiness and run one inference:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/ready
curl.exe -X POST "http://127.0.0.1:8000/v1/detect" -F "image=@test_data/ai_generated_car.jpg;type=image/jpeg" -F "confidence_threshold=0.5"
```

With a successfully loaded model, `/ready` should return HTTP 200 and
`/v1/detect` should return a JSON response containing `detections`.
Object counts depend on model confidence and the image. API docs:
http://127.0.0.1:8000/docs

To run **without** a model, leave `MODEL_PATH` empty: `/health` returns 200,
`/ready` returns 503, and `/v1/detect` returns 503. The automated tests use
fake detectors and do not require model weights or CUDA.

For configuration defaults, see `.env.example`. Copy it to `.env` for local
settings but **never commit `.env`**. Environment variables override `.env`;
restart Uvicorn after changing settings. For Docker and CI, see
[Step 2.2.10](STEP_02_2_10.md).

## Docker (CPU-only model-free API)

The default container is intentionally lightweight: it **does not** include
Ultralytics, PyTorch, CUDA or `.pt` weights. It demonstrates that API
startup and health endpoints work without hardware or model dependencies.
It is **not** a real YOLO inference image.

With Docker Desktop running in Linux-container mode, from the repository
root in PowerShell:

```powershell
docker build -t cv-inference-api:cpu .
docker run --rm -d --name cv-api-local -p 127.0.0.1:8000:8000 -e MODEL_PATH= cv-inference-api:cpu
python scripts/smoke_http.py --base-url http://127.0.0.1:8000
docker logs cv-api-local
docker stop cv-api-local
```

The smoke script verifies `/health` = 200, `/ready` = 503 (expected without
a model) and `/openapi.json` = 200. Read [Step 2.2.10](STEP_02_2_10.md)
for troubleshooting. For **real** YOLO inference, use the host-side model
dependencies and weights described in Quick start; GPU container support
will be addressed separately.

## GitHub Actions CI

The workflow at `.github/workflows/ci.yml` runs the **92-test suite** on
Python 3.12 for Windows and Linux, builds the CPU-only Docker image and
performs an HTTP smoke test. It requires neither a GPU nor model weights.
Check the repository's **Actions** tab after pushing changes; the workflow
has not executed on GitHub merely by existing locally.

The runtime-only dependencies live in `requirements-runtime.txt`, the
unit/integration-test dependencies in `requirements.txt`, and optional
Ultralytics dependencies in `requirements-model.txt`.

## Measurements and limitations

Measured Windows 11 RTX 3070 Laptop GPU results using **YOLO11n** and the
**AI-generated 1152 × 600 car image** reached **70.74 successful
requests/s** for two thread-owned model instances at concurrency 4,
versus **44.05 requests/s** for one model instance in that experiment.
See the [nine curated benchmark CSV files and methodology](docs/benchmarks/README.md)
for profiling, latencies, overload handling, and limitations. These are
machine- and workload-specific measurements, not guaranteed capacity.
The Docker CI smoke test does not reproduce or validate GPU throughput.

Before public deployment, add proxy-level request limits, appropriate
authentication, TLS, centralized metrics, logging, deployment-specific
resource limits and an explicit review of third-party Ultralytics/model
license obligations (notably AGPL-3.0 versus Enterprise licensing). Consider a dependency lockfile and pinned base-image
digest for reproducible releases.

## Learning notes

Earlier learning-step notes may describe deprecated synthetic routes such as `/benchmark/direct`, `/benchmark/thread`, and `/benchmark/process`; those endpoints are **not present** in this repository. The current deployment notes are in [Step 2.2.10](STEP_02_2_10.md).
