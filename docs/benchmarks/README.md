# Inference benchmarks — AI-generated car image

These CSV files contain measured results from a local **Computer Vision Inference API** benchmark suite. They use the same AI-generated, generic sedan image (`test_data/ai_generated_car.jpg`) instead of the earlier third-party car photograph. The results are **machine-specific measurements**, not general performance guarantees.

## Hardware, model, and methodology

- **Environment:** Windows 11; NVIDIA GeForce RTX 3070 Laptop GPU (8 GB); Python 3.12; PyTorch 2.5.1 + CUDA 12.1.
- **Model:** Ultralytics YOLO11n (`yolo11n.pt`), executed on CUDA device `0`.
- **Input:** AI-generated JPEG, **1152 × 600**, object detection at **confidence threshold 0.5**. The project maintains the image and its provenance under `test_data/`.
- **Profiling run:** 1 initial request, 10 warm-up requests, 50 measured requests.
- **Concurrency runs:** 10 warm-up requests per repeat, 80 measured requests per repeat, **3 repeats** for each of the tested concurrency levels. Four configurations were tested: serial `Semaphore(1)`, one-model worker pool, two-model worker pool, and bounded admission/backpressure.
- **Backpressure configuration:** 2 model workers, `MAX_QUEUE_SIZE=2`, `QUEUE_TIMEOUT_SECONDS=0.1`. These are load-test settings, not recommended universal production defaults.

## Included CSV files (9)

| Filename | Contents |
|---|---|
| [`inference_profile_gpu_ai_car.csv`](inference_profile_gpu_ai_car.csv) | Per-request GPU profiling: validation, RGB decode, detector, and YOLO internal timings. |
| [`concurrent_summary_gpu_semaphore1_ai_car_20261117T145009Z.csv`](concurrent_summary_gpu_semaphore1_ai_car_20261117T145009Z.csv) | Single-model Semaphore baseline; per-repeat and aggregated concurrency metrics. |
| [`concurrent_details_gpu_semaphore1_ai_car_20261117T145009Z.csv`](concurrent_details_gpu_semaphore1_ai_car_20261117T145009Z.csv) | Every warm-up and measured request in the Semaphore baseline. |
| [`concurrent_summary_gpu_pool1_ai_car_20261117T145127Z.csv`](concurrent_summary_gpu_pool1_ai_car_20261117T145127Z.csv) | One model worker; per-repeat and aggregated metrics. |
| [`concurrent_details_gpu_pool1_ai_car_20261117T145127Z.csv`](concurrent_details_gpu_pool1_ai_car_20261117T145127Z.csv) | Every request in the one-worker experiment. |
| [`concurrent_summary_gpu_pool2_ai_car_20261117T145225Z.csv`](concurrent_summary_gpu_pool2_ai_car_20261117T145225Z.csv) | Two independent model workers; per-repeat and aggregated metrics. |
| [`concurrent_details_gpu_pool2_ai_car_20261117T145225Z.csv`](concurrent_details_gpu_pool2_ai_car_20261117T145225Z.csv) | Every request in the two-worker experiment; includes `model_worker_id`. |
| [`concurrent_summary_gpu_backpressure_ai_car_20261117T145319Z.csv`](concurrent_summary_gpu_backpressure_ai_car_20261117T145319Z.csv) | Bounded-admission results including HTTP 503 counts. |
| [`concurrent_details_gpu_backpressure_ai_car_20261117T145319Z.csv`](concurrent_details_gpu_backpressure_ai_car_20261117T145319Z.csv) | Individual responses, including overload/queue-timeout error codes. |

The four `concurrent_summary_*` files each contain a `repeat=all` row for each concurrency level. The `concurrent_details_*` files provide the underlying measured requests (and warm-ups). These details files are retained so a reader can independently inspect latency distributions and failure counts.

## Results with the AI-generated image

### GPU profiling (50 measured requests)

| Stage | Mean (ms) |
|---|---:|
| Client latency | 24.62 |
| Server inference timing | 21.67 |
| Service total | 21.25 |
| Validation | 0.092 |
| Decode / RGB | 4.18 |
| Detector call | 16.96 |
| YOLO preprocessing | 1.37 |
| YOLO neural-network inference | 10.74 |
| YOLO postprocessing | 1.84 |

**Do not add all rows together:** the internal YOLO measurements are part of the detector call, which is part of the service total. Server and client latencies also include different measurement scopes.

### Successful throughput (requests/second)

These values come from the `repeat=all` summary rows. They measure successful responses per wall-clock second, **not** total responses per second including errors.

| Concurrent clients | Semaphore(1) | Pool(1) | Pool(2) | Pool(2) client P95 (ms) |
|---:|---:|---:|---:|---:|
| 1 | 42.45 | 41.30 | 40.98 | 31.12 |
| 2 | 47.82 | 47.14 | 58.96 | 53.26 |
| 4 | 47.96 | 44.05 | 70.74 | 76.38 |
| 8 | 46.93 | 45.09 | 70.78 | 133.45 |

At concurrency **4**, Pool(2) reached **70.74 successful requests/s** versus **44.05 requests/s** for Pool(1), a **60.6%** increase. Client P95 decreased from **119.42 ms** to **76.38 ms** (**36.0%** decrease). These results describe **two independent YOLO instances in separate threads within one server process**, not two operating-system processes, and do not establish CUDA kernel overlap.

### Bounded admission and backpressure

| Concurrent clients | Successful / total | HTTP 503 | Success RPS | Successful client P95 (ms) |
|---:|---:|---:|---:|---:|
| 1 | 240/240 | 0 | 41.54 | 32.15 |
| 2 | 240/240 | 0 | 61.12 | 40.73 |
| 4 | 240/240 | 0 | 63.49 | 106.00 |
| 8 | 64/240 | 176 | 49.24 | 109.34 |
| 16 | 49/240 | 191 | 43.93 | 161.77 |

Across the **1,200 measured** backpressure requests, **833 returned HTTP 200** and **367 returned HTTP 503**. Of the 503 responses, **365** were `INFERENCE_OVERLOADED` and **2** were `INFERENCE_QUEUE_TIMEOUT`; no transport errors were recorded. Rejection under overload is **expected admission-control behavior**, not a successful inference. The P95 in the table is computed for **successful responses** only.

## Reproduce

Run from the **repository root** in PowerShell. Use a separate `cv-inference-api-env` virtual environment and an existing, separately obtained YOLO11n weight file at `weights/yolo11n.pt` (do not commit weights). Start a server with one Uvicorn process; do **not** add `--reload` or `--workers 2` to these comparisons.

Example for the **two-worker** configuration:

```powershell
$env:MODEL_PATH = (Resolve-Path ".\weights\yolo11n.pt").Path
$env:MODEL_DEVICE = "0"
$env:MODEL_POOL_WORKERS = "2"
$env:DETECTION_RATE_LIMIT = "5000/minute" # local benchmark only
$env:ENABLE_INFERENCE_PROFILING = "1"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```powershell
python -m scripts.benchmark_concurrent_inference `
  --image ".\test_data\ai_generated_car.jpg" `
  --label gpu_pool2_ai_car `
  --concurrencies 1 2 4 8 `
  --requests 80 `
  --repeats 3 `
  --warmup 10 `
  --threshold 0.5
```

For the other concurrency configurations, restart the server and change **only the corresponding configuration**: `MODEL_POOL_WORKERS=0` for Semaphore(1), `1` for Pool(1), or `2` for Pool(2). For the backpressure experiment, configure `MODEL_POOL_WORKERS=2`, `MAX_QUEUE_SIZE=2` and `QUEUE_TIMEOUT_SECONDS=0.1`, and add concurrency `16` to the client command. To run inference profiling, use `scripts.profile_inference_api` with `--warmup 10 --runs 50 --threshold 0.5` and `--image ".\test_data\ai_generated_car.jpg"`.

The normal output directory `benchmark_results/` is intentionally ignored by Git. Files in this `docs/benchmarks/` directory are **curated snapshots** for publication.

## Interpretation and limitations

- These are local laptop results. Thermal behavior, GPU clocking, other workloads, operating system scheduling, and first-request initialization can affect the outcome.
- The concurrency script issues a **closed-loop** workload with a fixed number of clients, not an open-loop arrival-rate test. Its measured RPS should not be interpreted as a guaranteed production capacity.
- `Pool(2)` refers to separate model objects in **two dedicated threads of one Uvicorn process**, sharing the same physical GPU. This is distinct from multiprocessing.
- The 503 count is intentional under the tested queue settings; `completed_rps` may be high simply because rejected requests complete quickly. Use **`success_rps`** when comparing inference capacity.
- Results from the earlier third-party image are **not directly interchangeable** with these results; the input image, decoding cost and detection content can affect latency. Older synthetic concurrency tests belong to the separate `fastapi-concurrency-benchmark` repository.
- An AI-generated input image reduces reliance on third-party photographs, but results do not prove detection accuracy; this is a latency/throughput evaluation.

**Image provenance:** `test_data/ai_generated_car.jpg` is an AI-generated, generic car image created with OpenAI image generation for this repository. Keep its provenance note in `test_data/README.md`.
