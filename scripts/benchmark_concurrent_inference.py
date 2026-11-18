"""Measure GPU inference under concurrent HTTP clients with legacy or pool workers.

All requests use the *existing* POST /v1/detect with profile=true. Server-side
queue_wait_ms excludes HTTP parsing/upload time. Client latency includes those
steps and any waiting. Tested with one Uvicorn worker and a shared detector.

Example (Windows PowerShell):
    python -m scripts.benchmark_concurrent_inference `
      --image .\\test_data\\ai_generated_car.jpg `
      --label gpu_semaphore1 --concurrencies 1 2 4 8 `
      --requests 80 --repeats 3 --warmup 10
"""

import argparse
import asyncio
import csv
from datetime import datetime, timezone
import mimetypes
from pathlib import Path
import statistics
import time

import httpx

from scripts.benchmark_inference_api import percentile


DETAIL_COLUMNS = [
    "label", "repeat", "concurrency", "phase", "request_index",
    "status_code", "client_ms", "queue_wait_ms", "inference_time_ms",
    "service_total_ms", "detections", "model_worker_id", "error",
]
SUMMARY_COLUMNS = [
    "label", "repeat", "concurrency", "requests", "successes", "failures",
    "http_429", "http_503", "transport_errors", "wall_seconds",
    "completed_rps", "success_rps", "client_mean_ms", "client_p50_ms",
    "client_p95_ms", "client_p99_ms", "queue_mean_ms", "queue_p95_ms",
    "inference_mean_ms", "inference_p95_ms", "service_mean_ms",
]


def _maybe_stats(records: list[dict], key: str) -> tuple[float | None, float | None, float | None, float | None]:
    values = [float(r[key]) for r in records if r.get(key) is not None]
    if not values:
        return None, None, None, None
    return (
        round(statistics.mean(values), 3),
        round(percentile(values, 50), 3),
        round(percentile(values, 95), 3),
        round(percentile(values, 99), 3),
    )


def summarize_case(
    records: list[dict], *, label: str, repeat: int | str,
    concurrency: int, wall_seconds: float,
) -> dict:
    """Summarize measured requests; compute success throughput from wall time.

    HTTP errors and transport exceptions are counted, not silently dropped.
    Only successful requests contribute to latency percentiles.
    """
    successful = [r for r in records if r["status_code"] == 200]
    client = _maybe_stats(successful, "client_ms")
    queue = _maybe_stats(successful, "queue_wait_ms")
    inference = _maybe_stats(successful, "inference_time_ms")
    service = _maybe_stats(successful, "service_total_ms")
    return {
        "label": label,
        "repeat": repeat,
        "concurrency": concurrency,
        "requests": len(records),
        "successes": len(successful),
        "failures": len(records) - len(successful),
        "http_429": sum(r["status_code"] == 429 for r in records),
        "http_503": sum(r["status_code"] == 503 for r in records),
        "transport_errors": sum(r["status_code"] == 0 for r in records),
        "wall_seconds": round(wall_seconds, 4),
        "completed_rps": round(len(records) / wall_seconds, 3) if wall_seconds > 0 else None,
        "success_rps": round(len(successful) / wall_seconds, 3) if wall_seconds > 0 else None,
        "client_mean_ms": client[0],
        "client_p50_ms": client[1],
        "client_p95_ms": client[2],
        "client_p99_ms": client[3],
        "queue_mean_ms": queue[0],
        "queue_p95_ms": queue[2],
        "inference_mean_ms": inference[0],
        "inference_p95_ms": inference[2],
        "service_mean_ms": service[0],
    }


async def post_image(
    client: httpx.AsyncClient, image_name: str, image_bytes: bytes,
    threshold: float,
) -> dict:
    """Send one request, timing it at the client. Never share a file handle."""
    mime = mimetypes.guess_type(image_name)[0] or "application/octet-stream"
    started = time.perf_counter()
    try:
        response = await client.post(
            "/v1/detect",
            files={"image": (image_name, image_bytes, mime)},
            data={"confidence_threshold": str(threshold), "profile": "true"},
        )
        client_ms = round((time.perf_counter() - started) * 1000, 3)
        try:
            body = response.json()
        except ValueError:
            body = {}

        ok = response.status_code == 200 and body.get("status") == "success"
        if ok:
            # Missing queue fields should fail the preflight instead of
            # silently yielding misleading successful measurements.
            if body.get("queue_wait_ms") is None or body.get("inference_time_ms") is None:
                raise ValueError("Missing server timings; is the Step 2.2.6 API deployed?")
            return {
                "status_code": 200, "client_ms": client_ms,
                "queue_wait_ms": float(body["queue_wait_ms"]),
                "inference_time_ms": float(body["inference_time_ms"]),
                "service_total_ms": (
                    float(body["profiling"]["service_total_ms"])
                    if isinstance(body.get("profiling"), dict) and body["profiling"].get("service_total_ms") is not None
                    else None
                ),
                "detections": len(body.get("detections", [])),
                "model_worker_id": body.get("model_worker_id"), "error": "",
            }
        return {
            "status_code": response.status_code, "client_ms": client_ms,
            "queue_wait_ms": None, "inference_time_ms": None,
            "service_total_ms": None, "detections": None,
            "model_worker_id": None,
            "error": str(body.get("detail", response.reason_phrase))[:150],
        }
    except httpx.RequestError as exc:
        return {
            "status_code": 0,
            "client_ms": round((time.perf_counter() - started) * 1000, 3),
            "queue_wait_ms": None, "inference_time_ms": None,
            "service_total_ms": None, "detections": None,
            "model_worker_id": None,
            "error": type(exc).__name__,
        }


async def measure_case(
    client: httpx.AsyncClient, *, image_name: str, image_bytes: bytes,
    threshold: float, concurrency: int, requests: int,
) -> tuple[list[dict], float]:
    """Exactly `requests` calls via exactly `concurrency` async workers."""
    if concurrency < 1 or requests < 1:
        raise ValueError("concurrency and requests must be positive")
    records: list[dict] = []

    async def worker(worker_id: int) -> None:
        for request_index in range(worker_id + 1, requests + 1, concurrency):
            response = await post_image(client, image_name, image_bytes, threshold)
            records.append({"request_index": request_index, **response})

    started = time.perf_counter()
    await asyncio.gather(*(worker(i) for i in range(min(concurrency, requests))))
    elapsed_seconds = time.perf_counter() - started
    records.sort(key=lambda record: record["request_index"])
    return records, elapsed_seconds


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


async def run_benchmark(args: argparse.Namespace) -> tuple[Path, Path]:
    image_bytes = args.image.read_bytes()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    details_path = args.output_dir / f"concurrent_details_{args.label}_{stamp}.csv"
    summary_path = args.output_dir / f"concurrent_summary_{args.label}_{stamp}.csv"
    details: list[dict] = []
    summaries: list[dict] = []
    groups: dict[int, list[dict]] = {n: [] for n in args.concurrencies}
    durations: dict[int, float] = {n: 0.0 for n in args.concurrencies}

    max_concurrency = max(args.concurrencies)
    async with httpx.AsyncClient(
        base_url=args.url.rstrip("/"),
        timeout=httpx.Timeout(args.timeout),
        limits=httpx.Limits(max_connections=max_concurrency + 2),
    ) as client:
        ready = await client.get("/ready")
        if ready.status_code != 200:
            raise RuntimeError(f"Model not ready (HTTP {ready.status_code}): {ready.text[:200]}")
        probe = await post_image(client, args.image.name, image_bytes, args.threshold)
        if probe["status_code"] != 200:
            raise RuntimeError(f"Profiling preflight failed: {probe['status_code']} {probe['error']}")
        print(f"Model ready: {ready.json()}; profiling probe OK")
        print(f"Image={args.image.name}; size={len(image_bytes)} bytes; concurrency={args.concurrencies}")

        for repeat in range(1, args.repeats + 1):
            for concurrency in args.concurrencies:
                # Warm-up is sequential to avoid changing model concurrency;
                # warm-up observations never enter measured throughput.
                for index in range(1, args.warmup + 1):
                    row = await post_image(client, args.image.name, image_bytes, args.threshold)
                    if row["status_code"] != 200:
                        raise RuntimeError(f"Warm-up failed: HTTP {row['status_code']}, {row['error']}")
                    details.append({
                        "label": args.label, "repeat": repeat, "concurrency": concurrency,
                        "phase": "warmup", "request_index": index, **row,
                    })

                records, elapsed = await measure_case(
                    client, image_name=args.image.name, image_bytes=image_bytes,
                    threshold=args.threshold, concurrency=concurrency, requests=args.requests,
                )
                details.extend({
                    "label": args.label, "repeat": repeat, "concurrency": concurrency,
                    "phase": "measured", **row,
                } for row in records)
                summary = summarize_case(
                    records, label=args.label, repeat=repeat,
                    concurrency=concurrency, wall_seconds=elapsed,
                )
                summaries.append(summary)
                groups[concurrency].extend(records)
                durations[concurrency] += elapsed
                print(
                    f"repeat={repeat}, clients={concurrency}: "
                    f"success={summary['successes']}/{summary['requests']}, "
                    f"RPS={summary['success_rps']}, "
                    f"client P95={summary['client_p95_ms']} ms, "
                    f"queue P95={summary['queue_p95_ms']} ms, "
                    f"inference mean={summary['inference_mean_ms']} ms"
                )

    for concurrency in args.concurrencies:
        summaries.append(summarize_case(
            groups[concurrency], label=args.label, repeat="all",
            concurrency=concurrency, wall_seconds=durations[concurrency],
        ))
    write_csv(details_path, details, DETAIL_COLUMNS)
    write_csv(summary_path, summaries, SUMMARY_COLUMNS)
    print(f"\nDetails: {details_path.resolve()}")
    print(f"Summary: {summary_path.resolve()}")
    print("Only measured requests enter summary. 'queue_wait_ms' starts AFTER upload read.")
    print("'inference_time_ms' excludes semaphore wait; client latency includes it.")
    return details_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--label", default="gpu_semaphore1")
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--requests", type=int, default=80)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results"))
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"Image not found: {args.image}")
    if not (args.label.replace("_", "").replace("-", "").isalnum()):
        parser.error("Label must contain only letters, digits, underscore, or hyphen")
    if len(set(args.concurrencies)) != len(args.concurrencies) or any(n < 1 for n in args.concurrencies):
        parser.error("Concurrencies must be unique positive integers")
    if args.requests < 1 or args.repeats < 1 or args.warmup < 0 or args.timeout <= 0:
        parser.error("Requires requests>=1, repeats>=1, warmup>=0, timeout>0")
    if not 0 <= args.threshold <= 1:
        parser.error("Threshold must be between 0 and 1")
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
