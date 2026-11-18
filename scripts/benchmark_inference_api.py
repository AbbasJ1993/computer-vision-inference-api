"""Measure HTTP and server-side inference timings with a single client.

Run in two fresh server processes (CPU then CUDA) using the exact same image.
This script measures serial requests only; it is not a load test.

Examples:
    python -m scripts.benchmark_inference_api --image test_data/ai_generated_car.jpg --label cpu
    python -m scripts.benchmark_inference_api --image test_data/ai_generated_car.jpg --label gpu
"""

import argparse
import csv
import math
import mimetypes
import statistics
import time
from pathlib import Path

import httpx


def percentile(values: list[float], p: float) -> float:
    """Linearly interpolate percentile p on a sorted copy (p in [0, 100])."""
    if not values:
        raise ValueError("values must not be empty")
    if not 0 <= p <= 100:
        raise ValueError("p must be between 0 and 100")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p / 100
    lo, hi = math.floor(rank), math.ceil(rank)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def describe(values: list[float]) -> str:
    if not values:
        return "no measurements"
    return (
        f"mean={statistics.mean(values):.2f} ms; "
        f"median={statistics.median(values):.2f} ms; "
        f"p95={percentile(values, 95):.2f} ms; "
        f"min={min(values):.2f} ms; max={max(values):.2f} ms"
    )


def single_request(
    client: httpx.Client,
    image_path: Path,
    image_bytes: bytes,
    threshold: float,
) -> dict:
    mime = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    started = time.perf_counter()
    response = client.post(
        "/v1/detect",
        files={"image": (image_path.name, image_bytes, mime)},
        data={"confidence_threshold": str(threshold)},
    )
    client_ms = (time.perf_counter() - started) * 1000
    if response.status_code != 200:
        raise RuntimeError(
            f"HTTP {response.status_code}: {response.text[:400]}. "
            "If 429, use DETECTION_RATE_LIMIT=100/minute on a LOCAL server "
            "and restart it before benchmarking."
        )
    payload = response.json()
    if payload.get("status") != "success":
        raise RuntimeError(f"Unexpected API response: {payload}")
    return {
        "client_ms": round(client_ms, 3),
        "server_ms": float(payload["inference_time_ms"]),
        "detections": len(payload["detections"]),
        "model": payload.get("model_name", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--label", default="cpu", help="Output label (e.g. cpu or gpu)")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=3, help="Additional warm-up requests after initial")
    parser.add_argument("--runs", type=int, default=20, help="Measured requests (sequential)")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"Image does not exist: {args.image}")
    if args.warmup < 0 or args.runs < 1 or not 0 <= args.threshold <= 1:
        parser.error("Require warmup >= 0, runs >= 1, and 0 <= threshold <= 1")
    if args.timeout <= 0:
        parser.error("Timeout must be positive")
    if not args.label.replace("_", "").replace("-", "").isalnum():
        parser.error("Label must contain only letters, digits, _ or -")

    image_bytes = args.image.read_bytes()
    output = args.output or Path("benchmark_results") / f"inference_api_{args.label}.csv"
    rows: list[dict] = []
    with httpx.Client(base_url=args.url.rstrip("/"), timeout=args.timeout) as client:
        ready = client.get("/ready")
        if ready.status_code != 200:
            raise RuntimeError(f"Model not ready: HTTP {ready.status_code}: {ready.text[:400]}")
        print(f"Ready: {ready.json()}; image={args.image.name}; size={len(image_bytes)} bytes")

        phases = (["initial"] + ["warmup"] * args.warmup + ["measured"] * args.runs)
        for i, phase in enumerate(phases, start=1):
            result = single_request(client, args.image, image_bytes, args.threshold)
            row = {"label": args.label, "phase": phase, "request": i, **result}
            rows.append(row)
            print(
                f"{phase:8s} #{i:02d}: server={result['server_ms']:9.2f} ms "
                f"client={result['client_ms']:9.2f} ms detections={result['detections']}"
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    measured = [r for r in rows if r["phase"] == "measured"]
    print(f"\nInitial server time: {rows[0]['server_ms']:.2f} ms")
    print(f"Steady-state server: {describe([r['server_ms'] for r in measured])}")
    print(f"Steady-state client: {describe([r['client_ms'] for r in measured])}")
    print(f"CSV saved: {output.resolve()}")
    print("Note: server_ms includes validation, decode, YOLO, output conversion, not model load.")
    print("Client timing additionally includes HTTP transfer, routing, queueing and serialization.")


if __name__ == "__main__":
    main()
