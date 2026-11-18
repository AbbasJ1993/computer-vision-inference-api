"""Sequential CPU/GPU stage-level profiling for POST /v1/detect.

Requires the *local* server to run with ENABLE_INFERENCE_PROFILING=1.
Do not expose a profiling-enabled development server to public traffic.
"""

import argparse
import csv
import mimetypes
import statistics
import time
from pathlib import Path

import httpx

from scripts.benchmark_inference_api import describe


PROFILE_COLUMNS = [
    "server_ms", "client_ms", "service_total_ms", "validation_ms",
    "decode_rgb_ms", "detector_call_ms", "result_build_ms",
    "yolo_predict_wall_ms", "yolo_preprocess_ms", "yolo_inference_ms",
    "yolo_postprocess_ms", "adapter_conversion_ms",
]


def flatten_profile(payload: dict) -> dict:
    """Normalize the JSON profile to CSV fields, retaining missing values."""
    profile = payload["profiling"]
    yolo = profile.get("yolo") or {}
    return {
        "service_total_ms": profile["service_total_ms"],
        "validation_ms": profile["validation_ms"],
        "decode_rgb_ms": profile["decode_rgb_ms"],
        "detector_call_ms": profile["detector_call_ms"],
        "result_build_ms": profile["result_build_ms"],
        "yolo_predict_wall_ms": yolo.get("predict_wall_ms"),
        "yolo_preprocess_ms": yolo.get("preprocess_ms"),
        "yolo_inference_ms": yolo.get("inference_ms"),
        "yolo_postprocess_ms": yolo.get("postprocess_ms"),
        "adapter_conversion_ms": yolo.get("adapter_conversion_ms"),
    }


def single_profiled_request(
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
        data={"confidence_threshold": str(threshold), "profile": "true"},
    )
    client_ms = (time.perf_counter() - started) * 1000
    if response.status_code != 200:
        raise RuntimeError(
            f"HTTP {response.status_code}: {response.text[:400]}; "
            "for local profiling set ENABLE_INFERENCE_PROFILING=1 and "
            "DETECTION_RATE_LIMIT=100/minute, then restart Uvicorn."
        )
    payload = response.json()
    if payload.get("status") != "success" or "profiling" not in payload:
        raise RuntimeError("Server did not return a profiling-enabled detection response")
    return {
        "client_ms": round(client_ms, 3),
        "server_ms": float(payload["inference_time_ms"]),
        "detections": len(payload["detections"]),
        "model": payload.get("model_name") or "",
        **flatten_profile(payload),
    }


def measured_summary(rows: list[dict]) -> dict[str, str]:
    """Describe only measured rows; skip null Ultralytics metrics."""
    measured = [r for r in rows if r["phase"] == "measured"]
    return {
        column: describe([float(r[column]) for r in measured if r.get(column) is not None])
        for column in PROFILE_COLUMNS
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--label", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"Image not found: {args.image}")
    if args.warmup < 0 or args.runs < 1 or not 0 <= args.threshold <= 1:
        parser.error("Require warmup >= 0, runs >= 1, and 0 <= threshold <= 1")
    if args.timeout <= 0:
        parser.error("Timeout must be positive")
    if not args.label.replace("_", "").replace("-", "").isalnum():
        parser.error("Label must contain letters, digits, underscore or hyphen only")

    image_bytes = args.image.read_bytes()
    output = args.output or Path("benchmark_results") / f"inference_profile_{args.label}.csv"
    rows: list[dict] = []

    with httpx.Client(base_url=args.url.rstrip("/"), timeout=args.timeout) as client:
        ready = client.get("/ready")
        if ready.status_code != 200:
            raise RuntimeError(f"Model not ready: HTTP {ready.status_code}: {ready.text[:400]}")
        print(f"Ready: {ready.json()}; image={args.image.name}")

        phases = ["initial", *(["warmup"] * args.warmup), *(["measured"] * args.runs)]
        for index, phase in enumerate(phases, start=1):
            current = single_profiled_request(client, args.image, image_bytes, args.threshold)
            rows.append({"label": args.label, "phase": phase, "request": index, **current})
            print(
                f"{phase:8} #{index:02}: server={current['server_ms']:7.2f} ms, "
                f"YOLO-inference={current['yolo_inference_ms']} ms, "
                f"YOLO-preprocess={current['yolo_preprocess_ms']} ms, "
                f"YOLO-postprocess={current['yolo_postprocess_ms']} ms"
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\nSteady-state measurements (initial and warm-ups excluded):")
    for column, description in measured_summary(rows).items():
        print(f"{column:24}: {description}")
    print(f"\nCSV: {output.resolve()}")
    print(
        "Timing boundaries overlap: do not add YOLO internal speeds to "
        "model.predict wall time or detector call time."
    )


if __name__ == "__main__":
    main()
