"""HTTP smoke test for an API server started without a YOLO model.

Uses only the Python standard library, so it can be run on the CI host.
It deliberately expects /ready to return 503 when MODEL_PATH is unset.
"""

import argparse
import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


def fetch_json(base_url: str, path: str) -> tuple[int, dict]:
    try:
        with urlopen(f"{base_url.rstrip('/')}{path}", timeout=3) as response:
            return response.status, json.load(response)
    except HTTPError as exc:
        try:
            return exc.code, json.load(exc)
        finally:
            exc.close()


def check(base_url: str, wait_seconds: float) -> None:
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            status, body = fetch_json(base_url, "/health")
            if status == 200 and body == {"status": "ok"}:
                break
            last_error = f"/health status={status}, response={body}"
        except (URLError, OSError, ValueError) as exc:
            last_error = str(exc)
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Health check did not pass: {last_error}")
        time.sleep(1)

    status, body = fetch_json(base_url, "/ready")
    if status != 503 or "not ready" not in str(body.get("detail", "")).lower():
        raise AssertionError(f"Expected unready (503) without model: {status}, {body}")

    status, body = fetch_json(base_url, "/openapi.json")
    if status != 200 or not all(
        path in body.get("paths", {})
        for path in ("/health", "/ready", "/predict", "/v1/detect")
    ):
        raise AssertionError("OpenAPI schema is missing required endpoints")

    print("PASS /health=200, /ready=503 (no model), /openapi.json=200")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--wait-seconds", type=float, default=30)
    args = parser.parse_args()
    check(args.base_url, args.wait_seconds)


if __name__ == "__main__":
    main()
