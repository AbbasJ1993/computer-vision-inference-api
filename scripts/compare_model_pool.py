"""Compare Step 2.2.7 single- and two-model summary CSV files.

Uses only `repeat=all` rows and reports success rates and latency per
concurrency. A model-worker pool should be judged on RPS *and* client P95,
not just inference time. The input runs must use equal test configurations.
"""

import argparse
import csv
from pathlib import Path


COLUMNS = [
    "concurrency", "single_success_rps", "dual_success_rps", "rps_speedup",
    "single_client_p95_ms", "dual_client_p95_ms", "p95_ratio",
    "single_queue_p95_ms", "dual_queue_p95_ms",
    "single_failures", "dual_failures",
]


def read_aggregate(path: Path) -> dict[int, dict]:
    rows = {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row.get("repeat") != "all":
                continue
            concurrency = int(row["concurrency"])
            if concurrency in rows:
                raise ValueError(f"Duplicate aggregate row for concurrency={concurrency}")
            rows[concurrency] = row
    if not rows:
        raise ValueError(f"No repeat=all rows in {path}")
    return rows


def _metric(row: dict, key: str) -> float:
    value = row.get(key)
    if value in (None, ""):
        raise ValueError(f"Missing metric: {key}")
    return float(value)


def compare(single: dict[int, dict], dual: dict[int, dict]) -> list[dict]:
    if single.keys() != dual.keys():
        raise ValueError("Both runs must have identical concurrency values")
    result = []
    for concurrency in sorted(single):
        left, right = single[concurrency], dual[concurrency]
        if int(left["requests"]) != int(right["requests"]):
            raise ValueError("Both runs must use the same measured request count")
        a_rps = _metric(left, "success_rps")
        b_rps = _metric(right, "success_rps")
        a_p95 = _metric(left, "client_p95_ms")
        b_p95 = _metric(right, "client_p95_ms")
        result.append({
            "concurrency": concurrency,
            "single_success_rps": a_rps,
            "dual_success_rps": b_rps,
            "rps_speedup": round(b_rps / a_rps, 3) if a_rps else None,
            "single_client_p95_ms": a_p95,
            "dual_client_p95_ms": b_p95,
            "p95_ratio": round(b_p95 / a_p95, 3) if a_p95 else None,
            "single_queue_p95_ms": _metric(left, "queue_p95_ms"),
            "dual_queue_p95_ms": _metric(right, "queue_p95_ms"),
            "single_failures": int(left["failures"]),
            "dual_failures": int(right["failures"]),
        })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single", type=Path, required=True)
    parser.add_argument("--dual", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("benchmark_results/model_pool_comparison.csv"))
    args = parser.parse_args()
    rows = compare(read_aggregate(args.single), read_aggregate(args.dual))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print("clients | single RPS | dual RPS | speedup | single P95 | dual P95 | failures single/dual")
    for row in rows:
        print(
            f"{row['concurrency']:>7} | {row['single_success_rps']:>10.2f} | "
            f"{row['dual_success_rps']:>8.2f} | {str(row['rps_speedup']):>7} | "
            f"{row['single_client_p95_ms']:>10.2f} | {row['dual_client_p95_ms']:>8.2f} | "
            f"{row['single_failures']}/{row['dual_failures']}"
        )
    print(f"CSV: {args.output.resolve()}")


if __name__ == "__main__":
    main()
