"""Compare two profiling CSV files using only the measured phases."""

import argparse
import csv
import statistics
from pathlib import Path

from scripts.benchmark_inference_api import percentile
from scripts.profile_inference_api import PROFILE_COLUMNS


def read_measured(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = [r for r in csv.DictReader(file) if r["phase"] == "measured"]
    if not rows:
        raise ValueError(f"No measured rows in {path}")
    return rows


def summarize(rows: list[dict], column: str) -> tuple[float, float, float] | None:
    values = [float(r[column]) for r in rows if r.get(column) not in (None, "", "None")]
    if not values:
        return None
    return statistics.mean(values), statistics.median(values), percentile(values, 95)


def compare(cpu: list[dict], gpu: list[dict]) -> list[dict]:
    comparisons = []
    for column in PROFILE_COLUMNS:
        cpu_stats = summarize(cpu, column)
        gpu_stats = summarize(gpu, column)
        comparisons.append({
            "metric": column,
            "cpu_mean_ms": round(cpu_stats[0], 3) if cpu_stats else None,
            "gpu_mean_ms": round(gpu_stats[0], 3) if gpu_stats else None,
            "speedup_cpu_over_gpu": round(cpu_stats[0] / gpu_stats[0], 3)
            if cpu_stats and gpu_stats and gpu_stats[0] > 0 else None,
            "cpu_p50_ms": round(cpu_stats[1], 3) if cpu_stats else None,
            "gpu_p50_ms": round(gpu_stats[1], 3) if gpu_stats else None,
            "cpu_p95_ms": round(cpu_stats[2], 3) if cpu_stats else None,
            "gpu_p95_ms": round(gpu_stats[2], 3) if gpu_stats else None,
        })
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=Path, required=True)
    parser.add_argument("--gpu", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("benchmark_results/profile_comparison.csv"))
    args = parser.parse_args()
    table = compare(read_measured(args.cpu), read_measured(args.gpu))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(f"{'Metric':25} {'CPU mean':>10} {'GPU mean':>10} {'Speedup':>9}")
    for row in table:
        def display(value):
            return "N/A" if value is None else f"{value:.3f}"
        print(f"{row['metric']:25} {display(row['cpu_mean_ms']):>10} "
              f"{display(row['gpu_mean_ms']):>10} {display(row['speedup_cpu_over_gpu']):>9}")
    print(f"CSV: {args.output.resolve()}")


if __name__ == "__main__":
    main()
