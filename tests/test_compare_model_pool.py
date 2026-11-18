"""CSV comparison helper tests."""
import csv
import pytest

from scripts.compare_model_pool import compare, read_aggregate


def example(concurrency, *, rps, p95, requests=240):
    return {
        "repeat": "all", "concurrency": str(concurrency),
        "requests": str(requests), "success_rps": str(rps),
        "client_p95_ms": str(p95), "queue_p95_ms": "12.0", "failures": "0",
    }


def test_compare_aggregate_rows():
    left = {1: example(1, rps=40, p95=30), 8: example(8, rps=44, p95=200)}
    right = {1: example(1, rps=41, p95=29), 8: example(8, rps=55, p95=140)}
    result = compare(left, right)
    assert [x["concurrency"] for x in result] == [1, 8]
    assert result[1]["rps_speedup"] == 1.25
    assert result[1]["p95_ratio"] == 0.7


def test_compare_rejects_different_experiment_sizes():
    with pytest.raises(ValueError, match="same measured request count"):
        compare({1: example(1, rps=20, p95=30)}, {1: example(1, rps=21, p95=20, requests=150)})


def test_read_aggregate_ignores_individual_repeats(tmp_path):
    path = tmp_path / "summary.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(example(2, rps=42, p95=45)))
        writer.writeheader()
        writer.writerow({**example(2, rps=40, p95=50), "repeat": 1})
        writer.writerow(example(2, rps=42, p95=45))
    assert read_aggregate(path)[2]["success_rps"] == "42"
