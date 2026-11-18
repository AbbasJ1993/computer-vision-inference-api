"""Refactoring tests: URL compatibility and typed configuration validation."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.core.config import Settings, load_settings
from app.main import app


def test_default_settings_without_env(monkeypatch):
    for key in (
        "MODEL_PATH", "MODEL_DEVICE", "MODEL_POOL_WORKERS", "MAX_QUEUE_SIZE",
        "QUEUE_TIMEOUT_SECONDS", "ENABLE_INFERENCE_PROFILING", "DETECTION_RATE_LIMIT",
    ):
        monkeypatch.delenv(key, raising=False)
    settings = Settings(_env_file=None)
    assert settings.model_pool_workers == 0
    assert settings.max_queue_size == 16
    assert settings.queue_timeout_seconds == 2.0
    assert not settings.enable_inference_profiling


@pytest.mark.parametrize("name,value", [
    ("MODEL_POOL_WORKERS", "3"),
    ("MAX_QUEUE_SIZE", "-2"),
    ("MAX_QUEUE_SIZE", "hello"),
    ("QUEUE_TIMEOUT_SECONDS", "nan"),
    ("QUEUE_TIMEOUT_SECONDS", "0"),
    ("ENABLE_INFERENCE_PROFILING", "maybe"),
])
def test_invalid_settings_rejected(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_environment_values_read_at_startup(monkeypatch):
    monkeypatch.delenv("MODEL_PATH", raising=False)
    monkeypatch.setenv("MODEL_POOL_WORKERS", "2")
    monkeypatch.setenv("MAX_QUEUE_SIZE", "4")
    monkeypatch.setenv("QUEUE_TIMEOUT_SECONDS", "0.5")
    with TestClient(app) as client:
        assert app.state.settings.model_pool_workers == 2
        assert app.state.settings.max_queue_size == 4
        assert app.state.settings.queue_timeout_seconds == 0.5
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503


def test_benchmark_routes_removed_but_inference_routes_remain(monkeypatch):
    monkeypatch.delenv("MODEL_PATH", raising=False)
    with TestClient(app) as client:
        for path in ("/benchmark/direct", "/benchmark/thread", "/benchmark/process"):
            assert client.get(path).status_code == 404
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert client.post("/v1/detect").status_code in (422, 503)
        assert client.post("/predict").status_code == 422
        paths = client.get("/openapi.json").json()["paths"]
        assert all(path in paths for path in ("/health", "/ready", "/predict", "/v1/detect"))
        assert all(not path.startswith("/benchmark/") for path in paths)
