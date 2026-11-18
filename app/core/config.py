"""Validated runtime configuration. Read once per ASGI lifespan.

Environment variables override .env, which overrides defaults.
The .env file is optional and never belongs in Git.
"""

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    model_path: str = Field(default="", validation_alias="MODEL_PATH")
    model_device: str = Field(default="cpu", min_length=1, validation_alias="MODEL_DEVICE")
    # 0 intentionally preserves the existing single-model semaphore mode.
    model_pool_workers: int = Field(default=0, validation_alias="MODEL_POOL_WORKERS")
    max_queue_size: int = Field(default=16, ge=0, validation_alias="MAX_QUEUE_SIZE")
    queue_timeout_seconds: float = Field(
        default=2.0,
        gt=0,
        allow_inf_nan=False,
        validation_alias="QUEUE_TIMEOUT_SECONDS",
    )
    detection_rate_limit: str = Field(
        default="10/minute", min_length=1, validation_alias="DETECTION_RATE_LIMIT"
    )
    enable_inference_profiling: bool = Field(
        default=False, validation_alias="ENABLE_INFERENCE_PROFILING"
    )


    @field_validator("model_pool_workers")
    @classmethod
    def validate_pool_worker_count(cls, value: int) -> int:
        if value not in (0, 1, 2):
            raise ValueError("MODEL_POOL_WORKERS must be 0, 1 or 2.")
        return value


def load_settings() -> Settings:
    """No cache: tests and each application startup read their own environment."""
    return Settings()
