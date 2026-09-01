"""Central runtime configuration.  Secrets are read only from the environment."""
from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    app_env: str = os.getenv("WEATHERGPT_ENV", "development")
    database_path: str = os.getenv("WEATHERGPT_DB_PATH", "weathergpt.db")
    request_max_bytes: int = int(os.getenv("WEATHERGPT_REQUEST_MAX_BYTES", "65536"))
    source_timeout_seconds: float = float(os.getenv("WEATHERGPT_SOURCE_TIMEOUT_SECONDS", "20"))
    source_retries: int = int(os.getenv("WEATHERGPT_SOURCE_RETRIES", "2"))
    forecast_cache_ttl_seconds: int = int(os.getenv("WEATHERGPT_FORECAST_CACHE_TTL_SECONDS", "900"))
    location_cache_ttl_seconds: int = int(os.getenv("WEATHERGPT_LOCATION_CACHE_TTL_SECONDS", "2592000"))
    enable_llm: bool = os.getenv("WEATHERGPT_ENABLE_LLM", "true").lower() == "true"
    cors_origins: tuple[str, ...] = tuple(
        item.strip() for item in os.getenv("WEATHERGPT_CORS_ORIGINS", "").split(",") if item.strip()
    )


settings = Settings()
