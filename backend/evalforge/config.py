"""Runtime configuration.

All settings come from environment variables prefixed with ``EVALFORGE_`` (see
``.env.example`` at the repository root).  Defaults implement the *lite* profile:
SQLite metadata store, embedded MLflow (SQLite backend + local artifacts) and an
in-process thread executor, so the platform runs on an 8 GB laptop without Docker.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EVALFORGE_", env_file=".env", extra="ignore")

    env: Literal["dev", "test", "prod"] = "dev"
    data_dir: Path = Path("./var")

    # Metadata store. SQLite for lite profile, PostgreSQL in compose/prod.
    database_url: str | None = None

    # MLflow. If unset, an embedded SQLite tracking store under data_dir is used.
    mlflow_tracking_uri: str | None = None
    mlflow_experiment_prefix: str = "evalforge"

    # Job execution: "thread" (in-process, lite), "rq" (Redis worker), "sync" (tests).
    executor: Literal["thread", "rq", "sync"] = "thread"
    redis_url: str = "redis://localhost:6379/0"
    max_concurrent_jobs: int = Field(default=1, ge=1, le=8)
    job_timeout_seconds: int = 1800

    # Upload guard rails.
    max_upload_mb: int = Field(default=50, ge=1)
    max_rows: int = Field(default=2_000_000, ge=1)
    max_columns: int = Field(default=500, ge=1)

    # --- Security --------------------------------------------------------------------
    bootstrap_password: str | None = None  # initial admin/engineer/approver/viewer password
    token_ttl_hours: int = Field(default=12, ge=1, le=24 * 30)
    allow_self_approval: bool = False  # separation of duties; enable only for solo demos
    rate_limit_login_per_min: int = 10
    rate_limit_write_per_min: int = 240
    rate_limit_serve_per_min: int = 1200
    # Use the left-most X-Forwarded-For address as the client IP. Enable only when the API is
    # reachable exclusively through a trusted proxy (e.g. the Next.js server), never when the
    # API port itself is exposed, or clients could choose their own rate-limit key.
    trust_forwarded_for: bool = False

    # --- Serving / governance -------------------------------------------------------
    gate_min_f1: float = 0.80
    gate_max_f1_regression: float = 0.01  # candidate may trail champion by at most this
    gate_max_p95_latency_ms: float = 250.0
    model_cache_size: int = Field(default=4, ge=1, le=32)

    # --- LLM providers (optional; nothing is called unless configured) --------------
    anthropic_api_key: str | None = None
    anthropic_base_url: str = "https://api.anthropic.com"
    openai_api_key: str | None = None
    openai_base_url: str | None = None  # any OpenAI-compatible endpoint (vLLM, Ollama, ...)
    llm_timeout_seconds: float = 60.0
    copilot_llm_model: str | None = None  # e.g. "anthropic:claude-sonnet-4-5"; None = rules planner

    scheduler_enabled: bool = True
    scheduler_interval_seconds: float = 30.0

    # --- Observability --------------------------------------------------------------
    tracing_enabled: bool = True
    otlp_endpoint: str | None = None  # e.g. http://otel-collector:4318
    service_name: str = "evalforge-api"

    # --- Copilot budgets --------------------------------------------------------------
    copilot_max_steps: int = Field(default=12, ge=1, le=50)
    copilot_timeout_seconds: float = Field(default=60.0, ge=1)

    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    git_commit: str | None = None  # injected by CI / Docker build when .git is absent

    @computed_field  # type: ignore[prop-decorator]
    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"sqlite:///{(self.data_dir / 'evalforge.db').resolve()}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def resolved_mlflow_uri(self) -> str:
        if self.mlflow_tracking_uri:
            return self.mlflow_tracking_uri
        return f"sqlite:///{(self.data_dir / 'mlflow.db').resolve()}"

    @property
    def datasets_dir(self) -> Path:
        return self.data_dir / "datasets"

    @property
    def mlflow_artifacts_dir(self) -> Path:
        return self.data_dir / "mlartifacts"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.datasets_dir, self.mlflow_artifacts_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
