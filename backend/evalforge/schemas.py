"""Pydantic API contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, ConfigDict, Field


def _as_utc(v: datetime | None) -> datetime | None:
    """SQLite drops tzinfo on read; all stored timestamps are UTC, so say so explicitly.

    Without this the API emits naive ISO strings and browsers parse them as local time.
    """
    if v is not None and v.tzinfo is None:
        return v.replace(tzinfo=UTC)
    return v


UTCDateTime = Annotated[datetime, AfterValidator(_as_utc)]


class ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())


class DatasetVersionSummary(ORM):
    id: str
    version: int
    sha256: str
    source_filename: str
    file_format: str
    size_bytes: int
    n_rows: int
    n_columns: int
    task_type: str | None
    status: str
    note: str
    created_at: UTCDateTime
    checks_failed: int = 0
    checks_warned: int = 0


class DatasetVersionDetail(DatasetVersionSummary):
    dataset_id: str
    schema_: list[dict[str, Any]] = Field(serialization_alias="schema")
    profile: dict[str, Any]
    quality_checks: list[dict[str, Any]]
    split: dict[str, Any] | None


class DatasetOut(ORM):
    id: str
    name: str
    description: str
    target_column: str
    created_at: UTCDateTime
    versions: list[DatasetVersionSummary]


class TrainRequest(BaseModel):
    dataset_version_id: str
    algorithm: str
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    cv_folds: int = Field(default=0, ge=0, le=10, description="0 disables cross-validation")


class EvaluateRequest(BaseModel):
    dataset_version_id: str


class JobOut(ORM):
    id: str
    kind: str
    status: str
    stage: str
    progress: float
    error: str | None
    logs: list[dict[str, Any]]
    result: dict[str, Any] | None
    created_at: UTCDateTime
    started_at: UTCDateTime | None
    finished_at: UTCDateTime | None


class RunOut(ORM):
    id: str
    job_id: str
    dataset_version_id: str
    dataset_name: str | None = None
    dataset_version_number: int | None = None
    algorithm: str
    hyperparameters: dict[str, Any]
    seed: int
    cv_folds: int
    status: str
    metrics: dict[str, float] | None
    mlflow_experiment_id: str | None
    mlflow_run_id: str | None
    model_uri: str | None
    training_seconds: float | None
    git_commit: str | None
    error: str | None
    created_at: UTCDateTime
    finished_at: UTCDateTime | None


class RunDetail(RunOut):
    report: dict[str, Any] | None
    environment: dict[str, Any] | None
    feature_columns: list[str] | None
    dropped_columns: list[dict[str, str]] | None
    job: JobOut | None = None


class EvaluationOut(ORM):
    id: str
    run_id: str
    dataset_version_id: str
    job_id: str | None
    split: str
    status: str
    metrics: dict[str, float] | None
    report: dict[str, Any] | None
    error: str | None
    created_at: UTCDateTime


class LaunchResponse(BaseModel):
    job: JobOut
    run: RunOut | None = None
    evaluation: EvaluationOut | None = None


class CompareOut(BaseModel):
    runs: list[RunOut]
    metric_keys: list[str]
    param_keys: list[str]
    differing_params: list[str]
    same_holdout: bool
    best: dict[str, list[str]]
    warnings: list[str]
