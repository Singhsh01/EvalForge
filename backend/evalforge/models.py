"""ORM models — EvalForge's own metadata store.

MLflow remains the system of record for run parameters, metrics and model
artifacts; these tables hold platform state that MLflow does not model well
(dataset versions and their quality reports, job lifecycle, evaluations against
other dataset versions, audit history) and keep a pointer (``mlflow_run_id``)
into MLflow for every training run.
"""

from __future__ import annotations

import enum
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from evalforge.db import Base


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(UTC)


class TaskType(enum.StrEnum):
    binary_classification = "binary_classification"
    multiclass_classification = "multiclass_classification"


class VersionStatus(enum.StrEnum):
    validated = "validated"  # all blocking checks passed
    rejected = "rejected"  # at least one blocking check failed; cannot be trained on


class JobStatus(enum.StrEnum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class JobKind(enum.StrEnum):
    train = "train"
    evaluate = "evaluate"


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    description: Mapped[str] = mapped_column(Text, default="")
    target_column: Mapped[str] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    versions: Mapped[list[DatasetVersion]] = relationship(
        back_populates="dataset", order_by="DatasetVersion.version", cascade="all, delete-orphan"
    )


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"
    __table_args__ = (
        UniqueConstraint("dataset_id", "version", name="uq_dataset_version"),
        UniqueConstraint("dataset_id", "sha256", name="uq_dataset_content"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"))
    version: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    source_filename: Mapped[str] = mapped_column(String(512))
    file_format: Mapped[str] = mapped_column(String(16))
    storage_path: Mapped[str] = mapped_column(String(1024))
    size_bytes: Mapped[int] = mapped_column(Integer)
    n_rows: Mapped[int] = mapped_column(Integer)
    n_columns: Mapped[int] = mapped_column(Integer)
    task_type: Mapped[TaskType | None] = mapped_column(Enum(TaskType), nullable=True)
    status: Mapped[VersionStatus] = mapped_column(Enum(VersionStatus))
    schema_: Mapped[list[dict[str, Any]]] = mapped_column("schema", JSON)
    profile: Mapped[dict[str, Any]] = mapped_column(JSON)
    quality_checks: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    split: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    dataset: Mapped[Dataset] = relationship(back_populates="versions")


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    kind: Mapped[JobKind] = mapped_column(Enum(JobKind))
    status: Mapped[JobStatus] = mapped_column(Enum(JobStatus), default=JobStatus.queued, index=True)
    stage: Mapped[str] = mapped_column(String(64), default="queued")
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    logs: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"))
    dataset_version_id: Mapped[str] = mapped_column(ForeignKey("dataset_versions.id"), index=True)
    algorithm: Mapped[str] = mapped_column(String(64))
    hyperparameters: Mapped[dict[str, Any]] = mapped_column(JSON)
    seed: Mapped[int] = mapped_column(Integer)
    cv_folds: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[JobStatus] = mapped_column(Enum(JobStatus), default=JobStatus.queued)
    feature_columns: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    dropped_columns: Mapped[list[dict[str, str]] | None] = mapped_column(JSON, nullable=True)
    metrics: Mapped[dict[str, float] | None] = mapped_column(JSON, nullable=True)
    report: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    environment: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    mlflow_experiment_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    mlflow_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    model_uri: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    training_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    git_commit: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    dataset_version: Mapped[DatasetVersion] = relationship()
    job: Mapped[Job] = relationship()


class Evaluation(Base):
    """Evaluation of a trained run's model against a (possibly different) dataset version."""

    __tablename__ = "evaluations"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("training_runs.id", ondelete="CASCADE"), index=True
    )
    dataset_version_id: Mapped[str] = mapped_column(ForeignKey("dataset_versions.id"))
    job_id: Mapped[str | None] = mapped_column(ForeignKey("jobs.id"), nullable=True)
    split: Mapped[str] = mapped_column(String(16))  # "holdout" | "full"
    status: Mapped[JobStatus] = mapped_column(Enum(JobStatus), default=JobStatus.queued)
    metrics: Mapped[dict[str, float] | None] = mapped_column(JSON, nullable=True)
    report: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    actor: Mapped[str] = mapped_column(String(128), default="system")
    action: Mapped[str] = mapped_column(String(64), index=True)
    entity_type: Mapped[str] = mapped_column(String(64))
    entity_id: Mapped[str] = mapped_column(String(64), index=True)
    data: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
