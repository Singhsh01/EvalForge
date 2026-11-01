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
    pipeline = "pipeline"
    eval_suite = "eval_suite"


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


# ---------------------------------------------------------------------------
# Enterprise operations (milestones 1-7). Status fields are plain strings with
# validation in the service layer, so adding a state never needs an enum migration.
# ---------------------------------------------------------------------------


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(256))
    role: Mapped[str] = mapped_column(String(16))  # viewer | engineer | approver | admin
    active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class ApiToken(Base):
    __tablename__ = "api_tokens"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(64), default="session")
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    revoked: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    user: Mapped[User] = relationship()


class ModelVersion(Base):
    """Mirror of an MLflow registered model version plus platform governance state."""

    __tablename__ = "model_versions"
    __table_args__ = (UniqueConstraint("name", "version", name="uq_model_version"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(128), index=True)
    version: Mapped[int] = mapped_column(Integer)
    run_id: Mapped[str] = mapped_column(ForeignKey("training_runs.id"), index=True)
    source_uri: Mapped[str] = mapped_column(String(1024))
    status: Mapped[str] = mapped_column(String(16), default="registered")
    # registered | approved | rejected | archived
    is_champion: Mapped[bool] = mapped_column(default=False)
    gate: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_by: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    run: Mapped[TrainingRun] = relationship()


class Approval(Base):
    __tablename__ = "approvals"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    kind: Mapped[str] = mapped_column(String(32), index=True)
    # model_promotion | deployment | rollback | retraining | evaluation
    target_type: Mapped[str] = mapped_column(String(32))
    target_id: Mapped[str] = mapped_column(String(64))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    summary: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    # pending | approved | rejected | executed | failed
    requested_by: Mapped[str] = mapped_column(String(64))
    decided_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    decision_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Deployment(Base):
    __tablename__ = "deployments"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    model_name: Mapped[str] = mapped_column(String(128), index=True)
    model_version_id: Mapped[str] = mapped_column(ForeignKey("model_versions.id"))
    environment: Mapped[str] = mapped_column(String(16), index=True)  # staging | production
    traffic_percent: Mapped[int] = mapped_column(Integer, default=100)
    role: Mapped[str] = mapped_column(String(16), default="stable")  # stable | canary
    status: Mapped[str] = mapped_column(String(16), default="active", index=True)
    # active | superseded | rolled_back | failed
    health: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    approval_id: Mapped[str | None] = mapped_column(ForeignKey("approvals.id"), nullable=True)
    previous_deployment_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    deployed_by: Mapped[str] = mapped_column(String(64))
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    model_version: Mapped[ModelVersion] = relationship()


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    request_id: Mapped[str] = mapped_column(String(64), index=True)
    deployment_id: Mapped[str] = mapped_column(ForeignKey("deployments.id"), index=True)
    model_version_id: Mapped[str] = mapped_column(ForeignKey("model_versions.id"), index=True)
    environment: Mapped[str] = mapped_column(String(16))
    inputs: Mapped[dict[str, Any]] = mapped_column(JSON)
    prediction: Mapped[str] = mapped_column(String(128))
    probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    latency_ms: Mapped[float] = mapped_column(Float)
    label: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    pipeline: Mapped[str] = mapped_column(String(64), index=True)
    config: Mapped[dict[str, Any]] = mapped_column(JSON)
    config_hash: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(24), default="queued", index=True)
    trigger: Mapped[str] = mapped_column(String(32), default="manual")  # manual|schedule|incident
    triggered_by: Mapped[str] = mapped_column(String(64))
    job_id: Mapped[str | None] = mapped_column(ForeignKey("jobs.id"), nullable=True)
    lineage: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    steps: Mapped[list[PipelineStep]] = relationship(
        back_populates="run", order_by="PipelineStep.position", cascade="all, delete-orphan"
    )


class PipelineStep(Base):
    __tablename__ = "pipeline_steps"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(ForeignKey("pipeline_runs.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(64))
    position: Mapped[int] = mapped_column(Integer)
    depends_on: Mapped[list[str]] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(16), default="pending")
    # pending | running | succeeded | failed | skipped | waiting_approval
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    output: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    run: Mapped[PipelineRun] = relationship(back_populates="steps")


class PipelineSchedule(Base):
    __tablename__ = "pipeline_schedules"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    pipeline: Mapped[str] = mapped_column(String(64))
    config: Mapped[dict[str, Any]] = mapped_column(JSON)
    interval_minutes: Mapped[int] = mapped_column(Integer)
    enabled: Mapped[bool] = mapped_column(default=True)
    last_run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_by: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Prompt(Base):
    __tablename__ = "prompts"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    versions: Mapped[list[PromptVersion]] = relationship(
        back_populates="prompt", order_by="PromptVersion.version", cascade="all, delete-orphan"
    )


class PromptVersion(Base):
    __tablename__ = "prompt_versions"
    __table_args__ = (UniqueConstraint("prompt_id", "version", name="uq_prompt_version"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    prompt_id: Mapped[str] = mapped_column(ForeignKey("prompts.id", ondelete="CASCADE"))
    version: Mapped[int] = mapped_column(Integer)
    system: Mapped[str] = mapped_column(Text, default="")
    template: Mapped[str] = mapped_column(Text)
    variables: Mapped[list[str]] = mapped_column(JSON, default=list)
    output_schema: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    sha256: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), default="draft")  # draft | released
    note: Mapped[str] = mapped_column(Text, default="")
    created_by: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    prompt: Mapped[Prompt] = relationship(back_populates="versions")


class EvalSuite(Base):
    """A versioned, immutable set of evaluation cases plus evaluator configuration."""

    __tablename__ = "eval_suites"
    __table_args__ = (UniqueConstraint("name", "version", name="uq_eval_suite_version"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(128), index=True)
    version: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(16))  # prompt | rag | agent
    description: Mapped[str] = mapped_column(Text, default="")
    cases: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    corpus: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    evaluators: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    gates: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    sha256: Mapped[str] = mapped_column(String(64))
    created_by: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class EvalRun(Base):
    __tablename__ = "eval_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    suite_id: Mapped[str] = mapped_column(ForeignKey("eval_suites.id"), index=True)
    subject: Mapped[dict[str, Any]] = mapped_column(JSON)
    subject_key: Mapped[str] = mapped_column(String(256), index=True)
    status: Mapped[str] = mapped_column(String(16), default="queued")
    job_id: Mapped[str | None] = mapped_column(ForeignKey("jobs.id"), nullable=True)
    summary: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    baseline_run_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    regression: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    trace_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_by: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    suite: Mapped[EvalSuite] = relationship()
    results: Mapped[list[EvalCaseResult]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class EvalCaseResult(Base):
    __tablename__ = "eval_case_results"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(ForeignKey("eval_runs.id", ondelete="CASCADE"), index=True)
    case_id: Mapped[str] = mapped_column(String(64))
    output: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    scores: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    passed: Mapped[bool] = mapped_column(default=False)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    trace_id: Mapped[str | None] = mapped_column(String(32), nullable=True)

    run: Mapped[EvalRun] = relationship(back_populates="results")


class LLMCall(Base):
    __tablename__ = "llm_calls"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    provider: Mapped[str] = mapped_column(String(32))
    model: Mapped[str] = mapped_column(String(128), index=True)
    purpose: Mapped[str] = mapped_column(String(32))  # eval | judge | copilot
    prompt_version_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    trace_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)


class Span(Base):
    """OpenTelemetry spans persisted by EvalForge's DB exporter (trace explorer source)."""

    __tablename__ = "spans"

    span_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    trace_id: Mapped[str] = mapped_column(String(32), index=True)
    parent_id: Mapped[str | None] = mapped_column(String(16), nullable=True)
    name: Mapped[str] = mapped_column(String(256), index=True)
    kind: Mapped[str] = mapped_column(String(16))
    service: Mapped[str] = mapped_column(String(64))
    start_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    duration_ms: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(16))  # ok | error | unset
    status_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    attributes: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    events: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)


class DriftReport(Base):
    __tablename__ = "drift_reports"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    model_name: Mapped[str] = mapped_column(String(128), index=True)
    model_version_id: Mapped[str] = mapped_column(ForeignKey("model_versions.id"))
    source: Mapped[str] = mapped_column(String(32))  # predictions | dataset_version
    reference: Mapped[dict[str, Any]] = mapped_column(JSON)
    current: Mapped[dict[str, Any]] = mapped_column(JSON)
    features: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    prediction_drift: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    performance: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    summary: Mapped[dict[str, Any]] = mapped_column(JSON)
    thresholds: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_by: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)


class Incident(Base):
    __tablename__ = "incidents"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    kind: Mapped[str] = mapped_column(String(32))  # data_drift | performance | agent_regression
    severity: Mapped[str] = mapped_column(String(16))  # low | medium | high
    status: Mapped[str] = mapped_column(String(16), default="open", index=True)
    # open | investigating | mitigated | resolved
    title: Mapped[str] = mapped_column(String(256))
    source_type: Mapped[str] = mapped_column(String(32))
    source_id: Mapped[str] = mapped_column(String(64))
    details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    findings: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    events: Mapped[list[IncidentEvent]] = relationship(
        back_populates="incident", order_by="IncidentEvent.created_at", cascade="all, delete-orphan"
    )


class IncidentEvent(Base):
    __tablename__ = "incident_events"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    incident_id: Mapped[str] = mapped_column(ForeignKey("incidents.id", ondelete="CASCADE"))
    kind: Mapped[str] = mapped_column(String(32))
    message: Mapped[str] = mapped_column(Text)
    data: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    actor: Mapped[str] = mapped_column(String(64), default="system")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    incident: Mapped[Incident] = relationship(back_populates="events")


class AgentConfig(Base):
    """Versioned configuration of the Copilot agent (tools enabled, planner rules, budgets)."""

    __tablename__ = "agent_configs"
    __table_args__ = (UniqueConstraint("name", "version", name="uq_agent_config"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(64))
    version: Mapped[int] = mapped_column(Integer)
    config: Mapped[dict[str, Any]] = mapped_column(JSON)
    sha256: Mapped[str] = mapped_column(String(64))
    note: Mapped[str] = mapped_column(Text, default="")
    created_by: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class CopilotSession(Base):
    __tablename__ = "copilot_sessions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    question: Mapped[str] = mapped_column(Text)
    agent_config_id: Mapped[str] = mapped_column(ForeignKey("agent_configs.id"))
    planner: Mapped[str] = mapped_column(String(32))  # rules | llm:<model>
    status: Mapped[str] = mapped_column(String(16), default="running")
    # running | completed | failed | cancelled | budget_exceeded
    report: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    steps: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    tool_calls: Mapped[int] = mapped_column(Integer, default=0)
    duration_ms: Mapped[float] = mapped_column(Float, default=0.0)
    trace_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    incident_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_by: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
