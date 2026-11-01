"""Multi-stage MLOps pipelines with persisted DAG state, retries, resume and lineage.

The continuous-training pipeline is::

    validate_data → train → evaluate → register → quality_gate → deploy_staging
                  → request_promotion (human approval)

Each step is a plain function over a shared context. The runner persists every
state transition, retries failed steps up to ``max_retries``, and can *resume* a
failed run from the first unsuccessful step, reusing earlier outputs. Steps never
promote or deploy to production themselves: the last step files an approval.

Why not Prefect here: see docs/adr/0006-pipeline-orchestration.md. The step
functions have no runner coupling, so wrapping them in Prefect ``@task``s later is
mechanical.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import traceback
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from evalforge.db import session_scope
from evalforge.models import (
    AuditEvent,
    DatasetVersion,
    Evaluation,
    Job,
    JobKind,
    JobStatus,
    ModelVersion,
    PipelineRun,
    PipelineSchedule,
    PipelineStep,
    TrainingRun,
    VersionStatus,
)
from evalforge.services import jobs, registry, serving, tracking
from evalforge.services.auth import Principal
from evalforge.services.experiments import create_training_run
from evalforge.services.governance import request_approval

log = logging.getLogger(__name__)


class TrainPipelineConfig(BaseModel):
    dataset_version_id: str
    algorithm: str = "logistic_regression"
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    seed: int = 42
    cv_folds: int = Field(default=5, ge=0, le=10)
    eval_dataset_version_id: str | None = None
    deploy_to_staging: bool = True
    request_promotion: bool = True
    max_retries: int = Field(default=1, ge=0, le=5)


Ctx = dict[str, Any]
StepFn = Callable[[Session, Ctx, Principal], dict[str, Any]]


class StepFailed(Exception):
    """Deterministic failure (e.g. quality gate) that retrying cannot fix."""


def _validate(db: Session, ctx: Ctx, actor: Principal) -> dict[str, Any]:
    v = db.get(DatasetVersion, ctx["config"]["dataset_version_id"])
    if v is None:
        raise StepFailed("Dataset version not found")
    blocking = [c["id"] for c in v.quality_checks if c["severity"] == "error" and not c["passed"]]
    if v.status != VersionStatus.validated:
        raise StepFailed(f"Dataset version failed blocking checks: {blocking}")
    warnings = [c["id"] for c in v.quality_checks if c["severity"] == "warning" and not c["passed"]]
    return {
        "dataset": v.dataset.name,
        "dataset_version": v.version,
        "dataset_sha256": v.sha256,
        "rows": v.n_rows,
        "warnings": warnings,
    }


def _train(db: Session, ctx: Ctx, actor: Principal) -> dict[str, Any]:
    c = ctx["config"]
    job, run = create_training_run(
        db,
        dataset_version_id=c["dataset_version_id"],
        algorithm=c["algorithm"],
        hyperparameters=c["hyperparameters"],
        seed=c["seed"],
        cv_folds=c["cv_folds"],
        actor=actor.username,
    )
    db.commit()
    jobs.execute(job.id)  # we are already inside a worker: run inline
    db.expire_all()
    run = db.get(TrainingRun, run.id)
    assert run is not None
    if run.status != JobStatus.succeeded:
        raise RuntimeError(f"Training failed: {run.error}")
    return {
        "run_id": run.id,
        "mlflow_run_id": run.mlflow_run_id,
        "model_uri": run.model_uri,
        "f1": (run.metrics or {}).get("f1"),
        "training_seconds": run.training_seconds,
    }


def _evaluate(db: Session, ctx: Ctx, actor: Principal) -> dict[str, Any]:
    run_id = ctx["outputs"]["train"]["run_id"]
    holdout = db.scalar(
        select(Evaluation).where(Evaluation.run_id == run_id, Evaluation.split == "holdout")
    )
    out: dict[str, Any] = {"holdout": holdout.metrics if holdout else None}
    extra = ctx["config"].get("eval_dataset_version_id")
    if extra:
        job = Job(kind=JobKind.evaluate, payload={"run_id": run_id, "dataset_version_id": extra})
        db.add(job)
        db.flush()
        ev = Evaluation(run_id=run_id, dataset_version_id=extra, job_id=job.id, split="full")
        db.add(ev)
        db.commit()
        jobs.execute(job.id)
        db.expire_all()
        ev = db.get(Evaluation, ev.id)
        assert ev is not None
        if ev.status != JobStatus.succeeded:
            raise RuntimeError(f"Evaluation failed: {ev.error}")
        out["extra"] = {"dataset_version_id": extra, "metrics": ev.metrics}
    return out


def _register(db: Session, ctx: Ctx, actor: Principal) -> dict[str, Any]:
    run = db.get(TrainingRun, ctx["outputs"]["train"]["run_id"])
    assert run is not None
    existing = db.scalar(select(ModelVersion).where(ModelVersion.run_id == run.id))
    mv = existing or registry.register(db, run, actor)
    return {"model_version_id": mv.id, "name": mv.name, "version": mv.version}


def _gate(db: Session, ctx: Ctx, actor: Principal) -> dict[str, Any]:
    mv = db.get(ModelVersion, ctx["outputs"]["register"]["model_version_id"])
    assert mv is not None
    mv.gate = registry.evaluate_gate(db, mv)
    db.commit()
    if not mv.gate["passed"]:
        failed = [c["id"] for c in mv.gate["checks"] if c["blocking"] and not c["passed"]]
        raise StepFailed(f"Quality gate failed: {failed}")
    return mv.gate


def _deploy_staging(db: Session, ctx: Ctx, actor: Principal) -> dict[str, Any]:
    if not ctx["config"].get("deploy_to_staging", True):
        return {"skipped": True}
    mv = db.get(ModelVersion, ctx["outputs"]["register"]["model_version_id"])
    assert mv is not None
    stable, _ = serving.stable_and_canary(db, mv.name, "staging")
    if stable is not None and stable.model_version_id == mv.id:
        return {"deployment_id": stable.id, "already_active": True}
    dep = serving.deploy(db, mv, "staging", 100, actor, note="continuous-training pipeline")
    return {"deployment_id": dep.id, "health": dep.health}


def _request_promotion(db: Session, ctx: Ctx, actor: Principal) -> dict[str, Any]:
    if not ctx["config"].get("request_promotion", True):
        return {"skipped": True}
    mv = db.get(ModelVersion, ctx["outputs"]["register"]["model_version_id"])
    assert mv is not None
    a = request_approval(
        db,
        kind="model_promotion",
        target_type="model_version",
        target_id=mv.id,
        payload={"name": mv.name, "version": mv.version, "pipeline_run_id": ctx["run_id"]},
        summary=f"Promote {mv.name} v{mv.version} (pipeline {ctx['run_id'][:8]}) to champion",
        actor=actor,
    )
    return {"approval_id": a.id, "status": "waiting_approval"}


PIPELINES: dict[str, list[tuple[str, StepFn, list[str]]]] = {
    "continuous_training": [
        ("validate_data", _validate, []),
        ("train", _train, ["validate_data"]),
        ("evaluate", _evaluate, ["train"]),
        ("register", _register, ["evaluate"]),
        ("quality_gate", _gate, ["register"]),
        ("deploy_staging", _deploy_staging, ["quality_gate"]),
        ("request_promotion", _request_promotion, ["deploy_staging"]),
    ]
}


def verify_definition(name: str) -> None:
    """Static checks: known pipeline, unique steps, dependencies exist and precede use."""
    steps = PIPELINES.get(name)
    if not steps:
        raise ValueError(f"Unknown pipeline {name}")
    seen: set[str] = set()
    for step, _, deps in steps:
        if step in seen:
            raise ValueError(f"Duplicate step {step}")
        missing = [d for d in deps if d not in seen]
        if missing:
            raise ValueError(f"Step {step} depends on {missing} which do not precede it")
        seen.add(step)


def config_hash(cfg: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def create_run(
    db: Session, name: str, cfg: dict[str, Any], actor: Principal, trigger: str = "manual"
) -> PipelineRun:
    verify_definition(name)
    parsed = TrainPipelineConfig(**cfg).model_dump()
    run = PipelineRun(
        pipeline=name,
        config=parsed,
        config_hash=config_hash(parsed),
        triggered_by=actor.username,
        trigger=trigger,
        lineage={"git_commit": tracking.git_info()["commit"]},
    )
    for i, (step, _, deps) in enumerate(PIPELINES[name]):
        run.steps.append(PipelineStep(name=step, position=i, depends_on=deps))
    db.add(run)
    db.flush()
    job = Job(kind=JobKind.pipeline, payload={"pipeline_run_id": run.id})
    db.add(job)
    db.flush()
    run.job_id = job.id
    db.add(
        AuditEvent(
            actor=actor.username,
            action="pipeline.triggered",
            entity_type="pipeline_run",
            entity_id=run.id,
            data={"pipeline": name, "trigger": trigger, "config_hash": run.config_hash},
        )
    )
    return run


def execute_run(job_id: str) -> dict[str, Any]:
    """Job handler: run (or resume) the pipeline attached to this job."""
    with session_scope() as db:
        run = db.scalars(select(PipelineRun).where(PipelineRun.job_id == job_id)).one()
        run_id, name = run.id, run.pipeline
        actor = Principal("pipeline", run.triggered_by, "engineer")
        run.status = "running"
    fns = {s: f for s, f, _ in PIPELINES[name]}
    outputs: dict[str, Any] = {}
    with session_scope() as db:
        for st in db.get(PipelineRun, run_id).steps:  # type: ignore[union-attr]
            if st.status == "succeeded":
                outputs[st.name] = st.output or {}
        cfg = db.get(PipelineRun, run_id).config  # type: ignore[union-attr]
    ctx: Ctx = {"run_id": run_id, "config": cfg, "outputs": outputs}

    final = "succeeded"
    for step_name in fns:
        if step_name in outputs:
            continue  # resume: keep previous successful output
        max_retries = int(cfg.get("max_retries", 1))
        attempt = 0
        while True:
            attempt += 1
            with session_scope() as db:
                st = _step(db, run_id, step_name)
                st.status, st.attempts = "running", st.attempts + 1
                st.started_at, st.error = datetime.now(UTC), None
            try:
                with session_scope() as db:
                    out = fns[step_name](db, ctx, actor)
                    st = _step(db, run_id, step_name)
                    waiting = out.get("status") == "waiting_approval"
                    st.status = "waiting_approval" if waiting else "succeeded"
                    st.output, st.finished_at = out, datetime.now(UTC)
                    ctx["outputs"][step_name] = out
                    _update_lineage(db.get(PipelineRun, run_id), step_name, out)
                break
            except Exception as exc:
                retryable = not isinstance(exc, StepFailed) and attempt <= max_retries
                with session_scope() as db:
                    st = _step(db, run_id, step_name)
                    st.error = f"{type(exc).__name__}: {exc}"
                    if not retryable:
                        st.status, st.finished_at = "failed", datetime.now(UTC)
                        st.output = {"traceback": traceback.format_exc()[-3000:]}
                if retryable:
                    time.sleep(min(2**attempt * 0.1, 2))
                    continue
                final = "failed"
                break
        if final == "failed":
            break

    with session_scope() as db:
        run = db.get(PipelineRun, run_id)
        assert run is not None
        for st in run.steps:
            if st.status == "pending" and final == "failed":
                st.status = "skipped"
        waiting = any(s.status == "waiting_approval" for s in run.steps)
        run.status = "awaiting_approval" if final == "succeeded" and waiting else final
        run.error = next((s.error for s in run.steps if s.status == "failed"), None)
        run.finished_at = datetime.now(UTC)
        db.add(
            AuditEvent(
                actor=run.triggered_by,
                action=f"pipeline.{run.status}",
                entity_type="pipeline_run",
                entity_id=run.id,
                data=run.lineage,
            )
        )
        return {"pipeline_run_id": run_id, "status": run.status}


def _step(db: Session, run_id: str, name: str) -> PipelineStep:
    return db.scalars(
        select(PipelineStep).where(PipelineStep.run_id == run_id, PipelineStep.name == name)
    ).one()


def _update_lineage(run: PipelineRun | None, step: str, out: dict[str, Any]) -> None:
    if run is None:
        return
    keep = {
        "validate_data": ("dataset", "dataset_version", "dataset_sha256"),
        "train": ("run_id", "mlflow_run_id", "model_uri"),
        "register": ("model_version_id", "name", "version"),
        "deploy_staging": ("deployment_id",),
        "request_promotion": ("approval_id",),
    }
    lineage = dict(run.lineage or {})
    for k in keep.get(step, ()):
        if k in out:
            lineage[k] = out[k]
    lineage["dataset_version_id"] = run.config["dataset_version_id"]
    run.lineage = lineage


def resume(db: Session, run: PipelineRun, actor: Principal) -> Job:
    if run.status != "failed":
        raise ValueError("Only failed pipeline runs can be resumed")
    for st in run.steps:
        if st.status in ("failed", "skipped"):
            st.status, st.error = "pending", None
    run.status, run.error, run.finished_at = "queued", None, None
    job = Job(kind=JobKind.pipeline, payload={"pipeline_run_id": run.id, "resume": True})
    db.add(job)
    db.flush()
    run.job_id = job.id
    db.add(
        AuditEvent(
            actor=actor.username,
            action="pipeline.resumed",
            entity_type="pipeline_run",
            entity_id=run.id,
        )
    )
    return job


def on_promotion_executed(db: Session, pipeline_run_id: str) -> None:
    run = db.get(PipelineRun, pipeline_run_id)
    if run is None:
        return
    for st in run.steps:
        if st.status == "waiting_approval":
            st.status, st.finished_at = "succeeded", datetime.now(UTC)
    if run.status == "awaiting_approval":
        run.status = "succeeded"


def due_schedules(db: Session, now: datetime | None = None) -> list[PipelineSchedule]:
    now = now or datetime.now(UTC)
    out = []
    for s in db.scalars(select(PipelineSchedule).where(PipelineSchedule.enabled.is_(True))).all():
        last = s.last_run_at
        if last is not None and last.tzinfo is None:
            last = last.replace(tzinfo=UTC)
        if last is None or now - last >= timedelta(minutes=s.interval_minutes):
            out.append(s)
    return out


def run_due_schedules() -> list[str]:
    """Called by the scheduler loop (and tests). Returns triggered pipeline run ids."""
    started: list[tuple[str, str]] = []
    with session_scope() as db:
        for s in due_schedules(db):
            run = create_run(
                db,
                s.pipeline,
                s.config,
                Principal("scheduler", s.created_by, "engineer"),
                trigger="schedule",
            )
            s.last_run_at = datetime.now(UTC)
            started.append((run.id, run.job_id or ""))
    for _, job_id in started:
        jobs.submit(job_id)
    return [r for r, _ in started]
