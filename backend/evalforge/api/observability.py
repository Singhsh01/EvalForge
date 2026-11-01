"""Trace explorer, system health, cost/performance analytics and audit log."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from evalforge.db import get_db
from evalforge.models import (
    AuditEvent,
    Deployment,
    EvalRun,
    Job,
    JobStatus,
    LLMCall,
    PipelineStep,
    PredictionLog,
    Span,
    TrainingRun,
)
from evalforge.observability import tracing
from evalforge.services.auth import Principal, require

router = APIRouter(prefix="/api", tags=["observability"])
Viewer = Annotated[Principal, Depends(require("viewer"))]


def span_out(s: Span) -> dict[str, Any]:
    return {
        "span_id": s.span_id,
        "trace_id": s.trace_id,
        "parent_id": s.parent_id,
        "name": s.name,
        "kind": s.kind,
        "service": s.service,
        "start_time": s.start_time.isoformat()
        if s.start_time.tzinfo
        else s.start_time.replace(tzinfo=UTC).isoformat(),
        "duration_ms": round(s.duration_ms, 3),
        "status": s.status,
        "status_message": s.status_message,
        "attributes": s.attributes,
        "events": s.events,
    }


@router.get("/traces")
def list_traces(
    _: Viewer,
    db: Session = Depends(get_db),
    status: str | None = Query(None, pattern="^(error|ok|unset)$"),
    q: str | None = Query(None, max_length=100),
    limit: int = Query(50, ge=1, le=500),
) -> list[dict[str, Any]]:
    """Root spans (one per trace), newest first, with span counts and error flags."""
    tracing.flush(max_age_s=1.0)
    roots_q = select(Span).where(Span.parent_id.is_(None))
    if q:
        roots_q = roots_q.where(Span.name.contains(q))
    roots = db.scalars(roots_q.order_by(Span.start_time.desc()).limit(limit * 3)).all()
    # One grouped aggregate for all candidate traces instead of one query per root
    # (audit PERF-03: up to 150 queries per request).
    agg = (
        {
            tid: (n, int(errs or 0))
            for tid, n, errs in db.execute(
                select(
                    Span.trace_id,
                    func.count(),
                    func.sum(case((Span.status == "error", 1), else_=0)),
                )
                .where(Span.trace_id.in_({r.trace_id for r in roots}))
                .group_by(Span.trace_id)
            ).all()
        }
        if roots
        else {}
    )
    out = []
    for r in roots:
        n, errs = agg.get(r.trace_id, (1, 0))
        has_error = bool(errs) or r.status == "error"
        if status == "error" and not has_error:
            continue
        out.append(span_out(r) | {"span_count": n, "has_error": has_error})
        if len(out) >= limit:
            break
    return out


@router.get("/traces/{trace_id}")
def get_trace(trace_id: str, _: Viewer, db: Session = Depends(get_db)) -> dict[str, Any]:
    tracing.flush()  # single-trace view: always current
    spans = db.scalars(
        select(Span).where(Span.trace_id == trace_id).order_by(Span.start_time)
    ).all()
    if not spans:
        raise HTTPException(404, "Trace not found")
    t0 = min(s.start_time for s in spans)
    rows = [
        span_out(s) | {"offset_ms": round((s.start_time - t0).total_seconds() * 1000, 3)}
        for s in spans
    ]
    attrs = {k: v for s in spans for k, v in (s.attributes or {}).items()}
    correlations = {
        k: attrs[k]
        for k in (
            "job.id",
            "eval.run_id",
            "pipeline.run_id",
            "copilot.session_id",
            "model.name",
            "model.version",
            "deployment.id",
            "mlflow.run_id",
            "http.request_id",
        )
        if k in attrs
    }
    return {
        "trace_id": trace_id,
        "spans": rows,
        "correlations": correlations,
        "duration_ms": round(
            max((s.start_time - t0).total_seconds() * 1000 + s.duration_ms for s in spans), 3
        ),
    }


def _pct(values: list[float], q: float) -> float | None:
    return round(float(np.percentile(values, q)), 3) if values else None


@router.get("/observability/summary")
def summary(
    _: Viewer, db: Session = Depends(get_db), hours: int = Query(24, ge=1, le=24 * 30)
) -> dict[str, Any]:
    """Operational health computed from stored telemetry over the last ``hours``."""
    tracing.flush()
    since = datetime.now(UTC) - timedelta(hours=hours)
    http = db.scalars(select(Span).where(Span.kind == "server", Span.start_time >= since)).all()
    by_route: dict[str, list[Span]] = {}
    for s in http:
        by_route.setdefault(str(s.attributes.get("http.route", s.name)), []).append(s)
    routes = []
    for route, spans in sorted(by_route.items(), key=lambda kv: -len(kv[1]))[:25]:
        lat = [s.duration_ms for s in spans]
        errors = sum(1 for s in spans if int(s.attributes.get("http.status_code", 200)) >= 500)
        client_err = sum(
            1 for s in spans if 400 <= int(s.attributes.get("http.status_code", 200)) < 500
        )
        routes.append(
            {
                "route": route,
                "requests": len(spans),
                "p50_ms": _pct(lat, 50),
                "p95_ms": _pct(lat, 95),
                "server_errors": errors,
                "client_errors": client_err,
                "error_rate": round(errors / len(spans), 4),
            }
        )
    jobs = db.execute(
        select(Job.kind, Job.status, func.count())
        .where(Job.created_at >= since)
        .group_by(Job.kind, Job.status)
    ).all()
    retries = (
        db.scalar(
            select(func.coalesce(func.sum(PipelineStep.attempts - 1), 0)).where(
                PipelineStep.attempts > 1
            )
        )
        or 0
    )
    preds = db.scalars(
        select(PredictionLog.latency_ms).where(PredictionLog.created_at >= since)
    ).all()
    llm = db.execute(
        select(
            LLMCall.model,
            func.count(),
            func.sum(LLMCall.input_tokens),
            func.sum(LLMCall.output_tokens),
            func.sum(LLMCall.cost_usd),
            func.avg(LLMCall.latency_ms),
            func.sum((LLMCall.error.is_not(None)).cast(type_=func.count().type)),
        )
        .where(LLMCall.created_at >= since)
        .group_by(LLMCall.model)
    ).all()
    total_spans = (
        db.scalar(select(func.count()).select_from(Span).where(Span.start_time >= since)) or 0
    )
    error_spans = (
        db.scalar(
            select(func.count())
            .select_from(Span)
            .where(Span.start_time >= since, Span.status == "error")
        )
        or 0
    )
    return {
        "window_hours": hours,
        "http": {
            "requests": len(http),
            "routes": routes,
            "server_error_rate": round(
                sum(r["server_errors"] for r in routes) / max(len(http), 1), 4
            ),
        },
        "jobs": [{"kind": k.value, "status": s.value, "count": c} for k, s, c in jobs],
        "worker_failures": sum(c for _, s, c in jobs if s == JobStatus.failed),
        "pipeline_step_retries": int(retries),
        "inference": {
            "predictions": len(preds),
            "p50_ms": _pct(list(preds), 50),
            "p95_ms": _pct(list(preds), 95),
        },
        "llm": [
            {
                "model": m,
                "calls": n,
                "input_tokens": int(i or 0),
                "output_tokens": int(o or 0),
                "cost_usd": round(float(c or 0), 6),
                "avg_latency_ms": round(float(a or 0), 1),
                "errors": int(e or 0),
            }
            for m, n, i, o, c, a, e in llm
        ],
        "spans": {"total": total_spans, "errors": error_spans},
    }


@router.get("/analytics/cost-performance")
def cost_performance(_: Viewer, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Quality vs. latency vs. cost, from measured runs only."""
    models = []
    for r in db.scalars(
        select(TrainingRun)
        .where(TrainingRun.status == JobStatus.succeeded)
        .order_by(TrainingRun.created_at.desc())
        .limit(200)
    ).all():
        m = r.metrics or {}
        models.append(
            {
                "run_id": r.id,
                "algorithm": r.algorithm,
                "dataset": r.dataset_version.dataset.name,
                "dataset_version": r.dataset_version.version,
                "f1": m.get("f1"),
                "latency_p50_ms": m.get("latency_single_p50_ms"),
                "latency_p95_ms": m.get("latency_single_p95_ms"),
                "training_seconds": r.training_seconds,
            }
        )
    evals = []
    for e in db.scalars(
        select(EvalRun)
        .where(EvalRun.status == "succeeded")
        .order_by(EvalRun.created_at.desc())
        .limit(200)
    ).all():
        s = e.summary or {}
        evals.append(
            {
                "eval_run_id": e.id,
                "suite": f"{e.suite.name}@v{e.suite.version}",
                "subject": e.subject_key,
                "pass_rate": s.get("pass_rate"),
                "p95_ms": (s.get("latency_ms") or {}).get("p95"),
                "tokens": s.get("input_tokens", 0) + s.get("output_tokens", 0),
                "cost_usd": s.get("cost_usd"),
            }
        )
    serving = []
    for d in db.scalars(select(Deployment).where(Deployment.status == "active")).all():
        lat = db.scalars(
            select(PredictionLog.latency_ms).where(PredictionLog.deployment_id == d.id)
        ).all()
        serving.append(
            {
                "deployment_id": d.id,
                "model": d.model_name,
                "version": d.model_version.version,
                "environment": d.environment,
                "role": d.role,
                "predictions": len(lat),
                "p50_ms": _pct(list(lat), 50),
                "p95_ms": _pct(list(lat), 95),
            }
        )
    return {"training": models, "evaluations": evals, "serving": serving}


@router.get("/audit")
def audit(
    _: Viewer,
    db: Session = Depends(get_db),
    action: str | None = None,
    actor: str | None = None,
    limit: int = Query(100, ge=1, le=1000),
) -> list[dict[str, Any]]:
    q = select(AuditEvent).order_by(AuditEvent.created_at.desc()).limit(limit)
    if action:
        q = q.where(AuditEvent.action.startswith(action))
    if actor:
        q = q.where(AuditEvent.actor == actor)
    return [
        {
            "id": e.id,
            "actor": e.actor,
            "action": e.action,
            "entity_type": e.entity_type,
            "entity_id": e.entity_id,
            "data": e.data,
            "created_at": e.created_at.isoformat(),
        }
        for e in db.scalars(q).all()
    ]
