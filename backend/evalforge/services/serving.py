"""Deployments (staging/production, canary), release health checks, rollback, inference."""

from __future__ import annotations

import hashlib
import time
import uuid
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import insert, select
from sqlalchemy.orm import Session

from evalforge.config import get_settings
from evalforge.models import Approval, AuditEvent, Deployment, ModelVersion, PredictionLog
from evalforge.observability import metrics, tracing
from evalforge.services.auth import Principal
from evalforge.services.governance import GovernanceError, executor, request_approval
from evalforge.services.model_cache import load_cached
from evalforge.services.registry import holdout_frame

ENVIRONMENTS = ("staging", "production")
MAX_INSTANCES = 1000


def active(db: Session, name: str, env: str) -> list[Deployment]:
    return list(
        db.scalars(
            select(Deployment).where(
                Deployment.model_name == name,
                Deployment.environment == env,
                Deployment.status == "active",
            )
        ).all()
    )


def stable_and_canary(
    db: Session, name: str, env: str
) -> tuple[Deployment | None, Deployment | None]:
    rows = active(db, name, env)
    return (
        next((d for d in rows if d.role == "stable"), None),
        next((d for d in rows if d.role == "canary"), None),
    )


def health_check(mv: ModelVersion, n: int = 25) -> dict[str, Any]:
    """Release health: model loads from MLflow, predicts valid labels, meets latency budget."""
    t0 = time.perf_counter()
    try:
        model = load_cached(mv.source_uri, mv.artifact_sha256)
    except Exception as exc:  # an artifact that cannot be loaded safely is an unhealthy release
        return {
            "healthy": False,
            "checks": {"model_loads": False},
            "error": f"{type(exc).__name__}: {exc}",
            "checked_at": datetime.now(UTC).isoformat(),
        }
    load_ms = (time.perf_counter() - t0) * 1000
    feats = mv.run.feature_columns or []
    X, y = holdout_frame(mv.run.dataset_version)
    sample = X[feats].head(n)
    lat = []
    preds = []
    for i in range(len(sample)):
        t = time.perf_counter()
        preds.append(str(model.predict(sample.iloc[[i]])[0]))
        lat.append((time.perf_counter() - t) * 1000)
    labels = set((mv.run.report or {}).get("labels", []))
    p95 = float(np.percentile(lat, 95)) if lat else 0.0
    agree = float(np.mean([p == t for p, t in zip(preds, y.head(n), strict=False)])) if preds else 0
    checks = {
        "model_loads": True,
        "predictions_valid": all(p in labels for p in preds),
        "latency_ok": p95 <= get_settings().gate_max_p95_latency_ms,
        "smoke_accuracy_ok": agree >= 0.5,
    }
    return {
        "healthy": all(checks.values()),
        "checks": checks,
        "load_ms": round(load_ms, 2),
        "p95_ms": round(p95, 3),
        "smoke_accuracy": round(agree, 4),
        "samples": len(preds),
        "checked_at": datetime.now(UTC).isoformat(),
    }


def _end(d: Deployment, status: str) -> None:
    d.status, d.ended_at = status, datetime.now(UTC)


def deploy(
    db: Session,
    mv: ModelVersion,
    env: str,
    traffic: int,
    actor: Principal,
    approval_id: str | None = None,
    note: str = "",
) -> Deployment:
    if env not in ENVIRONMENTS:
        raise GovernanceError(f"Unknown environment {env}")
    if not 1 <= traffic <= 100:
        raise GovernanceError("traffic_percent must be 1–100")
    stable, canary = stable_and_canary(db, mv.name, env)
    role = "stable" if traffic == 100 else "canary"
    if role == "canary" and stable is None:
        raise GovernanceError("A canary needs an active stable deployment to split traffic with")
    if stable is not None and stable.model_version_id == mv.id and role == "stable":
        raise GovernanceError(f"v{mv.version} is already the stable {env} deployment")
    health = health_check(mv)
    dep = Deployment(
        model_name=mv.name,
        model_version_id=mv.id,
        environment=env,
        traffic_percent=traffic,
        role=role,
        health=health,
        approval_id=approval_id,
        deployed_by=actor.username,
        note=note,
        previous_deployment_id=stable.id if stable else None,
    )
    if not health["healthy"]:
        dep.status, dep.ended_at = "failed", datetime.now(UTC)
        db.add(dep)
        db.flush()
        db.add(
            AuditEvent(
                actor=actor.username,
                action="deployment.failed_health",
                entity_type="deployment",
                entity_id=dep.id,
                data=health,
            )
        )
        raise GovernanceError(f"Release health check failed: {health['checks']}")
    if canary is not None:
        _end(canary, "superseded")
    if role == "stable" and stable is not None:
        _end(stable, "superseded")
    db.add(dep)
    db.flush()
    db.add(
        AuditEvent(
            actor=actor.username,
            action="deployment.created",
            entity_type="deployment",
            entity_id=dep.id,
            data={
                "model": mv.name,
                "version": mv.version,
                "env": env,
                "traffic": traffic,
                "role": role,
            },
        )
    )
    return dep


def rollback(
    db: Session, name: str, env: str, actor: Principal, approval_id: str | None = None
) -> Deployment:
    stable, canary = stable_and_canary(db, name, env)
    if canary is not None and stable is not None:
        # Rolling back a canary simply removes it; the stable keeps 100 %.
        _end(canary, "rolled_back")
        db.add(
            AuditEvent(
                actor=actor.username,
                action="deployment.canary_rolled_back",
                entity_type="deployment",
                entity_id=canary.id,
                data={"env": env},
            )
        )
        return stable
    if stable is None or stable.previous_deployment_id is None:
        raise GovernanceError(f"No previous {env} deployment of {name} to roll back to")
    prev = db.get(Deployment, stable.previous_deployment_id)
    assert prev is not None
    health = health_check(prev.model_version)
    if not health["healthy"]:
        raise GovernanceError("Previous version fails its health check; rollback aborted")
    _end(stable, "rolled_back")
    dep = Deployment(
        model_name=name,
        model_version_id=prev.model_version_id,
        environment=env,
        traffic_percent=100,
        role="stable",
        health=health,
        approval_id=approval_id,
        deployed_by=actor.username,
        note=f"rollback from v{stable.model_version.version}",
        previous_deployment_id=prev.previous_deployment_id,
    )
    db.add(dep)
    db.flush()
    db.add(
        AuditEvent(
            actor=actor.username,
            action="deployment.rolled_back",
            entity_type="deployment",
            entity_id=dep.id,
            data={
                "env": env,
                "from_version": stable.model_version.version,
                "to_version": prev.model_version.version,
            },
        )
    )
    return dep


def promote_canary(
    db: Session, name: str, env: str, actor: Principal, approval_id: str | None = None
) -> Deployment:
    stable, canary = stable_and_canary(db, name, env)
    if canary is None:
        raise GovernanceError("No active canary to promote")
    # Re-check now: the health recorded at canary deploy time may be stale (audit RES-02).
    health = health_check(canary.model_version)
    if not health["healthy"]:
        canary.health = health
        raise GovernanceError(
            f"Canary fails its health check now; promotion aborted: {health['checks']}"
        )
    _end(canary, "superseded")
    if stable is not None:
        _end(stable, "superseded")
    dep = Deployment(
        model_name=name,
        model_version_id=canary.model_version_id,
        environment=env,
        traffic_percent=100,
        role="stable",
        health=health,
        approval_id=approval_id,
        deployed_by=actor.username,
        note="canary promoted to 100 %",
        previous_deployment_id=stable.id if stable else None,
    )
    db.add(dep)
    db.flush()
    db.add(
        AuditEvent(
            actor=actor.username,
            action="deployment.canary_promoted",
            entity_type="deployment",
            entity_id=dep.id,
            data={"env": env},
        )
    )
    return dep


def request_production_deploy(
    db: Session, mv: ModelVersion, traffic: int, actor: Principal
) -> Approval:
    if mv.status != "approved":
        raise GovernanceError("Only approved (promoted) model versions can go to production")
    return request_approval(
        db,
        kind="deployment",
        target_type="model_version",
        target_id=mv.id,
        payload={"environment": "production", "traffic_percent": traffic},
        summary=f"Deploy {mv.name} v{mv.version} to production at {traffic}% traffic",
        actor=actor,
    )


@executor("deployment")
def _exec_deploy(db: Session, a: Approval, actor: Principal) -> dict[str, Any]:
    mv = db.get(ModelVersion, a.target_id)
    assert mv is not None
    dep = deploy(
        db,
        mv,
        a.payload["environment"],
        int(a.payload["traffic_percent"]),
        actor,
        approval_id=a.id,
        note=f"approved by {actor.username}",
    )
    return {"deployment_id": dep.id, "health": dep.health}


@executor("rollback")
def _exec_rollback(db: Session, a: Approval, actor: Principal) -> dict[str, Any]:
    dep = rollback(db, a.payload["model_name"], a.payload["environment"], actor, approval_id=a.id)
    return {"active_deployment_id": dep.id, "version": dep.model_version.version}


@executor("canary_promotion")
def _exec_canary(db: Session, a: Approval, actor: Principal) -> dict[str, Any]:
    dep = promote_canary(db, a.payload["model_name"], a.payload["environment"], actor, a.id)
    return {"deployment_id": dep.id, "version": dep.model_version.version}


# --------------------------------------------------------------------------- inference


def _json_safe(v: Any) -> Any:
    if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
        return None
    return v


def route(stable: Deployment | None, canary: Deployment | None, request_id: str) -> Deployment:
    if stable is None:
        raise LookupError("No active deployment")
    if canary is None:
        return stable
    bucket = int(hashlib.sha256(request_id.encode()).hexdigest()[:8], 16) % 100
    return canary if bucket < canary.traffic_percent else stable


_META: dict[str, tuple[list[str], list[str], dict[str, Any]]] = {}


def _model_meta(mv: ModelVersion) -> tuple[list[str], list[str], dict[str, Any]]:
    """Feature list, numeric columns and labels of a model version. Immutable per version,
    so cached: serving no longer lazy-loads run and dataset rows on every request."""
    meta = _META.get(mv.id)
    if meta is None:
        feats = list(mv.run.feature_columns or [])
        kinds = {c["name"]: c["kind"] for c in mv.run.dataset_version.schema_}
        numeric = [c for c in feats if kinds.get(c) == "numeric"]
        meta = _META[mv.id] = (feats, numeric, {"version": mv.version, "name": mv.name})
    return meta


def predict(
    db: Session,
    name: str,
    env: str,
    instances: list[dict[str, Any]],
    request_id: str,
    version: int | None = None,
) -> dict[str, Any]:
    if not instances or len(instances) > MAX_INSTANCES:
        raise GovernanceError(f"Send between 1 and {MAX_INSTANCES} instances")
    deps = active(db, name, env)
    if version is not None:
        dep = next((d for d in deps if d.model_version.version == version), None)
        if dep is None:
            raise LookupError(f"v{version} is not deployed in {env}")
    else:
        stable = next((d for d in deps if d.role == "stable"), None)
        canary = next((d for d in deps if d.role == "canary"), None)
        dep = route(stable, canary, request_id)
    mv = dep.model_version
    feats, numeric, _ = _model_meta(mv)
    missing = sorted({c for row in instances for c in feats if c not in row})
    if missing:
        raise GovernanceError(f"Missing required features: {missing[:20]}")
    # Build columns already converted: assigning converted columns into an existing frame
    # costs one block split per column (the serving profile's top pandas cost).
    num = set(numeric)
    cols: dict[str, Any] = {}
    for c in feats:
        vals = [row.get(c) for row in instances]
        cols[c] = (
            pd.to_numeric(pd.Series(vals, dtype=object), errors="coerce") if c in num else vals
        )
    X = pd.DataFrame(cols, columns=feats)
    model = load_cached(mv.source_uri, mv.artifact_sha256)
    with tracing.span(
        "model.predict",
        {
            "model.name": name,
            "model.version": mv.version,
            "deployment.id": dep.id,
            "deployment.role": dep.role,
            "deployment.environment": env,
            "batch.size": len(instances),
        },
    ):
        t0 = time.perf_counter()
        if hasattr(model, "predict_proba"):
            # One pass through the pipeline: labels are the arg-max of the probabilities
            # (identical to Pipeline.predict for these classifiers).
            proba = model.predict_proba(X)
            preds = [str(p) for p in model.classes_[proba.argmax(axis=1)]]
            probs: list[float] | None = proba.max(axis=1).tolist()
        else:
            preds, probs = [str(p) for p in model.predict(X)], None
        ms = (time.perf_counter() - t0) * 1000
    metrics.PREDICTIONS.labels(name, str(mv.version), env, dep.role).inc(len(instances))
    metrics.INFERENCE_LATENCY.labels(name, env).observe(ms / 1000)
    now = datetime.now(UTC)
    rows = [
        {
            "id": uuid.uuid4().hex,
            "request_id": request_id,
            "deployment_id": dep.id,
            "model_version_id": mv.id,
            "environment": env,
            "inputs": {c: _json_safe(row.get(c)) for c in feats},
            "prediction": preds[i],
            "probability": probs[i] if probs else None,
            "latency_ms": ms / len(instances),
            "created_at": now,
        }
        for i, row in enumerate(instances)
    ]
    db.execute(insert(PredictionLog), rows)  # one executemany instead of N ORM flushes
    return {
        "request_id": request_id,
        "model": name,
        "version": mv.version,
        "deployment_id": dep.id,
        "role": dep.role,
        "environment": env,
        "latency_ms": round(ms, 3),
        "predictions": [
            {"id": r["id"], "label": p, "probability": round(probs[i], 6) if probs else None}
            for i, (r, p) in enumerate(zip(rows, preds, strict=True))
        ],
    }
