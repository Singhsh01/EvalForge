"""Registry, approvals, deployments and inference endpoints."""

from __future__ import annotations

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from evalforge.db import get_db
from evalforge.models import Approval, Deployment, ModelVersion, PredictionLog, TrainingRun
from evalforge.services import registry, serving
from evalforge.services.auth import Principal, require
from evalforge.services.governance import GovernanceError, decide, request_approval

router = APIRouter(prefix="/api", tags=["registry & deployment"])
Viewer = Annotated[Principal, Depends(require("viewer"))]
Engineer = Annotated[Principal, Depends(require("engineer"))]
Approver = Annotated[Principal, Depends(require("approver"))]


def _err(exc: Exception) -> HTTPException:
    if isinstance(exc, PermissionError):
        return HTTPException(403, str(exc))
    if isinstance(exc, LookupError):
        return HTTPException(404, str(exc))
    return HTTPException(409, str(exc))


def mv_out(mv: ModelVersion) -> dict[str, Any]:
    r = mv.run
    return {
        "id": mv.id,
        "name": mv.name,
        "version": mv.version,
        "status": mv.status,
        "is_champion": mv.is_champion,
        "run_id": mv.run_id,
        "algorithm": r.algorithm,
        "dataset_version": r.dataset_version.version,
        "dataset_version_id": r.dataset_version_id,
        "metrics": r.metrics,
        "gate": mv.gate,
        "source_uri": mv.source_uri,
        "mlflow_run_id": r.mlflow_run_id,
        "created_by": mv.created_by,
        "created_at": mv.created_at.isoformat(),
    }


def appr_out(a: Approval) -> dict[str, Any]:
    return {
        k: getattr(a, k)
        for k in (
            "id",
            "kind",
            "target_type",
            "target_id",
            "payload",
            "summary",
            "status",
            "requested_by",
            "decided_by",
            "decision_reason",
            "result",
        )
    } | {
        "created_at": a.created_at.isoformat(),
        "decided_at": a.decided_at.isoformat() if a.decided_at else None,
    }


def dep_out(d: Deployment) -> dict[str, Any]:
    return {
        "id": d.id,
        "model_name": d.model_name,
        "model_version_id": d.model_version_id,
        "version": d.model_version.version,
        "environment": d.environment,
        "traffic_percent": d.traffic_percent,
        "role": d.role,
        "status": d.status,
        "health": d.health,
        "approval_id": d.approval_id,
        "previous_deployment_id": d.previous_deployment_id,
        "deployed_by": d.deployed_by,
        "note": d.note,
        "created_at": d.created_at.isoformat(),
        "ended_at": d.ended_at.isoformat() if d.ended_at else None,
    }


# --------------------------------------------------------------------------- registry


class RegisterIn(BaseModel):
    run_id: str


@router.get("/registry/models")
def list_models(_: Viewer, db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    rows = db.scalars(
        select(ModelVersion).order_by(ModelVersion.name, ModelVersion.version.desc())
    ).all()
    out: dict[str, dict[str, Any]] = {}
    for mv in rows:
        m = out.setdefault(mv.name, {"name": mv.name, "versions": [], "champion": None})
        m["versions"].append(mv_out(mv))
        if mv.is_champion:
            m["champion"] = mv.version
    return list(out.values())


@router.post("/registry/versions", status_code=201)
def register(body: RegisterIn, p: Engineer, db: Session = Depends(get_db)) -> dict[str, Any]:
    run = db.get(TrainingRun, body.run_id)
    if run is None:
        raise HTTPException(404, "Run not found")
    try:
        mv = registry.register(db, run, p)
        mv.gate = registry.evaluate_gate(db, mv)
        db.commit()
    except GovernanceError as exc:
        db.rollback()
        raise _err(exc) from exc
    return mv_out(mv)


@router.get("/registry/versions/{mv_id}")
def get_version(mv_id: str, _: Viewer, db: Session = Depends(get_db)) -> dict[str, Any]:
    mv = db.get(ModelVersion, mv_id)
    if mv is None:
        raise HTTPException(404, "Model version not found")
    deps = db.scalars(
        select(Deployment)
        .where(Deployment.model_version_id == mv_id)
        .order_by(Deployment.created_at.desc())
    ).all()
    return mv_out(mv) | {"deployments": [dep_out(d) for d in deps]}


@router.post("/registry/versions/{mv_id}/gate")
def run_gate(mv_id: str, _: Engineer, db: Session = Depends(get_db)) -> dict[str, Any]:
    mv = db.get(ModelVersion, mv_id)
    if mv is None:
        raise HTTPException(404, "Model version not found")
    try:
        mv.gate = registry.evaluate_gate(db, mv)
    except GovernanceError as exc:
        raise _err(exc) from exc
    db.commit()
    return mv.gate


@router.get("/registry/versions/{mv_id}/card")
def card(mv_id: str, _: Viewer, db: Session = Depends(get_db)) -> dict[str, Any]:
    mv = db.get(ModelVersion, mv_id)
    if mv is None:
        raise HTTPException(404, "Model version not found")
    return registry.model_card(mv)


@router.post("/registry/versions/{mv_id}/request-promotion", status_code=201)
def request_promotion(mv_id: str, p: Engineer, db: Session = Depends(get_db)) -> dict[str, Any]:
    mv = db.get(ModelVersion, mv_id)
    if mv is None:
        raise HTTPException(404, "Model version not found")
    try:
        mv.gate = registry.evaluate_gate(db, mv)
        if not mv.gate["passed"]:
            failed = [c["id"] for c in mv.gate["checks"] if c["blocking"] and not c["passed"]]
            raise GovernanceError(f"Quality gate failed: {failed}")
        a = request_approval(
            db,
            kind="model_promotion",
            target_type="model_version",
            target_id=mv.id,
            payload={"name": mv.name, "version": mv.version},
            summary=f"Promote {mv.name} v{mv.version} to champion",
            actor=p,
        )
        db.commit()
    except GovernanceError as exc:
        db.commit()  # keep the refreshed gate result
        raise _err(exc) from exc
    return appr_out(a)


# --------------------------------------------------------------------------- approvals


class DecisionIn(BaseModel):
    approve: bool
    reason: str = Field(min_length=3, max_length=2000)


@router.get("/approvals")
def list_approvals(
    _: Viewer, status: str | None = None, db: Session = Depends(get_db)
) -> list[dict[str, Any]]:
    q = select(Approval).order_by(Approval.created_at.desc()).limit(200)
    if status:
        q = q.where(Approval.status == status)
    return [appr_out(a) for a in db.scalars(q).all()]


@router.post("/approvals/{approval_id}/decision")
def decide_approval(
    approval_id: str, body: DecisionIn, p: Approver, db: Session = Depends(get_db)
) -> dict[str, Any]:
    a = db.get(Approval, approval_id)
    if a is None:
        raise HTTPException(404, "Approval not found")
    try:
        decide(db, a, approve=body.approve, actor=p, reason=body.reason)
        db.commit()
    except (GovernanceError, PermissionError) as exc:
        db.rollback()
        raise _err(exc) from exc
    return appr_out(a)


# --------------------------------------------------------------------------- deployments


class DeployIn(BaseModel):
    model_version_id: str
    environment: str = Field(pattern="^(staging|production)$")
    traffic_percent: int = Field(default=100, ge=1, le=100)


class EnvActionIn(BaseModel):
    model_name: str
    environment: str = Field(pattern="^(staging|production)$")


@router.get("/deployments")
def list_deployments(
    _: Viewer, model_name: str | None = None, db: Session = Depends(get_db)
) -> list[dict[str, Any]]:
    q = select(Deployment).order_by(Deployment.created_at.desc()).limit(300)
    if model_name:
        q = q.where(Deployment.model_name == model_name)
    return [dep_out(d) for d in db.scalars(q).all()]


@router.post("/deployments", status_code=201)
def create_deployment(body: DeployIn, p: Engineer, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Staging deploys happen immediately; production deploys create an approval request."""
    mv = db.get(ModelVersion, body.model_version_id)
    if mv is None:
        raise HTTPException(404, "Model version not found")
    try:
        if body.environment == "production":
            a = serving.request_production_deploy(db, mv, body.traffic_percent, p)
            db.commit()
            return {"approval": appr_out(a)}
        dep = serving.deploy(db, mv, "staging", body.traffic_percent, p)
        db.commit()
        return {"deployment": dep_out(dep)}
    except GovernanceError as exc:
        db.commit()  # persist failed-health records for the audit trail
        raise _err(exc) from exc


def _env_action(kind: str, body: EnvActionIn, p: Principal, db: Session) -> dict[str, Any]:
    try:
        if body.environment == "production":
            a = request_approval(
                db,
                kind=kind,
                target_type="environment",
                target_id=f"{body.model_name}:production",
                payload=body.model_dump(),
                summary=f"{kind.replace('_', ' ').title()} of {body.model_name} in production",
                actor=p,
            )
            db.commit()
            return {"approval": appr_out(a)}
        fn = serving.rollback if kind == "rollback" else serving.promote_canary
        dep = fn(db, body.model_name, "staging", p)
        db.commit()
        return {"deployment": dep_out(dep)}
    except GovernanceError as exc:
        db.rollback()
        raise _err(exc) from exc


@router.post("/deployments/rollback")
def rollback(body: EnvActionIn, p: Engineer, db: Session = Depends(get_db)) -> dict[str, Any]:
    return _env_action("rollback", body, p, db)


@router.post("/deployments/promote-canary")
def promote_canary(body: EnvActionIn, p: Engineer, db: Session = Depends(get_db)) -> dict[str, Any]:
    return _env_action("canary_promotion", body, p, db)


@router.post("/deployments/{dep_id}/health")
def recheck_health(dep_id: str, _: Engineer, db: Session = Depends(get_db)) -> dict[str, Any]:
    d = db.get(Deployment, dep_id)
    if d is None:
        raise HTTPException(404, "Deployment not found")
    d.health = serving.health_check(d.model_version)
    db.commit()
    return d.health


# --------------------------------------------------------------------------- serving


class PredictIn(BaseModel):
    instances: list[dict[str, Any]] = Field(min_length=1, max_length=serving.MAX_INSTANCES)


class FeedbackIn(BaseModel):
    labels: dict[str, str] = Field(
        description="prediction id -> observed label", min_length=1, max_length=5000
    )


@router.post("/serve/{model_name}/predict")
def predict(
    model_name: str,
    body: PredictIn,
    request: Request,
    _: Viewer,
    environment: str = Query("production", pattern="^(staging|production)$"),
    version: int | None = None,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex
    try:
        out = serving.predict(db, model_name, environment, body.instances, rid, version)
        db.commit()
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except GovernanceError as exc:
        raise HTTPException(422, str(exc)) from exc
    return out


@router.post("/serve/feedback")
def feedback(body: FeedbackIn, _: Engineer, db: Session = Depends(get_db)) -> dict[str, int]:
    rows = db.scalars(select(PredictionLog).where(PredictionLog.id.in_(list(body.labels)))).all()
    for r in rows:
        r.label = str(body.labels[r.id])
    db.commit()
    return {"updated": len(rows), "unknown": len(body.labels) - len(rows)}
