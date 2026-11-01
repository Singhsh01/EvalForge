"""Drift monitoring, incidents and controlled retraining."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from evalforge.db import get_db
from evalforge.models import (
    AuditEvent,
    CopilotSession,
    DriftReport,
    Incident,
    IncidentEvent,
    ModelVersion,
)
from evalforge.services import drift
from evalforge.services.auth import Principal, require
from evalforge.services.governance import GovernanceError, request_approval

router = APIRouter(prefix="/api", tags=["monitoring"])
Viewer = Annotated[Principal, Depends(require("viewer"))]
Engineer = Annotated[Principal, Depends(require("engineer"))]


def report_out(r: DriftReport, full: bool = True) -> dict[str, Any]:
    out = {
        "id": r.id,
        "model_name": r.model_name,
        "model_version_id": r.model_version_id,
        "source": r.source,
        "reference": r.reference,
        "current": r.current,
        "summary": r.summary,
        "thresholds": r.thresholds,
        "created_by": r.created_by,
        "created_at": r.created_at.isoformat(),
    }
    if full:
        out |= {
            "features": r.features,
            "prediction_drift": r.prediction_drift,
            "performance": r.performance,
        }
    return out


def incident_out(i: Incident, full: bool = False) -> dict[str, Any]:
    out = {
        "id": i.id,
        "kind": i.kind,
        "severity": i.severity,
        "status": i.status,
        "title": i.title,
        "source_type": i.source_type,
        "source_id": i.source_id,
        "details": i.details,
        "findings": i.findings,
        "created_at": i.created_at.isoformat(),
        "resolved_at": i.resolved_at.isoformat() if i.resolved_at else None,
    }
    if full:
        out["events"] = [
            {
                "id": e.id,
                "kind": e.kind,
                "message": e.message,
                "data": e.data,
                "actor": e.actor,
                "created_at": e.created_at.isoformat(),
            }
            for e in i.events
        ]
    return out


class DriftIn(BaseModel):
    model_version_id: str
    source: str = Field("predictions", pattern="^(predictions|dataset_version)$")
    environment: str = Field("production", pattern="^(staging|production)$")
    hours: int = Field(24, ge=1, le=24 * 90)
    dataset_version_id: str | None = None
    thresholds: dict[str, float] | None = None


@router.post("/drift/reports", status_code=201)
def create_report(body: DriftIn, p: Engineer, db: Session = Depends(get_db)) -> dict[str, Any]:
    mv = db.get(ModelVersion, body.model_version_id)
    if mv is None:
        raise HTTPException(404, "Model version not found")
    try:
        r = drift.build_report(
            db,
            mv,
            source=body.source,
            environment=body.environment,
            hours=body.hours,
            dataset_version_id=body.dataset_version_id,
            thresholds=body.thresholds,
            actor=p.username,
        )
        db.commit()
    except drift.DriftError as exc:
        db.rollback()
        raise HTTPException(422, str(exc)) from exc
    return report_out(r)


@router.get("/drift/reports")
def list_reports(
    _: Viewer, model_name: str | None = None, db: Session = Depends(get_db)
) -> list[dict[str, Any]]:
    q = select(DriftReport).order_by(DriftReport.created_at.desc()).limit(100)
    if model_name:
        q = q.where(DriftReport.model_name == model_name)
    return [report_out(r, full=False) for r in db.scalars(q).all()]


@router.get("/drift/reports/{report_id}")
def get_report(report_id: str, _: Viewer, db: Session = Depends(get_db)) -> dict[str, Any]:
    r = db.get(DriftReport, report_id)
    if r is None:
        raise HTTPException(404, "Report not found")
    return report_out(r)


@router.get("/incidents")
def list_incidents(
    _: Viewer, status: str | None = None, db: Session = Depends(get_db)
) -> list[dict[str, Any]]:
    q = select(Incident).order_by(Incident.created_at.desc()).limit(200)
    if status:
        q = q.where(Incident.status == status)
    return [incident_out(i) for i in db.scalars(q).all()]


@router.get("/incidents/{incident_id}")
def get_incident(incident_id: str, _: Viewer, db: Session = Depends(get_db)) -> dict[str, Any]:
    i = db.get(Incident, incident_id)
    if i is None:
        raise HTTPException(404, "Incident not found")
    sessions = db.scalars(
        select(CopilotSession)
        .where(CopilotSession.incident_id == i.id)
        .order_by(CopilotSession.created_at)
    ).all()
    return incident_out(i, full=True) | {
        "investigations": [
            {
                "id": s.id,
                "status": s.status,
                "question": s.question,
                "report": s.report,
                "created_at": s.created_at.isoformat(),
            }
            for s in sessions
        ]
    }


class StatusIn(BaseModel):
    status: str = Field(pattern="^(open|investigating|mitigating|resolved)$")
    note: str = Field(min_length=3, max_length=2000)


@router.post("/incidents/{incident_id}/status")
def set_status(
    incident_id: str, body: StatusIn, p: Engineer, db: Session = Depends(get_db)
) -> dict[str, Any]:
    i = db.get(Incident, incident_id)
    if i is None:
        raise HTTPException(404, "Incident not found")
    i.status = body.status
    if body.status == "resolved":
        i.resolved_at = datetime.now(UTC)
    i.events.append(
        IncidentEvent(kind="status", message=f"{body.status}: {body.note}", actor=p.username)
    )
    db.add(
        AuditEvent(
            actor=p.username,
            action="incident.status",
            entity_type="incident",
            entity_id=i.id,
            data={"status": body.status},
        )
    )
    db.commit()
    return incident_out(i, full=True)


class PlanIn(BaseModel):
    dataset_version_id: str | None = None


@router.post("/incidents/{incident_id}/retraining-plan")
def plan(
    incident_id: str, body: PlanIn, _: Engineer, db: Session = Depends(get_db)
) -> dict[str, Any]:
    i = db.get(Incident, incident_id)
    if i is None:
        raise HTTPException(404, "Incident not found")
    try:
        return drift.retraining_plan(db, i, body.dataset_version_id)
    except drift.DriftError as exc:
        raise HTTPException(422, str(exc)) from exc


class RetrainIn(BaseModel):
    dataset_version_id: str | None = None
    override_reason: str | None = Field(None, max_length=2000)


@router.post("/incidents/{incident_id}/request-retraining", status_code=201)
def request_retraining(
    incident_id: str, body: RetrainIn, p: Engineer, db: Session = Depends(get_db)
) -> dict[str, Any]:
    i = db.get(Incident, incident_id)
    if i is None:
        raise HTTPException(404, "Incident not found")
    try:
        pl = drift.retraining_plan(db, i, body.dataset_version_id)
        if not pl["justified"] and not body.override_reason:
            raise GovernanceError(
                "Retraining is not justified by the evidence: "
                + " ".join(pl["reasons"])
                + " Provide override_reason to "
                "request it anyway."
            )
        a = request_approval(
            db,
            kind="retraining",
            target_type="incident",
            target_id=i.id,
            payload={
                "incident_id": i.id,
                "config": pl["config"],
                "plan": pl,
                "override_reason": body.override_reason,
            },
            summary=f"Retrain {pl['config']['algorithm']} on dataset "
            f"v{pl['training_data']['dataset_version']} for incident "
            f"{i.id[:8]}",
            actor=p,
        )
        i.events.append(
            IncidentEvent(
                kind="action",
                actor=p.username,
                message=(
                    f"Retraining requested (approval {a.id[:8]}). Justified: {pl['justified']}. "
                    + " ".join(pl["reasons"])
                ),
                data={"approval_id": a.id},
            )
        )
        db.commit()
    except (GovernanceError, drift.DriftError) as exc:
        db.rollback()
        raise HTTPException(409, str(exc)) from exc
    return {"approval_id": a.id, "plan": pl}
