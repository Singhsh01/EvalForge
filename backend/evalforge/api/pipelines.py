from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from evalforge.db import get_db
from evalforge.models import AuditEvent, PipelineRun, PipelineSchedule
from evalforge.services import jobs, pipelines
from evalforge.services.auth import Principal, require

router = APIRouter(prefix="/api/pipelines", tags=["pipelines"])
Viewer = Annotated[Principal, Depends(require("viewer"))]
Engineer = Annotated[Principal, Depends(require("engineer"))]


def run_out(r: PipelineRun, steps: bool = True) -> dict[str, Any]:
    out = {
        "id": r.id,
        "pipeline": r.pipeline,
        "status": r.status,
        "trigger": r.trigger,
        "triggered_by": r.triggered_by,
        "config": r.config,
        "config_hash": r.config_hash,
        "lineage": r.lineage,
        "error": r.error,
        "job_id": r.job_id,
        "created_at": r.created_at.isoformat(),
        "finished_at": r.finished_at.isoformat() if r.finished_at else None,
    }
    if steps:
        out["steps"] = [
            {
                "name": s.name,
                "position": s.position,
                "depends_on": s.depends_on,
                "status": s.status,
                "attempts": s.attempts,
                "output": s.output,
                "error": s.error,
                "started_at": s.started_at.isoformat() if s.started_at else None,
                "finished_at": s.finished_at.isoformat() if s.finished_at else None,
            }
            for s in r.steps
        ]
    return out


class RunIn(BaseModel):
    pipeline: str = "continuous_training"
    config: dict[str, Any]


class ScheduleIn(BaseModel):
    pipeline: str = "continuous_training"
    config: dict[str, Any]
    interval_minutes: int = Field(ge=5, le=60 * 24 * 30)


@router.get("")
def definitions(_: Viewer) -> list[dict[str, Any]]:
    return [
        {
            "name": n,
            "steps": [{"name": s, "depends_on": d} for s, _, d in steps],
            "config_schema": pipelines.TrainPipelineConfig.model_json_schema(),
        }
        for n, steps in pipelines.PIPELINES.items()
    ]


@router.post("/runs", status_code=202)
def trigger(body: RunIn, p: Engineer, db: Session = Depends(get_db)) -> dict[str, Any]:
    try:
        run = pipelines.create_run(db, body.pipeline, body.config, p)
    except (ValueError, ValidationError) as exc:
        raise HTTPException(422, str(exc)) from exc
    db.commit()
    jobs.submit(run.job_id or "")
    db.refresh(run)
    return run_out(run)


@router.get("/runs")
def list_runs(_: Viewer, db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    rows = db.scalars(select(PipelineRun).order_by(PipelineRun.created_at.desc()).limit(100))
    return [run_out(r) for r in rows.all()]


@router.get("/runs/{run_id}")
def get_run(run_id: str, _: Viewer, db: Session = Depends(get_db)) -> dict[str, Any]:
    r = db.get(PipelineRun, run_id)
    if r is None:
        raise HTTPException(404, "Pipeline run not found")
    return run_out(r)


@router.post("/runs/{run_id}/resume", status_code=202)
def resume(run_id: str, p: Engineer, db: Session = Depends(get_db)) -> dict[str, Any]:
    r = db.get(PipelineRun, run_id)
    if r is None:
        raise HTTPException(404, "Pipeline run not found")
    try:
        job = pipelines.resume(db, r, p)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    db.commit()
    jobs.submit(job.id)
    db.refresh(r)
    return run_out(r)


@router.get("/schedules")
def schedules(_: Viewer, db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    return [
        {
            "id": s.id,
            "pipeline": s.pipeline,
            "config": s.config,
            "interval_minutes": s.interval_minutes,
            "enabled": s.enabled,
            "last_run_at": s.last_run_at.isoformat() if s.last_run_at else None,
            "created_by": s.created_by,
        }
        for s in db.scalars(select(PipelineSchedule)).all()
    ]


@router.post("/schedules", status_code=201)
def create_schedule(body: ScheduleIn, p: Engineer, db: Session = Depends(get_db)) -> dict[str, str]:
    try:
        pipelines.verify_definition(body.pipeline)
        cfg = pipelines.TrainPipelineConfig(**body.config).model_dump()
    except (ValueError, ValidationError) as exc:
        raise HTTPException(422, str(exc)) from exc
    s = PipelineSchedule(
        pipeline=body.pipeline,
        config=cfg,
        interval_minutes=body.interval_minutes,
        created_by=p.username,
    )
    db.add(s)
    db.flush()
    db.add(
        AuditEvent(
            actor=p.username,
            action="schedule.created",
            entity_type="schedule",
            entity_id=s.id,
            data={"interval_minutes": s.interval_minutes},
        )
    )
    db.commit()
    return {"id": s.id}


@router.post("/schedules/{schedule_id}/toggle")
def toggle(schedule_id: str, p: Engineer, db: Session = Depends(get_db)) -> dict[str, Any]:
    s = db.get(PipelineSchedule, schedule_id)
    if s is None:
        raise HTTPException(404, "Schedule not found")
    s.enabled = not s.enabled
    db.commit()
    return {"id": s.id, "enabled": s.enabled}


@router.post("/schedules/tick")
def tick(_: Engineer) -> dict[str, list[str]]:
    """Run due schedules now (the API's scheduler loop calls the same function)."""
    return {"triggered": pipelines.run_due_schedules()}
