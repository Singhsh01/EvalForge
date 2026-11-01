"""AI MLOps Copilot: agent configurations, investigation sessions, incident investigation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from evalforge.copilot import agent
from evalforge.copilot.tools import TOOLS, ToolError
from evalforge.db import get_db, session_scope
from evalforge.models import AgentConfig, AuditEvent, CopilotSession, EvalSuite
from evalforge.services import evals
from evalforge.services.auth import Principal, require
from evalforge.services.governance import executor

router = APIRouter(prefix="/api", tags=["copilot"])
Viewer = Annotated[Principal, Depends(require("viewer"))]
Engineer = Annotated[Principal, Depends(require("engineer"))]
_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="copilot")


def cfg_out(c: AgentConfig) -> dict[str, Any]:
    return {
        "id": c.id,
        "name": c.name,
        "version": c.version,
        "config": c.config,
        "sha256": c.sha256,
        "note": c.note,
        "created_by": c.created_by,
        "created_at": c.created_at.isoformat(),
    }


def sess_out(s: CopilotSession) -> dict[str, Any]:
    return {
        "id": s.id,
        "question": s.question,
        "agent_config_id": s.agent_config_id,
        "planner": s.planner,
        "status": s.status,
        "report": s.report,
        "steps": s.steps,
        "tool_calls": s.tool_calls,
        "duration_ms": s.duration_ms,
        "trace_id": s.trace_id,
        "incident_id": s.incident_id,
        "error": s.error,
        "created_by": s.created_by,
        "created_at": s.created_at.isoformat(),
    }


@router.get("/copilot/tools")
def tools(_: Viewer) -> list[dict[str, Any]]:
    return [
        {"name": t.name, "description": t.description, "args": t.args, "permission": t.permission}
        for t in TOOLS.values()
    ]


@router.get("/copilot/configs")
def configs(_: Viewer, db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    agent.resolve_config(db, None)
    db.commit()
    return [
        cfg_out(c)
        for c in db.scalars(
            select(AgentConfig).order_by(AgentConfig.name, AgentConfig.version)
        ).all()
    ]


class ConfigIn(BaseModel):
    name: str = Field("copilot", pattern=r"^[a-z][a-z0-9_-]{1,63}$")
    config: dict[str, Any]
    note: str = Field("", max_length=2000)


@router.post("/copilot/configs", status_code=201)
def create_config(body: ConfigIn, p: Engineer, db: Session = Depends(get_db)) -> dict[str, Any]:
    unknown = set(body.config.get("tools_enabled", [])) - set(TOOLS)
    if unknown:
        raise HTTPException(422, f"Unknown tools {sorted(unknown)}")
    planner = str(body.config.get("planner", "rules"))
    if planner != "rules" and not planner.startswith("llm:"):
        raise HTTPException(422, "planner must be 'rules' or 'llm:<provider>:<model>'")
    merged = {**agent.DEFAULT_CONFIG, **body.config}
    n = (
        db.scalar(select(func.max(AgentConfig.version)).where(AgentConfig.name == body.name)) or 0
    ) + 1
    c = AgentConfig(
        name=body.name,
        version=n,
        config=merged,
        sha256=evals.sha(merged),
        note=body.note,
        created_by=p.username,
    )
    db.add(c)
    db.flush()
    db.add(
        AuditEvent(
            actor=p.username,
            action="agent_config.created",
            entity_type="agent_config",
            entity_id=c.id,
            data={"version": n, "note": body.note},
        )
    )
    db.commit()
    return cfg_out(c)


class AskIn(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    agent_config_id: str | None = None
    wait: bool = False


def _run_background(
    session_id: str, question: str, cfg_id: str | None, actor: str, role: str
) -> None:
    with session_scope() as db:
        agent.run_agent(db, question, cfg_id, actor, session_id=session_id, actor_role=role)


@router.post("/copilot/sessions", status_code=201)
def ask(body: AskIn, p: Viewer, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Read-only investigation; any recommended action still needs an approval."""
    try:
        if body.wait:
            res = agent.run_agent(
                db, body.question, body.agent_config_id, p.username, actor_role=p.role
            )
            return sess_out(db.get(CopilotSession, res["session_id"]))  # type: ignore[arg-type]
        cfg = agent.resolve_config(db, body.agent_config_id)
        s = CopilotSession(
            question=body.question,
            agent_config_id=cfg.id,
            planner=cfg.config.get("planner", "rules"),
            created_by=p.username,
        )
        db.add(s)
        db.commit()
    except ToolError as exc:
        raise HTTPException(404, str(exc)) from exc
    _pool.submit(_run_background, s.id, body.question, cfg.id, p.username, p.role)
    return sess_out(s)


@router.get("/copilot/sessions")
def sessions(_: Viewer, db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    return [
        sess_out(s)
        for s in db.scalars(
            select(CopilotSession).order_by(CopilotSession.created_at.desc()).limit(50)
        ).all()
    ]


@router.get("/copilot/sessions/{sid}")
def session(sid: str, _: Viewer, db: Session = Depends(get_db)) -> dict[str, Any]:
    s = db.get(CopilotSession, sid)
    if s is None:
        raise HTTPException(404, "Session not found")
    return sess_out(s)


@router.post("/copilot/sessions/{sid}/cancel")
def cancel(sid: str, p: Viewer, db: Session = Depends(get_db)) -> dict[str, str]:
    s = db.get(CopilotSession, sid)
    if s is None:
        raise HTTPException(404, "Session not found")
    if s.created_by != p.username and not p.has("engineer"):
        raise HTTPException(403, "Only the requester or an engineer can cancel")
    if s.status == "running":
        s.status = "cancelled"
        db.commit()
    return {"status": s.status}


@router.post("/incidents/{incident_id}/investigate", status_code=201)
def investigate(incident_id: str, p: Engineer, db: Session = Depends(get_db)) -> dict[str, Any]:
    try:
        res = agent.investigate_incident(db, incident_id, p.username, p.role)
    except ToolError as exc:
        raise HTTPException(404, str(exc)) from exc
    return sess_out(db.get(CopilotSession, res["session_id"]))  # type: ignore[arg-type]


@executor("evaluation")
def _exec_eval(db: Session, approval: Any, actor: Principal) -> dict[str, Any]:
    suite = db.get(EvalSuite, approval.payload["suite_id"])
    if suite is None:
        raise ToolError("Suite not found")
    run = evals.create_run(
        db,
        suite,
        approval.payload["subject"],
        approval.requested_by,
        approval.payload.get("baseline_run_id"),
    )
    return {"eval_run_id": run.id, "submit_job_id": run.job_id}
