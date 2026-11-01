"""Prompt registry, evaluation suites, evaluation runs, judge calibration."""

from __future__ import annotations

import difflib
from typing import Annotated, Any

import jsonschema
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from evalforge.db import get_db
from evalforge.llm import providers
from evalforge.llm.evaluators import EVALUATORS, judge_calibration
from evalforge.models import AuditEvent, EvalRun, EvalSuite, Prompt, PromptVersion
from evalforge.services import evals, jobs
from evalforge.services.auth import Principal, require
from evalforge.services.governance import GovernanceError, executor, request_approval

router = APIRouter(prefix="/api", tags=["llmops"])
Viewer = Annotated[Principal, Depends(require("viewer"))]
Engineer = Annotated[Principal, Depends(require("engineer"))]
NAME = r"^[a-z0-9][a-z0-9_.-]{1,63}$"


@router.get("/llm/providers")
def list_providers(_: Viewer) -> dict[str, Any]:
    return {"providers": providers.available(), "evaluators": sorted(EVALUATORS)}


# --------------------------------------------------------------------------- prompts


def pv_out(v: PromptVersion) -> dict[str, Any]:
    return {
        "id": v.id,
        "prompt": v.prompt.name,
        "version": v.version,
        "system": v.system,
        "template": v.template,
        "variables": v.variables,
        "output_schema": v.output_schema,
        "params": v.params,
        "sha256": v.sha256,
        "status": v.status,
        "note": v.note,
        "created_by": v.created_by,
        "created_at": v.created_at.isoformat(),
    }


class PromptVersionIn(BaseModel):
    system: str = Field("", max_length=20000)
    template: str = Field(min_length=1, max_length=50000)
    output_schema: dict[str, Any] | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    note: str = Field("", max_length=2000)


class PromptIn(PromptVersionIn):
    name: str = Field(pattern=NAME)
    description: str = Field("", max_length=2000)


def _add_version(db: Session, prompt: Prompt, body: PromptVersionIn, actor: str) -> PromptVersion:
    if body.output_schema is not None:
        try:
            jsonschema.Draft202012Validator.check_schema(body.output_schema)
        except jsonschema.SchemaError as exc:
            raise HTTPException(422, f"Invalid output_schema: {exc.message}") from exc
    content = {
        "system": body.system,
        "template": body.template,
        "output_schema": body.output_schema,
        "params": body.params,
    }
    digest = evals.sha(content)
    if any(v.sha256 == digest for v in prompt.versions):
        raise HTTPException(409, "An identical version already exists")
    n = (
        db.scalar(
            select(func.max(PromptVersion.version)).where(PromptVersion.prompt_id == prompt.id)
        )
        or 0
    ) + 1
    v = PromptVersion(
        prompt_id=prompt.id,
        version=n,
        system=body.system,
        template=body.template,
        variables=evals.template_variables(body.template),
        output_schema=body.output_schema,
        params=body.params,
        sha256=digest,
        note=body.note,
        created_by=actor,
    )
    db.add(v)
    db.flush()
    db.refresh(v)
    db.add(
        AuditEvent(
            actor=actor,
            action="prompt.version_created",
            entity_type="prompt_version",
            entity_id=v.id,
            data={"prompt": prompt.name, "version": n},
        )
    )
    return v


@router.get("/prompts")
def list_prompts(_: Viewer, db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    return [
        {
            "id": p.id,
            "name": p.name,
            "description": p.description,
            "versions": [pv_out(v) for v in p.versions],
        }
        for p in db.scalars(select(Prompt).order_by(Prompt.name)).all()
    ]


@router.post("/prompts", status_code=201)
def create_prompt(body: PromptIn, p: Engineer, db: Session = Depends(get_db)) -> dict[str, Any]:
    if db.scalar(select(Prompt).where(Prompt.name == body.name)):
        raise HTTPException(409, "Prompt exists; add a version instead")
    prompt = Prompt(name=body.name, description=body.description)
    db.add(prompt)
    db.flush()
    v = _add_version(db, prompt, body, p.username)
    db.commit()
    return pv_out(v)


@router.post("/prompts/{name}/versions", status_code=201)
def add_version(
    name: str, body: PromptVersionIn, p: Engineer, db: Session = Depends(get_db)
) -> dict[str, Any]:
    prompt = db.scalar(select(Prompt).where(Prompt.name == name))
    if prompt is None:
        raise HTTPException(404, "Prompt not found")
    v = _add_version(db, prompt, body, p.username)
    db.commit()
    return pv_out(v)


@router.get("/prompts/{name}/diff")
def diff(name: str, a: int, b: int, _: Viewer, db: Session = Depends(get_db)) -> dict[str, Any]:
    prompt = db.scalar(select(Prompt).where(Prompt.name == name))
    if prompt is None:
        raise HTTPException(404, "Prompt not found")
    va = next((v for v in prompt.versions if v.version == a), None)
    vb = next((v for v in prompt.versions if v.version == b), None)
    if va is None or vb is None:
        raise HTTPException(404, "Version not found")
    d = difflib.unified_diff(
        (va.system + "\n---\n" + va.template).splitlines(),
        (vb.system + "\n---\n" + vb.template).splitlines(),
        f"v{a}",
        f"v{b}",
        lineterm="",
    )
    return {"diff": "\n".join(d), "variables": {"a": va.variables, "b": vb.variables}}


class ReleaseIn(BaseModel):
    eval_run_id: str


@router.post("/prompts/versions/{pv_id}/request-release", status_code=201)
def request_release(
    pv_id: str, body: ReleaseIn, p: Engineer, db: Session = Depends(get_db)
) -> dict[str, Any]:
    v = db.get(PromptVersion, pv_id)
    run = db.get(EvalRun, body.eval_run_id)
    if v is None or run is None:
        raise HTTPException(404, "Prompt version or evaluation run not found")
    if run.subject.get("prompt_version_id") != v.id or run.status != "succeeded":
        raise HTTPException(409, "The evaluation run must be a finished run of this version")
    if not (run.summary or {}).get("gates_passed"):
        raise HTTPException(409, "The evaluation run did not pass its suite's gates")
    if (run.regression or {}).get("detected"):
        raise HTTPException(409, "The evaluation run shows a regression against its baseline")
    try:
        a = request_approval(
            db,
            kind="prompt_release",
            target_type="prompt_version",
            target_id=v.id,
            payload={"eval_run_id": run.id},
            summary=f"Release prompt {v.prompt.name} v{v.version} "
            f"(pass rate {run.summary['pass_rate']:.0%})",
            actor=p,
        )
    except GovernanceError as exc:
        raise HTTPException(409, str(exc)) from exc
    db.commit()
    return {"approval_id": a.id}


@executor("prompt_release")
def _release(db: Session, approval: Any, actor: Principal) -> dict[str, Any]:
    v = db.get(PromptVersion, approval.target_id)
    assert v is not None
    for other in v.prompt.versions:
        if other.status == "released":
            other.status = "superseded"
    v.status = "released"
    return {"released": f"{v.prompt.name} v{v.version}"}


# --------------------------------------------------------------------------- suites


class SuiteIn(BaseModel):
    name: str = Field(pattern=NAME)
    kind: str = Field(pattern="^(prompt|rag|agent)$")
    description: str = Field("", max_length=4000)
    cases: list[dict[str, Any]] = Field(min_length=1, max_length=2000)
    evaluators: list[dict[str, Any]] = Field(min_length=1, max_length=20)
    gates: dict[str, float] = Field(default_factory=dict)
    corpus: list[dict[str, Any]] | None = Field(None, max_length=20000)


def suite_out(s: EvalSuite, full: bool = False) -> dict[str, Any]:
    out = {
        "id": s.id,
        "name": s.name,
        "version": s.version,
        "kind": s.kind,
        "description": s.description,
        "n_cases": len(s.cases),
        "evaluators": s.evaluators,
        "gates": s.gates,
        "sha256": s.sha256,
        "corpus_size": len(s.corpus or []),
        "created_by": s.created_by,
        "created_at": s.created_at.isoformat(),
    }
    if full:
        out |= {"cases": s.cases, "corpus": s.corpus}
    return out


@router.post("/evals/suites", status_code=201)
def create_suite(body: SuiteIn, p: Engineer, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Creates a new immutable version of a suite (versions are numbered per name)."""
    try:
        evals.validate_suite(body.kind, body.cases, body.evaluators, body.corpus)
    except evals.EvalError as exc:
        raise HTTPException(422, str(exc)) from exc
    content = body.model_dump(exclude={"name", "description"})
    digest = evals.sha(content)
    prior = db.scalars(select(EvalSuite).where(EvalSuite.name == body.name)).all()
    if any(s.sha256 == digest for s in prior):
        raise HTTPException(409, "An identical suite version already exists")
    if prior and prior[0].kind != body.kind:
        raise HTTPException(422, "A suite's kind cannot change between versions")
    s = EvalSuite(
        name=body.name,
        version=len(prior) + 1,
        kind=body.kind,
        description=body.description,
        cases=body.cases,
        evaluators=body.evaluators,
        gates=body.gates,
        corpus=body.corpus,
        sha256=digest,
        created_by=p.username,
    )
    db.add(s)
    db.flush()
    db.add(
        AuditEvent(
            actor=p.username,
            action="eval_suite.created",
            entity_type="eval_suite",
            entity_id=s.id,
            data={"name": s.name, "version": s.version},
        )
    )
    db.commit()
    return suite_out(s)


@router.get("/evals/suites")
def list_suites(_: Viewer, db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    return [
        suite_out(s)
        for s in db.scalars(
            select(EvalSuite).order_by(EvalSuite.name, EvalSuite.version.desc())
        ).all()
    ]


@router.get("/evals/suites/{suite_id}")
def get_suite(suite_id: str, _: Viewer, db: Session = Depends(get_db)) -> dict[str, Any]:
    s = db.get(EvalSuite, suite_id)
    if s is None:
        raise HTTPException(404, "Suite not found")
    return suite_out(s, full=True)


# --------------------------------------------------------------------------- runs


class EvalRunIn(BaseModel):
    suite_id: str
    subject: dict[str, Any]
    baseline_run_id: str | None = None


def run_out(r: EvalRun, results: bool = False) -> dict[str, Any]:
    out = {
        "id": r.id,
        "suite_id": r.suite_id,
        "suite": f"{r.suite.name}@v{r.suite.version}",
        "kind": r.suite.kind,
        "subject": r.subject,
        "subject_key": r.subject_key,
        "status": r.status,
        "summary": r.summary,
        "baseline_run_id": r.baseline_run_id,
        "regression": r.regression,
        "trace_id": r.trace_id,
        "error": r.error,
        "created_by": r.created_by,
        "created_at": r.created_at.isoformat(),
        "finished_at": r.finished_at.isoformat() if r.finished_at else None,
    }
    if results:
        out["results"] = [
            {
                "case_id": c.case_id,
                "passed": c.passed,
                "scores": c.scores,
                "output": c.output,
                "latency_ms": round(c.latency_ms, 2),
                "input_tokens": c.input_tokens,
                "output_tokens": c.output_tokens,
                "cost_usd": c.cost_usd,
                "error": c.error,
                "trace_id": c.trace_id,
            }
            for c in sorted(r.results, key=lambda c: c.case_id)
        ]
    return out


@router.post("/evals/runs", status_code=202)
def launch(body: EvalRunIn, p: Engineer, db: Session = Depends(get_db)) -> dict[str, Any]:
    suite = db.get(EvalSuite, body.suite_id)
    if suite is None:
        raise HTTPException(404, "Suite not found")
    try:
        run = evals.create_run(db, suite, body.subject, p.username, body.baseline_run_id)
    except (evals.EvalError, KeyError) as exc:
        raise HTTPException(422, f"Invalid subject: {exc}") from exc
    db.commit()
    jobs.submit(run.job_id or "")
    db.refresh(run)
    return run_out(run)


@router.get("/evals/runs")
def list_runs(
    _: Viewer, suite_id: str | None = None, db: Session = Depends(get_db)
) -> list[dict[str, Any]]:
    q = select(EvalRun).order_by(EvalRun.created_at.desc()).limit(200)
    if suite_id:
        q = q.where(EvalRun.suite_id == suite_id)
    return [run_out(r) for r in db.scalars(q).all()]


@router.get("/evals/runs/{run_id}")
def get_run(run_id: str, _: Viewer, db: Session = Depends(get_db)) -> dict[str, Any]:
    r = db.get(EvalRun, run_id)
    if r is None:
        raise HTTPException(404, "Run not found")
    return run_out(r, results=True)


@router.get("/evals/compare")
def compare(a: str, b: str, _: Viewer, db: Session = Depends(get_db)) -> dict[str, Any]:
    ra, rb = db.get(EvalRun, a), db.get(EvalRun, b)
    if ra is None or rb is None:
        raise HTTPException(404, "Run not found")
    if ra.suite_id != rb.suite_id:
        raise HTTPException(422, "Runs must use the same suite version to be comparable")
    cmp = evals.compare(ra, rb, ra.suite.gates)
    per_case = []
    by_a = {c.case_id: c for c in ra.results}
    for c in sorted(rb.results, key=lambda c: c.case_id):
        o = by_a.get(c.case_id)
        per_case.append(
            {
                "case_id": c.case_id,
                "a_passed": o.passed if o else None,
                "b_passed": c.passed,
                "a_scores": {k: v["score"] for k, v in (o.scores or {}).items()} if o else {},
                "b_scores": {k: v["score"] for k, v in (c.scores or {}).items()},
            }
        )
    return {"a": run_out(ra), "b": run_out(rb), "comparison": cmp, "cases": per_case}


@router.get("/evals/runs/{run_id}/judge-calibration")
def calibration(run_id: str, _: Viewer, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Compare LLM-judge rubric scores with human scores stored on the suite's cases."""
    r = db.get(EvalRun, run_id)
    if r is None:
        raise HTTPException(404, "Run not found")
    human = {c["id"]: c.get("human_scores") for c in r.suite.cases if c.get("human_scores")}
    pairs = []
    for res in r.results:
        for sc in (res.scores or {}).values():
            js = (sc.get("detail") or {}).get("scores")
            if js and res.case_id in human:
                pairs.append((js, human[res.case_id]))
    if not pairs:
        raise HTTPException(422, "No cases with both judge scores and human_scores")
    return {"pairs": len(pairs), "criteria": judge_calibration(pairs)}
