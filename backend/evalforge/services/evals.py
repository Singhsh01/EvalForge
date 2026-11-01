"""Evaluation engine: run a versioned suite against a subject, score, gate, detect regressions.

Subjects:

* ``{"type": "prompt", "prompt_version_id", "model"}`` — render a registered prompt
  version with each case's ``inputs`` and call the model.
* ``{"type": "rag", "top_k", "model"?, "prompt_version_id"?}`` — TF-IDF retrieval over
  the suite's corpus, optionally followed by grounded generation. Without a model the
  run is retrieval-only (recall/precision/MRR are still real measurements).
* ``{"type": "agent", "agent_config_id"}`` — run the EvalForge Copilot agent with a
  specific versioned configuration on each case's question.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import traceback
from datetime import UTC, datetime
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy import select
from sqlalchemy.orm import Session

from evalforge.db import session_scope
from evalforge.llm.evaluators import EVALUATORS
from evalforge.llm.providers import complete
from evalforge.models import (
    AuditEvent,
    EvalCaseResult,
    EvalRun,
    EvalSuite,
    Incident,
    IncidentEvent,
    Job,
    JobKind,
    LLMCall,
    PromptVersion,
)
from evalforge.observability import metrics, tracing

VAR = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")


class EvalError(ValueError):
    pass


def sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def template_variables(template: str) -> list[str]:
    return sorted(set(VAR.findall(template)))


def render(template: str, values: dict[str, Any]) -> str:
    """Plain ``{{var}}`` substitution — no expression evaluation, so templates are safe."""
    missing = [v for v in template_variables(template) if v not in values]
    if missing:
        raise EvalError(f"Missing template variables: {missing}")
    return VAR.sub(lambda m: str(values[m.group(1)]), template)


def validate_suite(
    kind: str, cases: list[dict], evaluators: list[dict], corpus: list[dict] | None
) -> None:
    if kind not in ("prompt", "rag", "agent"):
        raise EvalError("kind must be prompt, rag or agent")
    if not cases:
        raise EvalError("A suite needs at least one case")
    ids = [c.get("id") for c in cases]
    if any(not i for i in ids) or len(set(ids)) != len(ids):
        raise EvalError("Every case needs a unique 'id'")
    for e in evaluators:
        if e.get("type") not in EVALUATORS:
            raise EvalError(f"Unknown evaluator {e.get('type')}. Available: {sorted(EVALUATORS)}")
        if e["type"] == "llm_judge" and not e.get("model"):
            raise EvalError("llm_judge needs a 'model' reference")
    if kind == "rag" and not corpus:
        raise EvalError("RAG suites need a corpus of {id, text} passages")


def subject_key(subject: dict[str, Any]) -> str:
    t = subject.get("type")
    if t == "prompt":
        return f"prompt:{subject['prompt_version_id']}@{subject['model']}"
    if t == "rag":
        return f"rag:k{subject.get('top_k', 3)}@{subject.get('model') or 'retrieval-only'}"
    if t == "agent":
        return f"agent:{subject['agent_config_id']}"
    raise EvalError("subject.type must be prompt, rag or agent")


def create_run(
    db: Session,
    suite: EvalSuite,
    subject: dict[str, Any],
    actor: str,
    baseline_run_id: str | None = None,
) -> EvalRun:
    key = subject_key(subject)
    if suite.kind == "prompt" and subject["type"] != "prompt":
        raise EvalError("Prompt suites need a prompt subject")
    if suite.kind == "agent" and subject["type"] != "agent":
        raise EvalError("Agent suites need an agent subject")
    if suite.kind == "rag" and subject["type"] != "rag":
        raise EvalError("RAG suites need a rag subject")
    job = Job(kind=JobKind.eval_suite, payload={"suite_id": suite.id, "subject": subject})
    db.add(job)
    db.flush()
    run = EvalRun(
        suite_id=suite.id,
        subject=subject,
        subject_key=key,
        job_id=job.id,
        created_by=actor,
        baseline_run_id=baseline_run_id,
    )
    db.add(run)
    db.flush()
    db.add(
        AuditEvent(
            actor=actor,
            action="eval.launched",
            entity_type="eval_run",
            entity_id=run.id,
            data={"suite": f"{suite.name}@v{suite.version}", "subject": key},
        )
    )
    return run


# --------------------------------------------------------------------------- subjects


class Retriever:
    def __init__(self, corpus: list[dict[str, Any]]) -> None:
        self.ids = [str(d["id"]) for d in corpus]
        self.texts = {str(d["id"]): str(d["text"]) for d in corpus}
        self.vec = TfidfVectorizer(stop_words="english", sublinear_tf=True)
        self.matrix = self.vec.fit_transform([self.texts[i] for i in self.ids])

    def search(self, query: str, k: int) -> list[str]:
        sims = (self.matrix @ self.vec.transform([query]).T).toarray().ravel()
        order = np.argsort(-sims, kind="stable")[:k]
        return [self.ids[i] for i in order if sims[i] > 0]


def _llm(
    db: Session,
    ref: str,
    system: str,
    user: str,
    purpose: str,
    pv_id: str | None,
    params: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    with tracing.span("llm.complete", {"llm.model": ref, "llm.purpose": purpose}) as sp:
        try:
            r = complete(
                ref,
                system,
                user,
                max_tokens=int(params.get("max_tokens", 1024)),
                temperature=float(params.get("temperature", 0.0)),
            )
        except Exception as exc:
            db.add(
                LLMCall(
                    provider=ref.split(":")[0],
                    model=ref.split(":", 1)[-1],
                    purpose=purpose,
                    prompt_version_id=pv_id,
                    error=str(exc)[:500],
                    trace_id=tracing.current_trace_id(),
                )
            )
            raise
        sp.set_attributes(
            {"llm.input_tokens": r.input_tokens, "llm.output_tokens": r.output_tokens}
        )
    metrics.LLM_TOKENS.labels(r.model, "input").inc(r.input_tokens)
    metrics.LLM_TOKENS.labels(r.model, "output").inc(r.output_tokens)
    db.add(
        LLMCall(
            provider=r.provider,
            model=r.model,
            purpose=purpose,
            prompt_version_id=pv_id,
            input_tokens=r.input_tokens,
            output_tokens=r.output_tokens,
            cost_usd=r.cost_usd or 0.0,
            latency_ms=r.latency_ms,
            trace_id=tracing.current_trace_id(),
        )
    )
    return r.text, {
        "input_tokens": r.input_tokens,
        "output_tokens": r.output_tokens,
        "cost_usd": r.cost_usd,
        "latency_ms": r.latency_ms,
    }


def run_subject(
    db: Session,
    suite: EvalSuite,
    subject: dict[str, Any],
    case: dict[str, Any],
    retriever: Retriever | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    t = subject["type"]
    if t == "prompt":
        pv = db.get(PromptVersion, subject["prompt_version_id"])
        if pv is None:
            raise EvalError("Prompt version not found")
        user = render(pv.template, case.get("inputs") or {})
        text, usage = _llm(db, subject["model"], pv.system, user, "eval", pv.id, pv.params)
        return {"text": text}, usage
    if t == "rag":
        assert retriever is not None
        k = int(subject.get("top_k", 3))
        ids = retriever.search(case["question"], k)
        contexts = {i: retriever.texts[i] for i in ids}
        out: dict[str, Any] = {"retrieved_ids": ids, "contexts": contexts, "text": ""}
        usage: dict[str, Any] = {}
        if subject.get("model"):
            pv = (
                db.get(PromptVersion, subject["prompt_version_id"])
                if subject.get("prompt_version_id")
                else None
            )
            ctx = "\n".join(f"[{i}] {contexts[i]}" for i in ids)
            if pv is not None:
                user = render(pv.template, {"question": case["question"], "context": ctx})
                system, params = pv.system, pv.params
            else:
                system = (
                    "Answer only from the context. Cite passages as [id] after each "
                    "sentence. If the context is insufficient, say so."
                )
                user, params = f"Context:\n{ctx}\n\nQuestion: {case['question']}", {}
            out["text"], usage = _llm(
                db, subject["model"], system, user, "eval", pv.id if pv else None, params
            )
        return out, usage
    if t == "agent":
        from evalforge.copilot.agent import run_agent

        res = run_agent(
            db, case["question"], subject["agent_config_id"], actor="evaluator", persist=False
        )
        if res["status"] == "failed":
            # The planner itself crashed (e.g. model endpoint unreachable). Tool failures no
            # longer fail a session, so this is an infrastructure error, not an agent
            # decision, and must not be scored as "used no tools" (audit BEN-04).
            raise EvalError(f"agent session failed: {res.get('error')}")
        return {
            "text": res["report"].get("answer", ""),
            "status": res["status"],
            "tool_calls": res["tool_calls"],
            "steps": len(res["tool_calls"]),
        }, {"latency_ms": res["duration_ms"], **(res.get("usage") or {})}
    raise EvalError(f"Unknown subject type {t}")


# --------------------------------------------------------------------------- execution


def execute_run(job_id: str) -> dict[str, Any]:
    with session_scope() as db:
        run = db.scalars(select(EvalRun).where(EvalRun.job_id == job_id)).one()
        run.status = "running"
        run_id, suite_id = run.id, run.suite_id
    with tracing.span("eval.run", {"eval.run_id": run_id}) as root:
        trace_id = tracing.current_trace_id()
        with session_scope() as db:
            run = db.get(EvalRun, run_id)
            suite = db.get(EvalSuite, suite_id)
            assert run is not None and suite is not None
            run.trace_id = trace_id
            root.set_attributes(
                {"eval.suite": f"{suite.name}@v{suite.version}", "eval.subject": run.subject_key}
            )
            retriever = Retriever(suite.corpus) if suite.kind == "rag" and suite.corpus else None
            # Commit before any case calls a model: a pending UPDATE would otherwise hold
            # SQLite's write lock for the whole (possibly minutes-long) first case and make
            # every other write in the platform time out (audit RES-05).
            db.commit()
            for case in suite.cases:
                _run_case(db, run, suite, case, retriever)
                db.commit()
            summary = summarize(run, suite)
            run.summary = summary
            base = baseline_for(db, run)
            if base is not None:
                run.baseline_run_id = base.id
                run.regression = compare(base, run, suite.gates)
                if run.regression["detected"]:
                    open_regression_incident(db, suite, base, run)
            run.status, run.finished_at = "succeeded", datetime.now(UTC)
            root.set_attributes({"eval.pass_rate": summary["pass_rate"]})
            return {"eval_run_id": run_id, "summary": summary}


def _run_case(
    db: Session, run: EvalRun, suite: EvalSuite, case: dict[str, Any], retriever: Retriever | None
) -> None:
    res = EvalCaseResult(run_id=run.id, case_id=case["id"])
    with tracing.span("eval.case", {"eval.case_id": case["id"]}):
        res.trace_id = tracing.current_trace_id()
        t0 = time.perf_counter()
        try:
            output, usage = run_subject(db, suite, run.subject, case, retriever)
        except Exception as exc:
            res.error = f"{type(exc).__name__}: {exc}"
            res.output = {"traceback": traceback.format_exc()[-2000:]}
            res.latency_ms = (time.perf_counter() - t0) * 1000
            db.add(res)
            return
        res.latency_ms = usage.get("latency_ms") or (time.perf_counter() - t0) * 1000
        res.input_tokens = usage.get("input_tokens", 0)
        res.output_tokens = usage.get("output_tokens", 0)
        res.cost_usd = usage.get("cost_usd") or 0.0
        res.output = output
        scores: dict[str, Any] = {}
        for ev in suite.evaluators:
            name = ev.get("name") or ev["type"]
            try:
                sc = EVALUATORS[ev["type"]](case, output, ev)
            except Exception as exc:
                sc = {"score": 0.0, "passed": False, "detail": {"error": str(exc)[:300]}}
            if "usage" in sc:
                u = sc.pop("usage")
                db.add(
                    LLMCall(
                        provider=u["provider"],
                        model=u["model"],
                        purpose="judge",
                        input_tokens=u["input_tokens"],
                        output_tokens=u["output_tokens"],
                        cost_usd=u["cost_usd"] or 0.0,
                        latency_ms=u["latency_ms"],
                        trace_id=res.trace_id,
                    )
                )
            scores[name] = sc
        res.scores = scores
        res.passed = all(s["passed"] for s in scores.values()) if scores else True
        db.add(res)


def summarize(run: EvalRun, suite: EvalSuite) -> dict[str, Any]:
    rs = list(run.results)
    n = len(rs)
    lat = [r.latency_ms for r in rs]
    names = [e.get("name") or e["type"] for e in suite.evaluators]
    means = {
        k: round(float(np.mean([r.scores[k]["score"] for r in rs if k in (r.scores or {})])), 4)
        for k in names
        if any(k in (r.scores or {}) for r in rs)
    }
    pass_rate = round(sum(r.passed and not r.error for r in rs) / n, 4) if n else 0.0
    gates = {}
    for metric, minimum in (suite.gates or {}).items():
        if metric == "regression_tolerance":
            continue
        value = pass_rate if metric == "pass_rate" else means.get(metric)
        gates[metric] = {
            "min": minimum,
            "value": value,
            "passed": value is not None and value >= minimum,
        }
    return {
        "cases": n,
        "passed": sum(r.passed and not r.error for r in rs),
        "errors": sum(bool(r.error) for r in rs),
        "pass_rate": pass_rate,
        "evaluator_means": means,
        "latency_ms": {
            "p50": round(float(np.percentile(lat, 50)), 2) if lat else None,
            "p95": round(float(np.percentile(lat, 95)), 2) if lat else None,
        },
        "input_tokens": sum(r.input_tokens for r in rs),
        "output_tokens": sum(r.output_tokens for r in rs),
        "cost_usd": round(sum(r.cost_usd for r in rs), 6),
        "gates": gates,
        "gates_passed": all(g["passed"] for g in gates.values()),
    }


def baseline_for(db: Session, run: EvalRun) -> EvalRun | None:
    if run.baseline_run_id:
        return db.get(EvalRun, run.baseline_run_id)
    return db.scalars(
        select(EvalRun)
        .where(
            EvalRun.suite_id == run.suite_id, EvalRun.status == "succeeded", EvalRun.id != run.id
        )
        .order_by(EvalRun.created_at.desc())
    ).first()


def compare(base: EvalRun, cur: EvalRun, gates: dict[str, Any]) -> dict[str, Any]:
    tol = float((gates or {}).get("regression_tolerance", 0.05))
    bs, cs = base.summary or {}, cur.summary or {}
    deltas = {"pass_rate": round(cs.get("pass_rate", 0) - bs.get("pass_rate", 0), 4)}
    for k, v in (cs.get("evaluator_means") or {}).items():
        if k in (bs.get("evaluator_means") or {}):
            deltas[k] = round(v - bs["evaluator_means"][k], 4)
    before = {r.case_id: r for r in base.results}
    flips = []
    for r in cur.results:
        b = before.get(r.case_id)
        if b is not None and b.passed and not r.passed:
            failed = [k for k, s in (r.scores or {}).items() if not s["passed"]]
            flips.append({"case_id": r.case_id, "failed_evaluators": failed, "error": r.error})
    regressed = {k: d for k, d in deltas.items() if d < -tol}
    return {
        "baseline_run_id": base.id,
        "baseline_subject": base.subject_key,
        "current_subject": cur.subject_key,
        "tolerance": tol,
        "deltas": deltas,
        "regressed_metrics": regressed,
        "newly_failing_cases": flips,
        "detected": bool(regressed),
    }


def open_regression_incident(db: Session, suite: EvalSuite, base: EvalRun, run: EvalRun) -> None:
    reg = run.regression or {}
    drop = -reg["deltas"].get("pass_rate", 0)
    inc = Incident(
        kind=f"{suite.kind}_regression",
        severity="high" if drop >= 0.2 else "medium",
        title=f"{suite.name}: pass rate {base.summary['pass_rate']:.0%} → "
        f"{run.summary['pass_rate']:.0%} ({run.subject_key})",
        source_type="eval_run",
        source_id=run.id,
        details={
            "suite_id": suite.id,
            "suite": f"{suite.name}@v{suite.version}",
            "baseline_run_id": base.id,
            "current_run_id": run.id,
            "baseline_subject": base.subject,
            "current_subject": run.subject,
            "regression": reg,
        },
    )
    inc.events.append(
        IncidentEvent(
            kind="detected",
            message=(
                f"Regression detected by eval run {run.id[:8]} vs baseline {base.id[:8]}: "
                f"{', '.join(f'{k} {v:+.2f}' for k, v in reg['regressed_metrics'].items())}; "
                f"{len(reg['newly_failing_cases'])} case(s) newly failing."
            ),
            data={"regressed_metrics": reg["regressed_metrics"]},
        )
    )
    db.add(inc)
    db.flush()
    db.add(
        AuditEvent(
            action="incident.opened",
            entity_type="incident",
            entity_id=inc.id,
            data={"kind": inc.kind, "source": run.id},
        )
    )
