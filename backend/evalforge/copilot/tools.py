"""Copilot tools: typed, permissioned functions over real EvalForge state.

Every tool returns JSON-safe data plus an ``evidence`` list of the identifiers it
read (run ids, dataset versions, MLflow run ids, eval runs, traces ...). The agent
may only cite identifiers that appear in some tool's evidence — that is how
unsupported conclusions are rejected.

``permission == "read"`` tools never change state. ``"action"`` tools only *file an
approval request*; a human approver decides, and the action runs on approval.
"""

from __future__ import annotations

import difflib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import mlflow
from sqlalchemy import select
from sqlalchemy.orm import Session

from evalforge.models import (
    AgentConfig,
    Dataset,
    DatasetVersion,
    Deployment,
    DriftReport,
    EvalRun,
    EvalSuite,
    Incident,
    JobStatus,
    ModelVersion,
    PipelineRun,
    Span,
    TrainingRun,
)
from evalforge.services import evals, tracking


class ToolError(ValueError):
    pass


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args: dict[str, str]  # arg -> description (all JSON scalars)
    permission: str  # read | action
    fn: Callable[..., dict[str, Any]]


def _r(x: float | None, n: int = 4) -> float | None:
    return None if x is None else round(float(x), n)


def _dataset(db: Session, name: str | None) -> Dataset:
    q = select(Dataset)
    if name:
        q = q.where(Dataset.name == name)
    ds = db.scalars(q.order_by(Dataset.created_at.desc())).first()
    if ds is None:
        raise ToolError(f"No dataset {name or ''} found".strip())
    return ds


# --------------------------------------------------------------------------- experiments


def query_mlflow_runs(db: Session, dataset: str | None = None, limit: int = 5) -> dict[str, Any]:
    """Recent finished runs read from MLflow (system of record), newest first."""
    ds = _dataset(db, dataset)
    tracking.configure()
    from evalforge.config import get_settings

    exp = mlflow.get_experiment_by_name(f"{get_settings().mlflow_experiment_prefix}/{ds.name}")
    if exp is None:
        return {"dataset": ds.name, "runs": [], "evidence": []}
    df = mlflow.search_runs(
        [exp.experiment_id], order_by=["start_time DESC"], max_results=int(limit)
    )
    runs = []
    for _, row in df.iterrows():
        runs.append(
            {
                "evalforge_run_id": row.get("tags.evalforge.run_id"),
                "mlflow_run_id": row["run_id"],
                "algorithm": row.get("params.algorithm"),
                "dataset_version": row.get("tags.evalforge.dataset_version"),
                "f1": _r(row.get("metrics.f1")),
                "roc_auc": _r(row.get("metrics.roc_auc")),
                "cv_f1_mean": _r(row.get("metrics.cv_f1_macro_mean")),
                "latency_p95_ms": _r(row.get("metrics.latency_single_p95_ms"), 3),
                "start_time": str(row["start_time"]),
            }
        )
    return {
        "dataset": ds.name,
        "experiment_id": exp.experiment_id,
        "runs": runs,
        "evidence": [r["evalforge_run_id"] for r in runs if r["evalforge_run_id"]]
        + [r["mlflow_run_id"] for r in runs],
    }


def get_run(db: Session, run_id: str) -> dict[str, Any]:
    r = db.get(TrainingRun, run_id)
    if r is None:
        raise ToolError(f"Run {run_id} not found")
    imp = (r.report or {}).get("feature_importance", [])[:5]
    return {
        "run_id": r.id,
        "status": r.status.value,
        "algorithm": r.algorithm,
        "hyperparameters": r.hyperparameters,
        "dataset": r.dataset_version.dataset.name,
        "dataset_version": r.dataset_version.version,
        "dataset_version_id": r.dataset_version_id,
        "metrics": r.metrics,
        "top_features": imp,
        "mlflow_run_id": r.mlflow_run_id,
        "git_commit": r.git_commit,
        "evidence": [r.id, r.dataset_version_id] + ([r.mlflow_run_id] if r.mlflow_run_id else []),
    }


def compare_runs(db: Session, run_a: str, run_b: str) -> dict[str, Any]:
    a, b = db.get(TrainingRun, run_a), db.get(TrainingRun, run_b)
    if a is None or b is None:
        raise ToolError("Run not found")
    keys = ["f1", "precision", "recall", "roc_auc", "cv_f1_macro_mean", "latency_single_p95_ms"]
    deltas = {
        k: _r((b.metrics or {}).get(k, 0) - (a.metrics or {}).get(k, 0))
        for k in keys
        if k in (a.metrics or {}) and k in (b.metrics or {})
    }
    params = sorted({*a.hyperparameters, *b.hyperparameters})
    changed = {
        k: [a.hyperparameters.get(k), b.hyperparameters.get(k)]
        for k in params
        if a.hyperparameters.get(k) != b.hyperparameters.get(k)
    }
    return {
        "a": {
            "run_id": a.id,
            "algorithm": a.algorithm,
            "dataset_version": a.dataset_version.version,
            "metrics": a.metrics,
        },
        "b": {
            "run_id": b.id,
            "algorithm": b.algorithm,
            "dataset_version": b.dataset_version.version,
            "metrics": b.metrics,
        },
        "metric_deltas_b_minus_a": deltas,
        "algorithm_changed": a.algorithm != b.algorithm,
        "hyperparameter_changes": changed,
        "same_dataset_version": a.dataset_version_id == b.dataset_version_id,
        "evidence": [a.id, b.id, a.dataset_version_id, b.dataset_version_id],
    }


def recommend_candidate(
    db: Session, dataset: str | None = None, limit: int = 10, max_latency_ms: float = 50.0
) -> dict[str, Any]:
    """Rank recent runs: Pareto-optimal on (CV F1 ↑, holdout F1 ↑, p95 latency ↓)."""
    ds = _dataset(db, dataset)
    runs = db.scalars(
        select(TrainingRun)
        .join(DatasetVersion)
        .where(DatasetVersion.dataset_id == ds.id, TrainingRun.status == JobStatus.succeeded)
        .order_by(TrainingRun.created_at.desc())
        .limit(limit)
    ).all()
    rows = []
    for r in runs:
        m = r.metrics or {}
        rows.append(
            {
                "run_id": r.id,
                "algorithm": r.algorithm,
                "dataset_version": r.dataset_version.version,
                "f1": m.get("f1"),
                "cv_f1": m.get("cv_f1_macro_mean"),
                "cv_std": m.get("cv_f1_macro_std"),
                "p95_ms": m.get("latency_single_p95_ms"),
            }
        )

    def dominated(x: dict, y: dict) -> bool:
        ks = [("f1", 1), ("cv_f1", 1), ("p95_ms", -1)]
        vals = [(x[k], y[k], s) for k, s in ks if x[k] is not None and y[k] is not None]
        return (
            bool(vals)
            and all(s * yv >= s * xv for xv, yv, s in vals)
            and any(s * yv > s * xv for xv, yv, s in vals)
        )

    pareto = [x for x in rows if not any(dominated(x, y) for y in rows if y is not x)]
    eligible = [x for x in pareto if (x["p95_ms"] or 0) <= max_latency_ms]
    best = max(
        eligible or pareto or rows,
        key=lambda x: ((x["cv_f1"] or x["f1"] or 0), -(x["p95_ms"] or 0)),
        default=None,
    )
    return {
        "dataset": ds.name,
        "considered": rows,
        "pareto_front": [x["run_id"] for x in pareto],
        "latency_budget_ms": max_latency_ms,
        "recommended": best,
        "evidence": [x["run_id"] for x in rows],
    }


# --------------------------------------------------------------------------- datasets


def list_dataset_versions(db: Session, dataset: str | None = None) -> dict[str, Any]:
    ds = _dataset(db, dataset)
    vs = [
        {
            "version": v.version,
            "id": v.id,
            "rows": v.n_rows,
            "status": v.status.value,
            "sha256": v.sha256[:12],
            "failed_checks": [c["id"] for c in v.quality_checks if not c["passed"]],
        }
        for v in ds.versions
    ]
    return {"dataset": ds.name, "versions": vs, "evidence": [v["id"] for v in vs]}


def diff_dataset_versions(
    db: Session, dataset: str | None = None, a: int | None = None, b: int | None = None
) -> dict[str, Any]:
    ds = _dataset(db, dataset)
    vs = {v.version: v for v in ds.versions}
    if len(vs) < 2 and (a is None or b is None):
        raise ToolError("Dataset has a single version")
    b = b or max(vs)
    a = a or max(v for v in vs if v < b)
    va, vb = vs.get(a), vs.get(b)
    if va is None or vb is None:
        raise ToolError("Version not found")
    pa, pb = va.profile["columns"], vb.profile["columns"]
    shifts = []
    for col in sorted(set(pa) & set(pb)):
        sa, sb = pa[col].get("stats"), pb[col].get("stats")
        entry: dict[str, Any] = {
            "column": col,
            "missing_ratio_change": _r(pb[col]["missing_ratio"] - pa[col]["missing_ratio"]),
        }
        if sa and sb and sa.get("std"):
            entry["mean_shift_in_std"] = _r((sb["mean"] - sa["mean"]) / sa["std"], 3)
        shifts.append(entry)
    shifts.sort(key=lambda e: -abs(e.get("mean_shift_in_std") or 0))
    ta = {
        c["label"]: c["count"] / va.n_rows for c in va.profile.get("target", {}).get("classes", [])
    }
    tb = {
        c["label"]: c["count"] / vb.n_rows for c in vb.profile.get("target", {}).get("classes", [])
    }
    warn_a = {c["id"] for c in va.quality_checks if not c["passed"]}
    warn_b = {c["id"] for c in vb.quality_checks if not c["passed"]}
    return {
        "dataset": ds.name,
        "a": a,
        "b": b,
        "rows": [va.n_rows, vb.n_rows],
        "added_columns": sorted(set(pb) - set(pa)),
        "largest_shifts": shifts[:8],
        "label_distribution": {
            "a": {k: _r(v) for k, v in ta.items()},
            "b": {k: _r(v) for k, v in tb.items()},
        },
        "new_quality_warnings": sorted(warn_b - warn_a),
        "evidence": [va.id, vb.id],
    }


# --------------------------------------------------------------------------- registry & ops


def get_model_versions(db: Session, model_name: str | None = None) -> dict[str, Any]:
    q = select(ModelVersion).order_by(ModelVersion.version.desc())
    if model_name:
        q = q.where(ModelVersion.name == model_name)
    mvs = db.scalars(q.limit(20)).all()
    return {
        "versions": [
            {
                "id": m.id,
                "name": m.name,
                "version": m.version,
                "status": m.status,
                "champion": m.is_champion,
                "run_id": m.run_id,
                "f1": (m.run.metrics or {}).get("f1"),
                "gate_passed": (m.gate or {}).get("passed"),
            }
            for m in mvs
        ],
        "evidence": [m.id for m in mvs] + [m.run_id for m in mvs],
    }


def get_deployments(db: Session, model_name: str | None = None, limit: int = 10) -> dict[str, Any]:
    q = select(Deployment).order_by(Deployment.created_at.desc()).limit(limit)
    if model_name:
        q = q.where(Deployment.model_name == model_name)
    ds = db.scalars(q).all()
    return {
        "deployments": [
            {
                "id": d.id,
                "model": d.model_name,
                "version": d.model_version.version,
                "environment": d.environment,
                "role": d.role,
                "traffic": d.traffic_percent,
                "status": d.status,
                "healthy": (d.health or {}).get("healthy"),
                "failed_checks": [
                    k for k, v in ((d.health or {}).get("checks") or {}).items() if not v
                ],
                "note": d.note,
                "created_at": d.created_at.isoformat(),
            }
            for d in ds
        ],
        "evidence": [d.id for d in ds],
    }


def get_drift_reports(db: Session, model_name: str | None = None, limit: int = 3) -> dict[str, Any]:
    q = select(DriftReport).order_by(DriftReport.created_at.desc()).limit(limit)
    if model_name:
        q = q.where(DriftReport.model_name == model_name)
    rs = db.scalars(q).all()
    return {
        "reports": [
            {
                "id": r.id,
                "model": r.model_name,
                "verdict": r.summary["verdict"],
                "drift_share": r.summary["drift_share"],
                "top_drifted": [
                    {"feature": f["feature"], "psi": f["psi"]} for f in r.features if f["drifted"]
                ][:5],
                "performance": r.performance
                and {
                    "baseline_f1": r.performance["baseline_f1"],
                    "current_f1": r.performance["current"]["f1"],
                    "labelled": r.performance["labelled_samples"],
                },
                "created_at": r.created_at.isoformat(),
            }
            for r in rs
        ],
        "evidence": [r.id for r in rs],
    }


def list_incidents(db: Session, status: str | None = None, limit: int = 5) -> dict[str, Any]:
    q = select(Incident).order_by(Incident.created_at.desc()).limit(limit)
    if status:
        q = q.where(Incident.status == status)
    inc = db.scalars(q).all()
    return {
        "incidents": [
            {
                "id": i.id,
                "kind": i.kind,
                "severity": i.severity,
                "status": i.status,
                "title": i.title,
                "details": i.details,
            }
            for i in inc
        ],
        "evidence": [i.id for i in inc],
    }


def get_pipeline_runs(db: Session, status: str | None = None, limit: int = 5) -> dict[str, Any]:
    q = select(PipelineRun).order_by(PipelineRun.created_at.desc()).limit(limit)
    if status:
        q = q.where(PipelineRun.status == status)
    runs = db.scalars(q).all()
    return {
        "runs": [
            {
                "id": r.id,
                "status": r.status,
                "error": r.error,
                "failed_step": next((s.name for s in r.steps if s.status == "failed"), None),
                "steps": {s.name: s.status for s in r.steps},
                "lineage": r.lineage,
            }
            for r in runs
        ],
        "evidence": [r.id for r in runs],
    }


def search_traces(
    db: Session, status: str = "error", name: str | None = None, limit: int = 10
) -> dict[str, Any]:
    from evalforge.observability import tracing

    tracing.flush()
    q = select(Span).order_by(Span.start_time.desc()).limit(limit)
    if status:
        q = q.where(Span.status == status)
    if name:
        q = q.where(Span.name.contains(name))
    spans = db.scalars(q).all()
    return {
        "spans": [
            {
                "trace_id": s.trace_id,
                "span_id": s.span_id,
                "name": s.name,
                "status": s.status,
                "message": s.status_message,
                "duration_ms": _r(s.duration_ms, 2),
            }
            for s in spans
        ],
        "evidence": sorted({s.trace_id for s in spans}),
    }


# --------------------------------------------------------------------------- evaluations


def list_eval_runs(db: Session, suite: str | None = None, limit: int = 5) -> dict[str, Any]:
    q = select(EvalRun).join(EvalSuite).order_by(EvalRun.created_at.desc()).limit(limit)
    if suite:
        q = q.where(EvalSuite.name == suite)
    rs = db.scalars(q).all()
    return {
        "runs": [
            {
                "id": r.id,
                "suite": f"{r.suite.name}@v{r.suite.version}",
                "subject": r.subject_key,
                "status": r.status,
                "pass_rate": (r.summary or {}).get("pass_rate"),
                "regression_detected": (r.regression or {}).get("detected"),
            }
            for r in rs
        ],
        "evidence": [r.id for r in rs],
    }


def compare_eval_runs(db: Session, baseline: str, current: str) -> dict[str, Any]:
    a, b = db.get(EvalRun, baseline), db.get(EvalRun, current)
    if a is None or b is None:
        raise ToolError("Evaluation run not found")
    cmp = evals.compare(a, b, a.suite.gates)
    failing = []
    for res in b.results:
        if res.passed:
            continue
        bad = {k: s.get("detail") for k, s in (res.scores or {}).items() if not s["passed"]}
        failing.append(
            {"case_id": res.case_id, "failed": bad, "error": res.error, "trace_id": res.trace_id}
        )
    return {
        "baseline": {"id": a.id, "subject": a.subject, "summary": a.summary},
        "current": {"id": b.id, "subject": b.subject, "summary": b.summary},
        "deltas": cmp["deltas"],
        "newly_failing_cases": cmp["newly_failing_cases"],
        "failing_case_details": failing[:15],
        "evidence": [a.id, b.id] + [f["trace_id"] for f in failing if f["trace_id"]],
    }


def diff_agent_configs(db: Session, a: str, b: str) -> dict[str, Any]:
    ca, cb = db.get(AgentConfig, a), db.get(AgentConfig, b)
    if ca is None or cb is None:
        raise ToolError("Agent config not found")
    ta, tb = set(ca.config.get("tools_enabled", [])), set(cb.config.get("tools_enabled", []))
    text = "\n".join(
        difflib.unified_diff(
            json.dumps(ca.config, indent=1, sort_keys=True).splitlines(),
            json.dumps(cb.config, indent=1, sort_keys=True).splitlines(),
            f"{ca.name}@v{ca.version}",
            f"{cb.name}@v{cb.version}",
            lineterm="",
        )
    )
    return {
        "a": f"{ca.name}@v{ca.version}",
        "b": f"{cb.name}@v{cb.version}",
        "tools_removed": sorted(ta - tb),
        "tools_added": sorted(tb - ta),
        "other_changes": {
            k: [ca.config.get(k), cb.config.get(k)]
            for k in sorted({*ca.config, *cb.config} - {"tools_enabled"})
            if ca.config.get(k) != cb.config.get(k)
        },
        "note_b": cb.note,
        "unified_diff": text[:4000],
        "evidence": [ca.id, cb.id],
    }


# --------------------------------------------------------------------------- actions


def request_evaluation(
    db: Session,
    suite_id: str,
    subject: str,
    requested_by: str = "copilot",
) -> dict[str, Any]:
    from evalforge.services.auth import Principal
    from evalforge.services.governance import request_approval

    s = db.get(EvalSuite, suite_id)
    if s is None:
        raise ToolError("Suite not found")
    subj = json.loads(subject)
    evals.subject_key(subj)
    a = request_approval(
        db,
        kind="evaluation",
        target_type="eval_suite",
        target_id=s.id,
        payload={"suite_id": s.id, "subject": subj},
        summary=f"Copilot requests evaluation of {evals.subject_key(subj)} on "
        f"{s.name}@v{s.version}",
        actor=Principal("copilot", requested_by, "engineer"),
    )
    return {"approval_id": a.id, "status": "pending_approval", "evidence": [a.id, s.id]}


TOOLS: dict[str, Tool] = {
    t.name: t
    for t in [
        Tool(
            "query_mlflow_runs",
            "Recent MLflow runs for a dataset with key metrics",
            {"dataset": "dataset name (optional)", "limit": "max runs"},
            "read",
            query_mlflow_runs,
        ),
        Tool(
            "get_run",
            "Metrics, params, top features and lineage of one training run",
            {"run_id": "EvalForge run id"},
            "read",
            get_run,
        ),
        Tool(
            "compare_runs",
            "Metric deltas and parameter changes between two runs (b - a)",
            {"run_a": "run id", "run_b": "run id"},
            "read",
            compare_runs,
        ),
        Tool(
            "recommend_candidate",
            "Pareto ranking of runs on quality, stability and latency",
            {"dataset": "dataset name", "limit": "runs", "max_latency_ms": "p95 budget"},
            "read",
            recommend_candidate,
        ),
        Tool(
            "list_dataset_versions",
            "Versions of a dataset with failed checks",
            {"dataset": "dataset name"},
            "read",
            list_dataset_versions,
        ),
        Tool(
            "diff_dataset_versions",
            "Distribution, schema, label and quality changes between two dataset versions",
            {"dataset": "name", "a": "older version", "b": "newer version"},
            "read",
            diff_dataset_versions,
        ),
        Tool(
            "get_model_versions",
            "Registered model versions, champion and gate results",
            {"model_name": "registered model"},
            "read",
            get_model_versions,
        ),
        Tool(
            "get_deployments",
            "Deployment history with health checks and rollbacks",
            {"model_name": "model", "limit": "rows"},
            "read",
            get_deployments,
        ),
        Tool(
            "get_drift_reports",
            "Latest drift reports with verdicts and drifted features",
            {"model_name": "model", "limit": "rows"},
            "read",
            get_drift_reports,
        ),
        Tool(
            "list_incidents",
            "Open or recent incidents",
            {"status": "filter", "limit": "rows"},
            "read",
            list_incidents,
        ),
        Tool(
            "get_pipeline_runs",
            "Pipeline runs with failed steps and lineage",
            {"status": "filter", "limit": "rows"},
            "read",
            get_pipeline_runs,
        ),
        Tool(
            "search_traces",
            "Error spans from distributed traces",
            {"status": "error|ok", "name": "span name contains", "limit": "rows"},
            "read",
            search_traces,
        ),
        Tool(
            "list_eval_runs",
            "Recent LLM/agent evaluation runs",
            {"suite": "suite name", "limit": "rows"},
            "read",
            list_eval_runs,
        ),
        Tool(
            "compare_eval_runs",
            "Regression analysis between two evaluation runs",
            {"baseline": "eval run id", "current": "eval run id"},
            "read",
            compare_eval_runs,
        ),
        Tool(
            "diff_agent_configs",
            "Differences between two agent configuration versions",
            {"a": "agent config id", "b": "agent config id"},
            "read",
            diff_agent_configs,
        ),
        Tool(
            "request_evaluation",
            "File an approval request to run an evaluation suite",
            {"suite_id": "suite id", "subject": "subject JSON"},
            "action",
            request_evaluation,
        ),
    ]
}

ALL_TOOL_NAMES = sorted(TOOLS)
