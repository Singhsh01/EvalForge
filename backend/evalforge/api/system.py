"""Health, platform metadata, overview aggregates, audit log and demo bootstrap."""

from __future__ import annotations

import io
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from evalforge import __version__
from evalforge.config import get_settings
from evalforge.db import get_db
from evalforge.ml.algorithms import catalogue
from evalforge.models import AuditEvent, Dataset, DatasetVersion, Job, JobStatus, TrainingRun
from evalforge.services import datasets as dsvc
from evalforge.services import tracking
from evalforge.services.auth import require

router = APIRouter(prefix="/api", tags=["system"])


def _iso(dt: datetime) -> str:
    return (dt if dt.tzinfo else dt.replace(tzinfo=UTC)).isoformat()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@router.get("/health/ready")
def ready(db: Session = Depends(get_db)) -> dict[str, Any]:
    checks: dict[str, str] = {}
    try:
        db.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:
        checks["database"] = f"error: {exc}"
    try:
        tracking.ping()
        checks["mlflow"] = "ok"
    except Exception as exc:
        checks["mlflow"] = f"error: {exc}"
    ok = all(v == "ok" for v in checks.values())
    if not ok:
        raise HTTPException(503, {"status": "degraded", "checks": checks})
    return {"status": "ready", "checks": checks}


@router.get("/meta")
def meta() -> dict[str, Any]:
    s = get_settings()
    return {
        "version": __version__,
        "executor": s.executor,
        "database": s.resolved_database_url.split(":")[0],
        "mlflow_tracking_uri": s.resolved_mlflow_uri if s.mlflow_tracking_uri else "embedded",
        "algorithms": catalogue(),
        "limits": {
            "max_upload_mb": s.max_upload_mb,
            "max_rows": s.max_rows,
            "max_columns": s.max_columns,
        },
    }


def _ops_counts(db: Session) -> dict[str, int]:
    from evalforge.models import (
        Approval,
        CopilotSession,
        Deployment,
        DriftReport,
        EvalRun,
        Incident,
        ModelVersion,
        PipelineRun,
    )

    def n(model: Any, *where: Any) -> int:
        return db.scalar(select(func.count()).select_from(model).where(*where)) or 0

    return {
        "model_versions": n(ModelVersion),
        "champions": n(ModelVersion, ModelVersion.is_champion.is_(True)),
        "active_deployments": n(Deployment, Deployment.status == "active"),
        "production_deployments": n(
            Deployment, Deployment.status == "active", Deployment.environment == "production"
        ),
        "rollbacks": n(Deployment, Deployment.status == "rolled_back"),
        "pipeline_runs": n(PipelineRun),
        "drift_reports": n(DriftReport),
        "incidents_open": n(Incident, Incident.status != "resolved"),
        "incident_retrains": n(PipelineRun, PipelineRun.trigger == "incident"),
        "eval_runs": n(EvalRun),
        "pending_approvals": n(Approval, Approval.status == "pending"),
        "copilot_sessions": n(CopilotSession),
    }


@router.get("/overview", dependencies=[Depends(require("viewer"))])
def overview(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Aggregates computed from persisted records only — empty when nothing has run."""
    status_counts = dict(
        db.execute(select(TrainingRun.status, func.count()).group_by(TrainingRun.status)).all()
    )
    recent = db.scalars(select(TrainingRun).order_by(TrainingRun.created_at.desc()).limit(8)).all()
    best_by_dataset: list[dict[str, Any]] = []
    for ds in db.scalars(select(Dataset)).all():
        runs = db.scalars(
            select(TrainingRun)
            .join(DatasetVersion)
            .where(DatasetVersion.dataset_id == ds.id, TrainingRun.status == JobStatus.succeeded)
        ).all()
        if runs:
            b = max(runs, key=lambda r: (r.metrics or {}).get("f1", -1))
            best_by_dataset.append(
                {
                    "dataset": ds.name,
                    "run_id": b.id,
                    "algorithm": b.algorithm,
                    "dataset_version": b.dataset_version.version,
                    "f1": (b.metrics or {}).get("f1"),
                    "roc_auc": (b.metrics or {}).get("roc_auc"),
                    "n_runs": len(runs),
                }
            )
    succeeded = [
        r
        for r in db.scalars(
            select(TrainingRun).where(TrainingRun.status == JobStatus.succeeded)
        ).all()
        if r.training_seconds is not None
    ]
    return {
        "counts": {
            "datasets": db.scalar(select(func.count()).select_from(Dataset)) or 0,
            "dataset_versions": db.scalar(select(func.count()).select_from(DatasetVersion)) or 0,
            "runs": sum(status_counts.values()),
            "runs_by_status": {
                k.value if hasattr(k, "value") else k: v for k, v in status_counts.items()
            },
            **_ops_counts(db),
            "active_jobs": db.scalar(
                select(func.count())
                .select_from(Job)
                .where(Job.status.in_([JobStatus.queued, JobStatus.running]))
            )
            or 0,
        },
        "total_training_seconds": round(sum(r.training_seconds or 0 for r in succeeded), 3),
        "best_by_dataset": best_by_dataset,
        "recent_runs": [
            {
                "id": r.id,
                "algorithm": r.algorithm,
                "status": r.status.value,
                "dataset": r.dataset_version.dataset.name,
                "dataset_version": r.dataset_version.version,
                "f1": (r.metrics or {}).get("f1"),
                "created_at": _iso(r.created_at),
            }
            for r in recent
        ],
        "recent_activity": [
            {
                "action": e.action,
                "entity_type": e.entity_type,
                "entity_id": e.entity_id,
                "data": e.data,
                "created_at": _iso(e.created_at),
            }
            for e in db.scalars(
                select(AuditEvent).order_by(AuditEvent.created_at.desc()).limit(12)
            ).all()
        ],
    }


@router.post("/demo/breast-cancer", status_code=201, dependencies=[Depends(require("engineer"))])
def load_demo(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Register the UCI Breast Cancer Wisconsin (Diagnostic) dataset bundled with scikit-learn.

    Public dataset (569 rows, 30 numeric features; CC BY 4.0). Loaded from the
    scikit-learn package, so no network access is needed.
    """
    from evalforge.demo.data import breast_cancer_csv

    name = "breast-cancer-wisconsin"
    existing = db.scalar(select(Dataset).where(Dataset.name == name))
    if existing:
        return {"dataset_id": existing.id, "created": False}
    data = breast_cancer_csv()
    ds = dsvc.create_dataset(
        db,
        name=name,
        target_column="diagnosis",
        description="UCI Breast Cancer Wisconsin (Diagnostic): 569 tumour samples, 30 numeric "
        "features computed from digitised FNA images; label malignant/benign.",
        data=data,
        filename="breast_cancer.csv",
    )
    db.commit()
    return {"dataset_id": ds.id, "created": True, "bytes": len(io.BytesIO(data).getvalue())}
