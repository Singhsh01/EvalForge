"""Model registry on top of MLflow, with quality gates and promotion via aliases.

MLflow registered-model *aliases* are used for lifecycle state (``champion``);
stage transitions are deprecated in MLflow and are not used anywhere.
"""

from __future__ import annotations

from typing import Any

import mlflow
import pandas as pd
from mlflow import MlflowClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from evalforge.config import get_settings
from evalforge.ml import metrics as M
from evalforge.models import (
    Approval,
    AuditEvent,
    DatasetVersion,
    JobStatus,
    ModelVersion,
    TrainingRun,
)
from evalforge.services import tracking
from evalforge.services.auth import Principal
from evalforge.services.governance import GovernanceError, executor
from evalforge.services.model_cache import load_cached
from evalforge.services.profiling import load_split
from evalforge.services.storage import load_frame

CHAMPION = "champion"


def model_name_for(run: TrainingRun) -> str:
    return run.dataset_version.dataset.name


def register(db: Session, run: TrainingRun, actor: Principal) -> ModelVersion:
    if run.status != JobStatus.succeeded or not run.model_uri:
        raise GovernanceError("Only successfully trained runs can be registered")
    existing = db.scalar(select(ModelVersion).where(ModelVersion.run_id == run.id))
    if existing is not None:
        raise GovernanceError(f"Run is already registered as {existing.name} v{existing.version}")
    tracking.configure()
    name = model_name_for(run)
    mv = mlflow.register_model(run.model_uri, name, await_registration_for=60)
    client = MlflowClient()
    for k, v in {
        "evalforge.run_id": run.id,
        "evalforge.dataset_version_id": run.dataset_version_id,
        "evalforge.dataset_sha256": run.dataset_version.sha256,
        "evalforge.algorithm": run.algorithm,
    }.items():
        client.set_model_version_tag(name, mv.version, k, v)
    row = ModelVersion(
        name=name,
        version=int(mv.version),
        run_id=run.id,
        source_uri=run.model_uri,
        # Digest pinned when training wrote the artifact; legacy runs are digested now.
        artifact_sha256=run.model_sha256 or tracking.artifact_digest(run.model_uri),
        created_by=actor.username,
    )
    db.add(row)
    db.flush()
    db.add(
        AuditEvent(
            actor=actor.username,
            action="model.registered",
            entity_type="model_version",
            entity_id=row.id,
            data={"name": name, "version": row.version, "run_id": run.id},
        )
    )
    return row


def champion(db: Session, name: str) -> ModelVersion | None:
    return db.scalar(
        select(ModelVersion).where(ModelVersion.name == name, ModelVersion.is_champion.is_(True))
    )


def holdout_frame(version: DatasetVersion) -> tuple[pd.DataFrame, pd.Series]:
    df = load_frame(version.storage_path)
    assert version.split is not None
    _, test_idx = load_split(version.split["path"])
    target = version.dataset.target_column
    return df.loc[test_idx], df.loc[test_idx, target].astype(str)


def score_on(mv: ModelVersion, version: DatasetVersion) -> dict[str, float]:
    """Score a registered model on another version's holdout rows (comparable numbers)."""
    feats = mv.run.feature_columns or []
    X, y = holdout_frame(version)
    missing = [c for c in feats if c not in X.columns]
    if missing:
        raise GovernanceError(
            f"Version lacks columns required by {mv.name} v{mv.version}: {missing}"
        )
    model = load_cached(mv.source_uri, mv.artifact_sha256)
    labels = sorted(set((mv.run.report or {}).get("labels", [])) | set(y.unique()))
    metrics, _ = M.classification_metrics(model, X[feats], y, labels)
    return metrics


def evaluate_gate(db: Session, mv: ModelVersion) -> dict[str, Any]:
    s = get_settings()
    m = mv.run.metrics or {}
    checks: list[dict[str, Any]] = []

    def check(cid: str, ok: bool, detail: str, blocking: bool = True) -> None:
        checks.append({"id": cid, "passed": ok, "blocking": blocking, "detail": detail})

    f1 = m.get("f1", 0.0)
    check("min_f1", f1 >= s.gate_min_f1, f"holdout F1 {f1:.4f} (minimum {s.gate_min_f1})")
    p95 = m.get("latency_single_p95_ms", 0.0)
    check(
        "latency_p95",
        p95 <= s.gate_max_p95_latency_ms,
        f"single-row p95 {p95:.1f} ms (maximum {s.gate_max_p95_latency_ms} ms)",
    )
    gap = m.get("generalization_gap_f1", 0.0)
    check(
        "overfitting",
        gap <= 0.15,
        f"train–test F1 gap {gap:.3f} (warning above 0.15)",
        blocking=False,
    )
    leak = next(
        (c for c in mv.run.dataset_version.quality_checks if c["id"] == "target_leakage"), None
    )
    if leak is not None:
        check("no_leakage_warning", leak["passed"], leak["message"], blocking=False)

    champ = champion(db, mv.name)
    comparison = None
    if champ is not None and champ.id != mv.id:
        champ_metrics = score_on(champ, mv.run.dataset_version)
        delta = f1 - champ_metrics["f1"]
        comparison = {
            "champion_id": champ.id,
            "champion_version": champ.version,
            "evaluated_on_dataset_version": mv.run.dataset_version.version,
            "champion_f1": round(champ_metrics["f1"], 6),
            "candidate_f1": round(f1, 6),
            "delta_f1": round(delta, 6),
            "champion_metrics": champ_metrics,
        }
        check(
            "no_regression_vs_champion",
            delta >= -s.gate_max_f1_regression,
            f"candidate F1 {f1:.4f} vs champion v{champ.version} {champ_metrics['f1']:.4f} "
            f"on the same holdout (Δ {delta:+.4f}, tolerance {s.gate_max_f1_regression})",
        )
    passed = all(c["passed"] for c in checks if c["blocking"])
    return {"passed": passed, "checks": checks, "champion_comparison": comparison}


def next_version_number(db: Session, name: str) -> int:
    return (
        db.scalar(select(func.max(ModelVersion.version)).where(ModelVersion.name == name)) or 0
    ) + 1


@executor("model_promotion")
def _promote(db: Session, approval: Approval, actor: Principal) -> dict[str, Any]:
    mv = db.get(ModelVersion, approval.target_id)
    if mv is None:
        raise GovernanceError("Model version no longer exists")
    gate = evaluate_gate(db, mv)  # re-check at decision time; the world may have changed
    mv.gate = gate
    if not gate["passed"]:
        raise GovernanceError("Quality gate no longer passes; promotion aborted")
    previous = champion(db, mv.name)
    if previous is not None:
        previous.is_champion = False
    mv.is_champion, mv.status = True, "approved"
    tracking.configure()
    MlflowClient().set_registered_model_alias(mv.name, CHAMPION, str(mv.version))
    db.add(
        AuditEvent(
            actor=actor.username,
            action="model.promoted",
            entity_type="model_version",
            entity_id=mv.id,
            data={
                "name": mv.name,
                "version": mv.version,
                "previous_champion": previous.version if previous else None,
            },
        )
    )
    if approval.payload.get("pipeline_run_id"):
        from evalforge.services.pipelines import on_promotion_executed

        on_promotion_executed(db, approval.payload["pipeline_run_id"])
    return {
        "champion": f"{mv.name}@{CHAMPION} -> v{mv.version}",
        "previous_champion_version": previous.version if previous else None,
    }


def model_card(mv: ModelVersion) -> dict[str, Any]:
    run, dv = mv.run, mv.run.dataset_version
    m = run.metrics or {}
    rep = run.report or {}
    warnings = [
        c["message"]
        for c in dv.quality_checks
        if not c["passed"] and c["severity"] in ("warning", "error")
    ]
    return {
        "model": f"{mv.name} v{mv.version}",
        "status": "champion" if mv.is_champion else mv.status,
        "algorithm": run.algorithm,
        "hyperparameters": run.hyperparameters,
        "intended_use": f"Classify '{dv.dataset.target_column}' for records shaped like dataset "
        f"'{dv.dataset.name}'. Decision support only; not validated for "
        "autonomous decisions.",
        "training_data": {
            "dataset": dv.dataset.name,
            "version": dv.version,
            "sha256": dv.sha256,
            "rows": dv.n_rows,
            "split": dv.split,
            "source_file": dv.source_filename,
        },
        "evaluation": {
            "holdout_rows": rep.get("n_samples"),
            "positive_label": rep.get("positive_label"),
            "metrics": m,
        },
        "features": {"used": run.feature_columns, "dropped": run.dropped_columns},
        "lineage": {
            "mlflow_run_id": run.mlflow_run_id,
            "model_uri": run.model_uri,
            "git_commit": run.git_commit,
            "environment": run.environment,
        },
        "quality_gate": mv.gate,
        "limitations": [
            f"Evaluated on {rep.get('n_samples')} holdout rows; small samples make metric "
            "differences of ~0.02 statistically fragile.",
            "Performance outside the training distribution is unknown; monitor drift.",
            *warnings,
        ],
    }
