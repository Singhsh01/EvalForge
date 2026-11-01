"""Training and evaluation job handlers.

These functions run inside a worker (thread, RQ process or inline in tests). They
open their own DB sessions, report progress through :class:`JobReporter`, and
always leave the job and the run in a terminal state.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from opentelemetry import trace
from sklearn.model_selection import StratifiedKFold, cross_val_score

from evalforge.db import session_scope
from evalforge.ml import metrics as M
from evalforge.ml.algorithms import build_pipeline, get_algorithm
from evalforge.models import (
    AuditEvent,
    DatasetVersion,
    Evaluation,
    Job,
    JobStatus,
    TrainingRun,
)
from evalforge.observability import tracing
from evalforge.services import tracking
from evalforge.services.profiling import load_split
from evalforge.services.storage import load_frame


class JobReporter:
    def __init__(self, job_id: str) -> None:
        self.job_id = job_id

    def update(self, stage: str, progress: float, message: str | None = None) -> None:
        trace.get_current_span().add_event(
            f"stage.{stage}", {"progress": progress, "message": message or ""}
        )
        with session_scope() as db:
            job = db.get(Job, self.job_id)
            assert job is not None
            job.stage = stage
            job.progress = round(progress, 3)
            if message:
                job.logs = [
                    *(job.logs or []),
                    {"ts": datetime.now(UTC).isoformat(), "stage": stage, "message": message},
                ]


def select_features(version: DatasetVersion, target: str) -> tuple[list[str], list[dict[str, str]]]:
    """All columns except target, identifiers and constants — with reasons recorded."""
    by_id = {c["id"]: c for c in version.quality_checks}
    drop: dict[str, str] = {}
    for c in by_id.get("identifier_columns", {}).get("details", {}).get("columns", []):
        drop[c] = "identifier-like"
    for c in by_id.get("constant_columns", {}).get("details", {}).get("columns", []):
        drop[c] = "constant"
    feats = [c["name"] for c in version.schema_ if c["name"] != target and c["name"] not in drop]
    return feats, [{"column": k, "reason": v} for k, v in drop.items()]


def run_training(job_id: str) -> dict[str, Any]:
    rep = JobReporter(job_id)
    with session_scope() as db:
        run = db.query(TrainingRun).filter_by(job_id=job_id).one()
        version = run.dataset_version
        dataset = version.dataset
        run_id, algo_key, hp, seed, cv_folds = (
            run.id,
            run.algorithm,
            run.hyperparameters,
            run.seed,
            run.cv_folds,
        )
        target, ds_name = dataset.target_column, dataset.name
        storage_path, split, sha = version.storage_path, version.split, version.sha256
        version_no, version_id = version.version, version.id
        feats, dropped = select_features(version, target)
        run.status = JobStatus.running
        run.feature_columns, run.dropped_columns = feats, dropped

    rep.update("loading_data", 0.05, f"Loading {ds_name} v{version_no}")
    df = load_frame(storage_path)
    assert split is not None
    train_idx, test_idx = load_split(split["path"])
    X_train, y_train = df.loc[train_idx, feats], df.loc[train_idx, target].astype(str)
    X_test, y_test = df.loc[test_idx, feats], df.loc[test_idx, target].astype(str)
    labels = sorted(pd.concat([y_train, y_test]).unique().tolist())

    algo = get_algorithm(algo_key)
    pipe = build_pipeline(algo, hp, X_train, seed)

    cv: dict[str, float] = {}
    if cv_folds and cv_folds >= 2:
        rep.update("cross_validation", 0.15, f"{cv_folds}-fold stratified CV on training split")
        scores = cross_val_score(
            pipe,
            X_train,
            y_train,
            cv=StratifiedKFold(cv_folds, shuffle=True, random_state=seed),
            scoring="f1_macro",
            n_jobs=1,
        )
        cv = {"cv_f1_macro_mean": float(scores.mean()), "cv_f1_macro_std": float(scores.std())}

    rep.update("training", 0.35, f"Fitting {algo.label} on {len(X_train)} rows")
    t0 = time.perf_counter()
    with tracing.span("model.fit", {"algorithm": algo_key, "rows": len(X_train)}):
        pipe.fit(X_train, y_train)
    train_s = time.perf_counter() - t0
    # Parallelism helps fitting but adds thread-dispatch overhead to every predict call
    # (measured ~15x slower single-row latency for RandomForest with n_jobs=2).
    if hasattr(pipe.named_steps["model"], "n_jobs"):
        pipe.named_steps["model"].set_params(n_jobs=1)

    rep.update("evaluating", 0.6, f"Evaluating on {len(X_test)} holdout rows")
    test_metrics, report = M.classification_metrics(pipe, X_test, y_test, labels)
    train_metrics, _ = M.classification_metrics(pipe, X_train, y_train, labels)
    latency = M.inference_latency(pipe, X_test)
    rep.update("diagnostics", 0.72, "Permutation feature importance")
    importance = M.feature_importance(pipe, X_test, y_test, seed)

    metrics = {
        **test_metrics,
        **{f"train_{k}": v for k, v in train_metrics.items() if k in ("accuracy", "f1")},
        **{k: round(v, 6) for k, v in cv.items()},
        **latency,
        "training_seconds": round(train_s, 4),
        "generalization_gap_f1": round(train_metrics["f1"] - test_metrics["f1"], 6),
    }
    report = {**report, "feature_importance": importance}
    git = tracking.git_info()
    env = tracking.environment_info()

    rep.update("tracking", 0.85, "Logging run and model to MLflow")
    exp_id = tracking.experiment_for(ds_name)
    log_span = tracing.tracer().start_span("mlflow.log_run", attributes={"experiment_id": exp_id})
    trace_id = tracing.current_trace_id() or ""
    mlflow_run_id, model_uri = tracking.log_training_run(
        experiment_id=exp_id,
        run_name=f"{algo_key}-{run_id[:8]}",
        params={
            **hp,
            "algorithm": algo_key,
            "seed": seed,
            "cv_folds": cv_folds,
            "n_features": len(feats),
            "n_train": len(X_train),
            "n_test": len(X_test),
        },
        metrics=metrics,
        tags={
            "evalforge.run_id": run_id,
            "evalforge.dataset": ds_name,
            "evalforge.dataset_version": str(version_no),
            "evalforge.dataset_version_id": version_id,
            "evalforge.dataset_sha256": sha,
            "evalforge.trace_id": trace_id,
            "evalforge.split_seed": str(split["seed"]),
            "mlflow.source.git.commit": str(git["commit"]),
            "evalforge.git_dirty": str(git["dirty"]),
            **{f"env.{k}": v for k, v in env.items()},
        },
        model=pipe,
        X_example=X_test,
        train_df=pd.concat([X_train, y_train], axis=1),
        dataset_name=f"{ds_name}@v{version_no}",
        dataset_digest=sha,
        dataset_source=storage_path,
        target=target,
        artifacts={
            "classification_report.json": report,
            "features.json": {"used": feats, "dropped": dropped},
            "split.json": split,
        },
    )
    log_span.set_attribute("mlflow.run_id", mlflow_run_id)
    log_span.end()

    with session_scope() as db:
        run = db.get(TrainingRun, run_id)
        assert run is not None
        run.metrics, run.report = metrics, report
        run.environment = {**env, "git": git}
        run.mlflow_experiment_id, run.mlflow_run_id, run.model_uri = (
            exp_id,
            mlflow_run_id,
            model_uri,
        )
        run.training_seconds, run.git_commit = train_s, str(git["commit"])
        run.status = JobStatus.succeeded
        run.finished_at = datetime.now(UTC)
        db.add(
            Evaluation(
                run_id=run_id,
                dataset_version_id=version_id,
                split="holdout",
                status=JobStatus.succeeded,
                metrics=test_metrics,
                report=report,
            )
        )
        db.add(
            AuditEvent(
                action="run.completed",
                entity_type="training_run",
                entity_id=run_id,
                data={"mlflow_run_id": mlflow_run_id, "f1": metrics["f1"]},
            )
        )
    rep.update("done", 1.0, f"F1={metrics['f1']:.4f} · MLflow run {mlflow_run_id[:8]}")
    return {"run_id": run_id, "mlflow_run_id": mlflow_run_id, "metrics": metrics}


def run_evaluation(job_id: str) -> dict[str, Any]:
    """Score an existing run's logged model on all labelled rows of another version."""
    rep = JobReporter(job_id)
    with session_scope() as db:
        ev = db.query(Evaluation).filter_by(job_id=job_id).one()
        run = db.get(TrainingRun, ev.run_id)
        version = db.get(DatasetVersion, ev.dataset_version_id)
        assert run is not None and version is not None and run.model_uri
        ev.status = JobStatus.running
        ev_id, model_uri, feats = ev.id, run.model_uri, run.feature_columns or []
        target, storage_path = version.dataset.target_column, version.storage_path
        train_labels = (run.report or {}).get("labels", [])

    rep.update("loading", 0.2, "Loading model from MLflow and dataset")
    model = tracking.load_model(model_uri)
    df = load_frame(storage_path)
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing feature columns required by the model: {missing}")
    df = df[df[target].notna()]
    X, y = df[feats], df[target].astype(str)
    labels = sorted(set(train_labels) | set(y.unique()))
    rep.update("scoring", 0.6, f"Scoring {len(X)} rows")
    metrics, report = M.classification_metrics(model, X, y, labels)
    unseen = sorted(set(y.unique()) - set(train_labels))
    report["unseen_labels"] = unseen
    with session_scope() as db:
        ev = db.get(Evaluation, ev_id)
        assert ev is not None
        ev.metrics, ev.report, ev.status = metrics, report, JobStatus.succeeded
    rep.update("done", 1.0, f"F1={metrics['f1']:.4f}")
    return {"evaluation_id": ev_id, "metrics": metrics}


def mark_failed(job_id: str, error: str) -> None:
    from evalforge.models import EvalRun, PipelineRun

    with session_scope() as db:
        run = db.query(TrainingRun).filter_by(job_id=job_id).one_or_none()
        if run is not None:
            run.status, run.error, run.finished_at = JobStatus.failed, error, datetime.now(UTC)
        ev = db.query(Evaluation).filter_by(job_id=job_id).one_or_none()
        if ev is not None:
            ev.status, ev.error = JobStatus.failed, error
        pr = db.query(PipelineRun).filter_by(job_id=job_id).one_or_none()
        if pr is not None and pr.status not in ("succeeded", "awaiting_approval"):
            pr.status, pr.error, pr.finished_at = "failed", error, datetime.now(UTC)
        er = db.query(EvalRun).filter_by(job_id=job_id).one_or_none()
        if er is not None:
            er.status, er.error, er.finished_at = "failed", error, datetime.now(UTC)
