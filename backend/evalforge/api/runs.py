from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from evalforge.db import get_db
from evalforge.ml.algorithms import HyperparameterError, get_algorithm
from evalforge.models import (
    AuditEvent,
    DatasetVersion,
    Evaluation,
    Job,
    JobKind,
    JobStatus,
    TrainingRun,
    VersionStatus,
)
from evalforge.schemas import (
    CompareOut,
    EvaluateRequest,
    EvaluationOut,
    JobOut,
    LaunchResponse,
    RunDetail,
    RunOut,
    TrainRequest,
)
from evalforge.services import jobs

router = APIRouter(prefix="/api", tags=["experiments"])

# Direction of "better" for comparison; metrics not listed are informational.
HIGHER_IS_BETTER = {
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "f1_weighted",
    "roc_auc",
    "roc_auc_ovr",
    "cv_f1_macro_mean",
}
LOWER_IS_BETTER = {
    "log_loss",
    "latency_single_p50_ms",
    "latency_single_p95_ms",
    "latency_batch_ms_per_row",
    "training_seconds",
}


def run_out(r: TrainingRun, cls: type[RunOut] = RunOut) -> RunOut:
    out = cls.model_validate(r)
    return out.model_copy(
        update={
            "dataset_name": r.dataset_version.dataset.name,
            "dataset_version_number": r.dataset_version.version,
        }
    )


@router.post("/runs", response_model=LaunchResponse, status_code=202)
def launch_run(req: TrainRequest, db: Session = Depends(get_db)) -> LaunchResponse:
    version = db.get(DatasetVersion, req.dataset_version_id)
    if version is None:
        raise HTTPException(404, "Dataset version not found")
    if version.status != VersionStatus.validated:
        raise HTTPException(
            422,
            "Dataset version failed blocking quality checks; fix the data and upload a new version",
        )
    try:
        algo = get_algorithm(req.algorithm)
        hp = algo.validate(req.hyperparameters)
    except HyperparameterError as exc:
        raise HTTPException(422, str(exc)) from exc

    job = Job(kind=JobKind.train, payload=req.model_dump())
    db.add(job)
    db.flush()
    run = TrainingRun(
        job_id=job.id,
        dataset_version_id=version.id,
        algorithm=algo.key,
        hyperparameters=hp,
        seed=req.seed,
        cv_folds=req.cv_folds,
    )
    db.add(run)
    db.flush()
    db.add(
        AuditEvent(
            action="run.launched",
            entity_type="training_run",
            entity_id=run.id,
            data={"algorithm": algo.key, "hyperparameters": hp, "dataset_version_id": version.id},
        )
    )
    db.commit()
    jobs.submit(job.id)
    db.refresh(job)
    db.refresh(run)
    return LaunchResponse(job=JobOut.model_validate(job), run=run_out(run))


@router.get("/runs", response_model=list[RunOut])
def list_runs(
    dataset_id: str | None = None,
    status: JobStatus | None = None,
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
) -> list[RunOut]:
    q = (
        select(TrainingRun)
        .join(TrainingRun.dataset_version)
        .options(joinedload(TrainingRun.dataset_version).joinedload(DatasetVersion.dataset))
        .order_by(TrainingRun.created_at.desc())
        .limit(limit)
    )
    if dataset_id:
        q = q.where(DatasetVersion.dataset_id == dataset_id)
    if status:
        q = q.where(TrainingRun.status == status)
    return [run_out(r) for r in db.scalars(q).unique().all()]


@router.get("/runs/compare", response_model=CompareOut)
def compare_runs(
    ids: str = Query(..., description="Comma-separated run ids (2–8)"),
    db: Session = Depends(get_db),
) -> CompareOut:
    id_list = [i for i in ids.split(",") if i]
    if not 2 <= len(id_list) <= 8:
        raise HTTPException(422, "Provide between 2 and 8 run ids")
    runs = [db.get(TrainingRun, i) for i in id_list]
    if any(r is None for r in runs):
        raise HTTPException(404, "One or more runs not found")
    runs_ok = [r for r in runs if r is not None]

    warnings: list[str] = []
    versions = {r.dataset_version_id for r in runs_ok}
    same_holdout = len(versions) == 1
    if not same_holdout:
        warnings.append(
            "Runs were evaluated on different dataset versions/holdouts — metric "
            "differences are not directly comparable."
        )
    unfinished = [r.id[:8] for r in runs_ok if r.status != JobStatus.succeeded]
    if unfinished:
        warnings.append(f"Runs without results: {', '.join(unfinished)}")

    metric_keys = sorted({k for r in runs_ok for k in (r.metrics or {})})
    params = [
        {**r.hyperparameters, "algorithm": r.algorithm, "seed": r.seed, "cv_folds": r.cv_folds}
        for r in runs_ok
    ]
    param_keys = sorted({k for p in params for k in p})
    differing = [k for k in param_keys if len({repr(p.get(k)) for p in params}) > 1]

    best: dict[str, list[str]] = {}
    for k in metric_keys:
        vals = [(r.metrics or {}).get(k) for r in runs_ok]
        scored = [(v, r.id) for v, r in zip(vals, runs_ok, strict=True) if v is not None]
        if not scored or k not in HIGHER_IS_BETTER | LOWER_IS_BETTER:
            continue
        target = max(v for v, _ in scored) if k in HIGHER_IS_BETTER else min(v for v, _ in scored)
        best[k] = [rid for v, rid in scored if v == target]  # ties keep every winner
    return CompareOut(
        runs=[run_out(r) for r in runs_ok],
        metric_keys=metric_keys,
        param_keys=param_keys,
        differing_params=differing,
        same_holdout=same_holdout,
        best=best,
        warnings=warnings,
    )


@router.get("/runs/{run_id}", response_model=RunDetail)
def get_run(run_id: str, db: Session = Depends(get_db)) -> RunDetail:
    r = db.get(TrainingRun, run_id)
    if r is None:
        raise HTTPException(404, "Run not found")
    out = run_out(r, RunDetail)
    return out.model_copy(update={"job": JobOut.model_validate(r.job)})  # type: ignore[return-value]


@router.post("/runs/{run_id}/evaluations", response_model=LaunchResponse, status_code=202)
def launch_evaluation(
    run_id: str, req: EvaluateRequest, db: Session = Depends(get_db)
) -> LaunchResponse:
    run = db.get(TrainingRun, run_id)
    if run is None:
        raise HTTPException(404, "Run not found")
    if run.status != JobStatus.succeeded or not run.model_uri:
        raise HTTPException(422, "Run has no trained model")
    version = db.get(DatasetVersion, req.dataset_version_id)
    if version is None:
        raise HTTPException(404, "Dataset version not found")
    if version.dataset_id != run.dataset_version.dataset_id:
        raise HTTPException(422, "Evaluation dataset must be a version of the run's dataset")
    job = Job(kind=JobKind.evaluate, payload={"run_id": run_id, **req.model_dump()})
    db.add(job)
    db.flush()
    ev = Evaluation(run_id=run_id, dataset_version_id=version.id, job_id=job.id, split="full")
    db.add(ev)
    db.commit()
    jobs.submit(job.id)
    db.refresh(job)
    db.refresh(ev)
    return LaunchResponse(
        job=JobOut.model_validate(job), evaluation=EvaluationOut.model_validate(ev)
    )


@router.get("/runs/{run_id}/evaluations", response_model=list[EvaluationOut])
def list_evaluations(run_id: str, db: Session = Depends(get_db)) -> list[EvaluationOut]:
    rows = db.scalars(
        select(Evaluation).where(Evaluation.run_id == run_id).order_by(Evaluation.created_at)
    ).all()
    return [EvaluationOut.model_validate(e) for e in rows]


@router.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: str, db: Session = Depends(get_db)) -> JobOut:
    j = db.get(Job, job_id)
    if j is None:
        raise HTTPException(404, "Job not found")
    return JobOut.model_validate(j)
