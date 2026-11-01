"""Creation of training runs, shared by the HTTP API, pipelines and the Copilot."""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from evalforge.ml.algorithms import HyperparameterError, get_algorithm
from evalforge.models import (
    AuditEvent,
    DatasetVersion,
    Job,
    JobKind,
    TrainingRun,
    VersionStatus,
)


class RunRequestError(ValueError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def create_training_run(
    db: Session,
    *,
    dataset_version_id: str,
    algorithm: str,
    hyperparameters: dict[str, Any] | None = None,
    seed: int = 42,
    cv_folds: int = 0,
    actor: str = "system",
) -> tuple[Job, TrainingRun]:
    version = db.get(DatasetVersion, dataset_version_id)
    if version is None:
        raise RunRequestError(404, "Dataset version not found")
    if version.status != VersionStatus.validated:
        raise RunRequestError(
            422,
            "Dataset version failed blocking quality checks; fix the data and upload a new version",
        )
    try:
        algo = get_algorithm(algorithm)
        hp = algo.validate(hyperparameters)
    except HyperparameterError as exc:
        raise RunRequestError(422, str(exc)) from exc
    payload = {
        "dataset_version_id": dataset_version_id,
        "algorithm": algo.key,
        "hyperparameters": hp,
        "seed": seed,
        "cv_folds": cv_folds,
    }
    job = Job(kind=JobKind.train, payload=payload)
    db.add(job)
    db.flush()
    run = TrainingRun(
        job_id=job.id,
        dataset_version_id=version.id,
        algorithm=algo.key,
        hyperparameters=hp,
        seed=seed,
        cv_folds=cv_folds,
    )
    db.add(run)
    db.flush()
    db.add(
        AuditEvent(
            actor=actor,
            action="run.launched",
            entity_type="training_run",
            entity_id=run.id,
            data={"algorithm": algo.key, "hyperparameters": hp, "dataset_version_id": version.id},
        )
    )
    return job, run
