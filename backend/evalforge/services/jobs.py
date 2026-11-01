"""Background job execution.

Three interchangeable executors behind one function, :func:`submit`:

* ``thread`` — bounded in-process pool (lite profile; no Redis needed).
* ``rq``     — Redis Queue; run ``python -m evalforge.worker`` in another process.
* ``sync``   — run inline (tests, CLI seeding).

Job state lives in the database, not in the executor, so the UI sees the same
lifecycle regardless of executor, and orphaned jobs are recoverable on restart.
"""

from __future__ import annotations

import logging
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from functools import lru_cache

from evalforge.config import get_settings
from evalforge.db import session_scope
from evalforge.models import Job, JobKind, JobStatus

log = logging.getLogger(__name__)


def execute(job_id: str) -> None:
    """Entry point for every executor. Never raises — failures are recorded."""
    from evalforge.services import training

    with session_scope() as db:
        job = db.get(Job, job_id)
        if job is None or job.status not in (JobStatus.queued, JobStatus.running):
            return
        job.status, job.stage = JobStatus.running, "starting"
        job.started_at = datetime.now(UTC)
        job.attempts += 1
        kind = job.kind
    try:
        handler = {JobKind.train: training.run_training, JobKind.evaluate: training.run_evaluation}
        result = handler[kind](job_id)
        with session_scope() as db:
            job = db.get(Job, job_id)
            assert job is not None
            job.status, job.result, job.progress = JobStatus.succeeded, result, 1.0
            job.finished_at = datetime.now(UTC)
    except Exception as exc:
        log.exception("job %s failed", job_id)
        msg = f"{type(exc).__name__}: {exc}"
        with session_scope() as db:
            job = db.get(Job, job_id)
            assert job is not None
            job.status, job.error, job.stage = JobStatus.failed, msg, "failed"
            job.finished_at = datetime.now(UTC)
            job.logs = [
                *(job.logs or []),
                {
                    "ts": datetime.now(UTC).isoformat(),
                    "stage": "failed",
                    "message": msg,
                    "traceback": traceback.format_exc()[-4000:],
                },
            ]
        training.mark_failed(job_id, msg)


@lru_cache
def _pool() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(
        max_workers=get_settings().max_concurrent_jobs, thread_name_prefix="evalforge-job"
    )


@lru_cache
def _queue():  # type: ignore[no-untyped-def]
    from redis import Redis
    from rq import Queue

    return Queue("evalforge", connection=Redis.from_url(get_settings().redis_url))


def submit(job_id: str) -> None:
    mode = get_settings().executor
    if mode == "sync":
        execute(job_id)
    elif mode == "thread":
        _pool().submit(execute, job_id)
    else:
        _queue().enqueue(
            "evalforge.services.jobs.execute",
            job_id,
            job_id=job_id,
            job_timeout=get_settings().job_timeout_seconds,
        )


def recover_orphans() -> int:
    """Thread executor: a restart kills in-flight jobs, so mark them failed, not running."""
    if get_settings().executor != "thread":
        return 0
    from evalforge.services.training import mark_failed

    with session_scope() as db:
        jobs = db.query(Job).filter(Job.status.in_([JobStatus.queued, JobStatus.running])).all()
        ids = [j.id for j in jobs]
        for j in jobs:
            j.status, j.stage = JobStatus.failed, "interrupted"
            j.error = "Interrupted by API restart (thread executor). Re-launch the run."
            j.finished_at = datetime.now(UTC)
    for i in ids:
        mark_failed(i, "Interrupted by API restart")
    return len(ids)
