# ADR 0003 — Pluggable job execution, state in the database

Status: accepted (2026-11-01)

## Context
Training must not block requests. Redis + a worker is the production shape, but the
laptop profile should not require Redis, and tests need determinism.

## Options
* Celery — heavyweight, many moving parts, prefork pool is unfriendly on macOS.
* Dramatiq / Arq — fine, but either async-first or extra middleware for little gain.
* **RQ** — small, Redis-only, `SimpleWorker` runs jobs in-process without fork.
* FastAPI `BackgroundTasks` — no isolation from the API process, no queue.

## Decision
One `submit(job_id)` function with three executors: `thread` (bounded pool, lite),
`rq` (Redis + `python -m evalforge.worker`), `sync` (tests/CLI). Job state, progress and
logs live in the `jobs` table, not in the executor. Prefect (Phase 3) orchestrates
multi-step flows *on top of* these job handlers rather than replacing them.

## Consequences
* Identical UI behaviour across executors; the thread executor marks interrupted jobs
  failed on restart.
* `max_concurrent_jobs=1` by default — CPU-bound sklearn on 8 GB should not oversubscribe.
