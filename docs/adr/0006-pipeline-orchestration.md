# ADR 0006 — Pipeline orchestration: persisted DAG runner instead of Prefect (for now)

Status: accepted (2026-11-01)

## Context
The continuous-training workflow needs ordered steps, retries, resume after failure,
lineage, schedules and a UI that shows real step state. Prefect was the suggested tool.

## Options
* **Prefect 3** — mature; but local use either spins up an ephemeral API server per
  process (SQLite, ~150–300 MB, seconds of start-up) or needs a long-running server and a
  worker beside our own RQ worker. Its state lives in Prefect's database, so the UI would
  need a second source of truth or Prefect's API.
* **Persisted DAG runner on top of the existing job system** — ~300 lines; steps are plain
  functions over a shared context; state transitions are rows in our database.

## Decision
Use the in-house runner (`services/pipelines.py`): step-level retries for transient errors
(no retry for deterministic `StepFailed`, e.g. a failed quality gate), resume from the first
failed step reusing earlier outputs, config hashing, lineage, interval schedules driven by
the API's scheduler loop, OpenTelemetry span per step.

Step functions have no runner coupling, so wrapping them as Prefect `@task`s inside a
`@flow` is mechanical if a team already operates Prefect. That is a deployment choice,
not an architectural one.

## Consequences
* One source of truth for pipeline state; the Pipeline studio reads it directly.
* No distributed fan-out: steps run sequentially in one worker. Acceptable for the
  CPU-bound, laptop-scale workloads this targets.
* The scheduler loop runs in the API process; with several API replicas a leader lock
  (Postgres advisory lock) would be needed. Documented limitation.
