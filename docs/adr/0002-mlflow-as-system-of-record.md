# ADR 0002 — MLflow is the system of record for runs and models

Status: accepted (2026-11-01)

## Decision
Every training run is logged to MLflow (params, metrics, tags, dataset input, diagnostics,
model with signature and input example). EvalForge's DB stores a pointer (`mlflow_run_id`,
`model_uri`) plus the workflow state MLflow does not model: validated dataset versions,
jobs, cross-version evaluations, approvals, audit.

Registry promotion (Phase 3) will use **model version aliases** (`champion`,
`challenger`) and tags. Stage transitions (`Staging`/`Production`) are deprecated in
MLflow and will not be used.

Models are logged with `serialization_format="skops"` so loading never unpickles arbitrary
code; untrusted types are computed at save time from our own pipeline.

## Consequences
* The MLflow UI works as a second window on the same experiments.
* Lite profile uses an embedded SQLite tracking store; full profile uses an MLflow server
  with a Postgres backend and served artifacts. Code is identical (tracking URI only).
