# MLOps pipeline guide

`continuous_training` (UI: Pipeline studio, API: `/api/pipelines`):

| Step | Does | Retries? |
|---|---|---|
| `validate_data` | version exists and passed blocking checks; records SHA-256 and warnings | no (deterministic) |
| `train` | creates a training run and executes it inline in the worker | yes |
| `evaluate` | holdout metrics; optional extra evaluation on `eval_dataset_version_id` | yes |
| `register` | registers the model version in MLflow (idempotent on resume) | yes |
| `quality_gate` | gate vs. champion; a failure stops the run without retry | no |
| `deploy_staging` | health-checked staging deploy (optional) | yes |
| `request_promotion` | files an approval and waits (`awaiting_approval`) | — |

* **Configuration** is validated (pydantic) and hashed (`config_hash`); identical configs are
  identifiable across runs.
* **Retries:** transient exceptions are retried up to `max_retries` with backoff;
  `StepFailed` (bad data, failed gate) is never retried.
* **Resume:** `POST /api/pipelines/runs/{id}/resume` re-runs from the first failed step,
  reusing outputs of succeeded steps (no retraining).
* **Lineage:** git commit, dataset name/version/SHA-256, run id, MLflow run id, model URI,
  model version, deployment id and approval id are collected on the run.
* **Schedules:** `POST /api/pipelines/schedules {config, interval_minutes ≥ 5}`; the API's
  scheduler loop checks every `EVALFORGE_SCHEDULER_INTERVAL_SECONDS` (30 s).
* **Incident-driven runs:** an approved retraining request starts the same pipeline with
  `trigger: incident`.
* **Verification:** `verify_definition()` checks step order and dependencies before every run;
  CI runs the full pipeline test suite on SQLite and PostgreSQL.

Why a custom runner rather than Prefect: [ADR 0006](../adr/0006-pipeline-orchestration.md).
