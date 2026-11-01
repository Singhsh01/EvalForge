# Roadmap and acceptance criteria

A phase is marked done only when its acceptance criteria are verified by tests or recorded
execution evidence.

| Phase | Scope | Status |
|---|---|---|
| 1 Architecture | Architecture, data model, API contracts, security notes, ADRs | **Done** — [architecture.md](architecture.md), [adr/](adr), [openapi.json](openapi.json) |
| 2 Vertical slice | Dataset upload → profiling/quality → training → MLflow tracking → evaluation → dashboard | **Done** — see evidence below |
| 3 MLOps infrastructure | DVC remote, Prefect flows, MLflow registry (aliases), approvals, serving endpoint, deploy/rollback, CI/CD hardening | Next |
| 4 AI evaluation | Provider abstraction (Claude + OpenAI-compatible), prompt versioning, deterministic / LLM-judge / RAG / agent evaluators, regression gates | Planned |
| 5 Observability | OpenTelemetry traces, Prometheus metrics, Grafana, trace explorer, redaction | Planned |
| 6 Drift & retraining | Evidently drift, alerting, incidents, Prefect retraining with human approval and rollback | Planned |
| 7 Frontend | Remaining modules, React Flow pipelines, trace waterfall, monitoring views | Planned |
| 8 Verification | E2E (Playwright) suite, evaluation benchmarks, load tests | Partially (backend unit/integration in place) |
| 9 Documentation | Case studies, benchmark methodology, final screenshots | Started |

## Phase 2 acceptance criteria and evidence

| Criterion | Evidence |
|---|---|
| CSV and JSON (array + JSON Lines) upload with size/row/column limits | `tests/test_api.py::test_upload_profile_and_versioning`, `::test_rejects_bad_uploads` |
| Schema, missing values, distributions, duplicates profiled | `tests/test_profiling.py::test_profile_has_histograms_and_top_values`; Dataset studio screenshot |
| Quality checks with blocking vs. warning severity; rejected versions can't be trained | `::test_rejected_version_cannot_be_trained`, `test_profiling.py` |
| Leakage checks: label-encoding features, train/test contamination, identifier columns | `test_profiling.py::test_target_leakage_detected`, `::test_split_is_deterministic_stratified_and_detects_overlap` |
| Content-hash dataset versioning, dedupe, schema-compatibility between versions | `::test_upload_profile_and_versioning` |
| Deterministic stratified split shared by all runs of a version | `test_profiling.py`, compare endpoint `same_holdout` |
| ≥2 algorithms with validated hyperparameters, launched from the UI | 3 algorithms; `test_algorithms.py`; Experiment lab screenshot |
| MLflow tracks params, metrics, dataset version, duration, artifacts, commit, environment | `::test_end_to_end_train_track_evaluate_compare` reads the run back from MLflow |
| Background execution with recorded failures | `::test_failed_job_is_recorded_not_raised`; RQ worker used for demo evidence |
| Evaluation against another dataset version using the model loaded from MLflow | `::test_end_to_end_train_track_evaluate_compare` |
| Results dashboard: overview, run detail, comparison | `docs/screenshots/*.png` captured from the running app |
| Works on SQLite and PostgreSQL | test suite run against both; CI runs both |
| Reproducible | Demo executed twice from clean databases produced identical quality metrics |

## Known limitations of the current slice
* No authentication yet — bind to localhost only (compose already does).
* Classification only (binary and multiclass ≤ 50 classes). Regression is a small addition
  to `infer_task_type`, the metrics module and the algorithm catalogue.
* Container images were written but not built in the development sandbox (Docker Hub was
  unreachable there); CI builds and smoke-tests them.
* Latency figures are in-process measurements on the machine that ran the job.
