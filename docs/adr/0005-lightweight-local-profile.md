# ADR 0005 — Lightweight local profile for an 8 GB laptop

Status: accepted (2026-11-01)

## Decision
* Default config needs no Docker: SQLite (WAL), embedded MLflow, thread executor, one job
  at a time, RandomForest `n_jobs=2` during fit and `n_jobs=1` for prediction.
* Docker Compose profile sets memory limits summing to ~2.3 GB and binds ports to
  localhost. Kubernetes/KServe, Grafana/Prometheus and Prefect server are optional and
  arrive in later phases behind compose profiles, never as requirements.
* Fonts are self-hosted from npm (`@fontsource`) so frontend builds work offline.
* Models are CPU-only scikit-learn; PyTorch is deferred until a task needs it.

## Evidence
Setting `n_jobs=1` for inference reduced measured single-row RandomForest p50 latency
from 66.8 ms to 16.2 ms in the demo (thread dispatch overhead dominated).
