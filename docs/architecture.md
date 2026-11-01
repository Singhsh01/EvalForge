# EvalForge architecture

EvalForge is a single FastAPI control plane with a Next.js UI, a background worker, MLflow as
the system of record for runs and models, and a relational metadata store for everything MLflow
does not model: validated data versions, governance, deployments, evaluations, telemetry,
incidents and agent sessions.

## 1. System context

```mermaid
flowchart LR
  user([Engineer / Approver / Viewer]) -->|browser| web[Next.js app :3000]
  web -->|/api/* proxied| api[FastAPI control plane :8000]
  api --> meta[(PostgreSQL / SQLite<br/>metadata, governance,<br/>spans, incidents)]
  api -->|enqueue + trace context| q[(Redis / RQ)]
  q --> worker[Worker]
  api -. lite: thread pool .-> worker
  worker --> meta
  worker -->|runs, models, aliases| mlflow[MLflow tracking + registry]
  api -->|load champion/canary| mlflow
  api --> files[(Content-addressed<br/>dataset store)]
  api -->|LLM calls, optional| llm[Anthropic / OpenAI-compatible]
  api -->|OTLP, optional| otel[OTel Collector → Jaeger]
  prom[Prometheus] -->|/metrics| api
  graf[Grafana] --> prom
```

| | Lite (`make dev`) | Full (`make compose-up`) | + observability profile |
|---|---|---|---|
| Metadata | SQLite (WAL) | PostgreSQL 16 | |
| MLflow | embedded | MLflow server + Postgres | |
| Jobs | thread pool | Redis + RQ worker | |
| Telemetry | spans in DB, `/metrics` | same | + Collector, Jaeger, Prometheus, Grafana |
| Memory | ~450 MB | ~2.3 GB limits | + ~0.9 GB |

## 2. Modules

| Layer | Package | Responsibility |
|---|---|---|
| API | `api/auth, datasets, runs, ops, pipelines, llmops, monitoring, copilot, observability, system` | HTTP contracts, role checks |
| Governance | `services/auth`, `services/governance` | tokens, roles, rate limits, approvals with executors |
| ML lifecycle | `services/datasets, profiling, training, experiments, registry, serving, model_cache` | data versions, training, gates, deployments, inference |
| Orchestration | `services/jobs`, `services/pipelines` | executors (thread/RQ/sync), DAG runs, retries, resume, schedules |
| LLMOps | `llm/providers`, `llm/evaluators`, `services/evals` | provider abstraction, evaluators, suites, runs, regressions |
| Monitoring | `services/drift` | KS/χ²/PSI drift, prediction drift, label-aware performance, incidents, retraining plans |
| Agent | `copilot/tools`, `copilot/agent` | permissioned tools, planners, evidence validation, budgets |
| Telemetry | `observability/tracing`, `observability/metrics` | OTel spans → DB/OTLP with redaction; Prometheus |

## 3. Data model

```mermaid
erDiagram
  datasets ||--o{ dataset_versions : has
  dataset_versions ||--o{ training_runs : trains
  training_runs ||--o| model_versions : registered_as
  model_versions ||--o{ deployments : deployed_as
  deployments ||--o{ prediction_logs : serves
  model_versions ||--o{ drift_reports : monitored_by
  drift_reports ||--o{ incidents : opens
  eval_suites ||--o{ eval_runs : run_as
  eval_runs ||--o{ eval_case_results : scores
  eval_runs ||--o{ incidents : opens
  incidents ||--o{ incident_events : timeline
  incidents ||--o{ copilot_sessions : investigated_by
  agent_configs ||--o{ copilot_sessions : configures
  prompts ||--o{ prompt_versions : versions
  pipeline_runs ||--o{ pipeline_steps : steps
  approvals ||--o{ deployments : authorises
  users ||--o{ api_tokens : holds
```

Plus `jobs` (lifecycle of every background task), `evaluations` (re-scoring of ML runs),
`llm_calls`, `spans`, `pipeline_schedules` and `audit_events`. See
[database-migrations.md](guides/database-migrations.md).

## 4. Key flows

### Governed promotion and deployment
```mermaid
sequenceDiagram
  actor E as Engineer
  actor A as Approver
  participant API
  participant ML as MLflow
  E->>API: register run → model version (+ gate)
  API->>ML: register_model, tags
  E->>API: request promotion
  API-->>E: approval (pending)
  A->>API: approve (≠ requester)
  API->>API: re-run gate (champion re-scored on candidate's holdout)
  API->>ML: set alias champion → vN
  E->>API: request production deploy (canary %)
  A->>API: approve
  API->>ML: load model (skops) · health check
  API-->>A: executed (deployment id, health)
```

### Request → worker trace propagation
The API injects the W3C trace context into the job payload; the worker (thread or RQ process)
extracts it, so request, job, training, MLflow logging and pipeline-step spans share one trace.

### Drift → incident → controlled retraining
`drift report` (reference = training split, current = logged predictions since T) →
verdict (`no_drift` | statistical drift | `performance_degradation`) → incident → Copilot
investigation → retraining **plan** (justified only by label-confirmed degradation and newer
data) → approval → pipeline (`trigger: incident`) → gate vs champion → staging → promotion
approval → production approval → re-measure.

### Copilot investigation
plan (intent via rules or LLM) → tool calls (allow-list, budgets, spans) → analyse →
hypotheses → **validate citations against tool evidence** → report; actions only as approvals.

## 5. Design decisions
[ADR 0001–0010](adr): monorepo, MLflow as system of record, job execution, dataset versioning,
laptop profile, pipeline orchestration, drift statistics, Copilot agent, auth & governance,
observability.

## 6. Deliberate limitations
* One API process holds the scheduler loop and the in-memory rate limiter; multiple replicas need
  a leader lock and Redis-backed limits.
* Classification models only; the serving path runs in-process (no KServe/K8s).
* Span retention is unbounded; add a retention job for long-running installs.
* Real-LLM evaluation numbers require a provider key (none was available during development).
