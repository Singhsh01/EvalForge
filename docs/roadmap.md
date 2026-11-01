# Implemented vs. planned

Status of the enterprise milestones (branch `feature/enterprise-ai-operations`). "Verified"
names the automated test or the recorded demo that exercises the feature.

| Area | Feature | Status | Verified by |
|---|---|---|---|
| **Security** | Bearer-token auth, PBKDF2 passwords, token revocation | ✅ | `test_ops::test_auth_required_and_roles`, `test_logout_revokes_token` |
| | Roles viewer/engineer/approver/admin | ✅ | `test_ops`, browser e2e |
| | Separation of duties on approvals | ✅ | `test_ops::test_self_approval_blocked_and_rejection` |
| | Rate limiting | ✅ | `test_ops::test_login_rate_limited` |
| | Secret/PII redaction in telemetry | ✅ | `test_observability::test_redaction_of_secrets_in_spans` |
| | SSO/OIDC, MFA | ⏳ planned | — |
| **Registry & deployment** | MLflow registration, `champion` alias, model cards | ✅ | `test_ops::test_full_lifecycle…`, case study 1 |
| | Quality gates incl. champion re-scored on the same holdout | ✅ | `test_ops::test_gate_blocks_regression_against_champion` |
| | Staging/production, release health checks, failed-health records | ✅ | `test_ops::test_production_requires_approved_version…` |
| | Canary routing, canary promotion, rollback | ✅ | `test_ops::test_full_lifecycle…`, case study 1 (101/400 canary) |
| | Versioned inference API, prediction logging, label feedback | ✅ | `test_ops`, `test_drift` |
| | KServe / Kubernetes serving | ⏳ planned (optional) | — |
| **Pipelines** | Persisted DAG, retries, resume, config hash, lineage | ✅ | `test_pipelines` (5 tests) |
| | Interval schedules | ✅ | `test_pipelines::test_schedules_trigger_when_due` |
| | Prefect / DVC integration | ⏳ deferred — see ADR 0006 / 0004 | — |
| **LLMOps** | Provider abstraction (Anthropic, OpenAI-compatible) | ✅ implemented | unit-tested via test double; **no live-LLM run** (no key) |
| | Prompt registry, versions, diff, approval-gated release | ✅ | `test_llmops::test_prompt_registry…`, `…release` |
| | Deterministic, JSON-schema, reference, tool-call, task, efficiency evaluators | ✅ | `test_llmops::test_agent_evaluators` etc. |
| | RAG retrieval metrics (TF-IDF retriever), groundedness, citation checks | ✅ | `test_llmops::test_rag_retrieval_suite…`, `test_groundedness…` |
| | LLM-as-judge + human calibration | ✅ implemented | test double only |
| | Regression detection → incident | ✅ | `test_llmops`, case study 2 |
| **Observability** | OTel spans (API, worker, training, MLflow, inference, LLM, evals, Copilot) | ✅ | `test_observability::test_request_trace…` |
| | DB exporter + OTLP export, trace explorer | ✅ | `test_observability`, screenshots |
| | Prometheus metrics, Grafana dashboard | ✅ metrics tested; dashboard provisioned | `test_observability::test_metrics…` |
| **Drift** | KS/χ²/PSI feature drift, prediction drift, label-aware performance | ✅ | `test_drift` (3 tests), case study 3 |
| | Incidents, retraining plans, approval-gated retraining | ✅ | `test_drift::test_drift_lifecycle…`, case study 3 |
| | Evidently | ⏳ deferred — ADR 0007 | — |
| **Copilot** | 16 permissioned tools, rules + LLM planners, evidence validation, budgets, cancellation | ✅ | `test_copilot` (6 tests) |
| | Incident investigation | ✅ | `test_copilot::test_agent_regression…`, case studies 2–3 |
| | LangGraph | ⏳ not used — ADR 0008 | — |
| **Frontend** | 15 modules + sign-in, all wired to the API | ✅ | `npm run build`, `frontend/e2e/smoke.mjs` |
| | Component unit tests | ⏳ planned (covered by e2e smoke for now) | — |
| **Ops** | Docker Compose (core + observability profile), CI (lint, tests on SQLite + PostgreSQL, migration drift, builds, security scans, e2e) | ✅ written | CI on GitHub; container images not built locally (Docker Hub blocked in dev sandbox) |

Results: [docs/results/test-report.md](results/test-report.md).
