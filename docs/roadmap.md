# Capability matrix

Status after the engineering audit ([audit report](audit/AUDIT-REPORT.md), branch
`feature/audit-hardening`). Evidence levels:

* **Verified**: an automated test reproduces the behaviour (failure cases included).
* **Measured**: a number from a recorded, reproducible run in `docs/results/`.
* **Implemented, unvalidated**: code exists and is exercised only with test doubles.
* **Planned / deferred**: not built (with the reason).

| Area | Capability | Level | Evidence |
|---|---|---|---|
| **Security** | Bearer tokens (hashed at rest), expiry, revocation | Verified | `test_ops::test_logout_revokes_token` |
| | Every operation authenticated; viewer cannot write (sweep of all 83 operations) | Verified | `test_security_audit::test_every_operation_requires_authentication`, `…viewer_cannot_call_any_write_operation` |
| | Separation of duties, including admins; atomic approval claim | Verified | `…separation_of_duties_holds_for_admin`, `…concurrent_approvals_execute_once` |
| | Rate limits per IP + username (login), per user (writes, serving), per IP (invalid tokens) | Verified | `…rate_limit_cannot_be_bypassed…` (SEC-01) |
| | Constant-time login, credential redaction in `/api/meta`, sanitised request ids | Verified | SEC-02, SEC-03, SEC-07 tests |
| | Model artifacts: skops-only, allowlisted types, SHA-256 verified on load | Verified | `…tampered_model_artifact_is_refused` (SEC-05) |
| | Secret/PII redaction in telemetry | Verified | `test_observability::test_redaction_of_secrets_in_spans` |
| | SSO/OIDC, MFA, HttpOnly-cookie sessions + CSP | Planned | audit §5 |
| **Registry & deployment** | MLflow registration, champion alias, model cards | Verified | `test_ops::test_full_lifecycle…` |
| | Quality gate with champion re-scored on the same holdout | Verified | `test_ops::test_gate_blocks_regression_against_champion` |
| | Paired bootstrap ΔF1 interval + distinguishability check in the gate | Verified, Measured | `test_benchmark_stats`, [case-study-uncertainty.json](results/case-study-uncertainty.json) |
| | One active stable/canary per model and environment (DB-enforced) | Verified | `test_resilience::test_concurrent_deploys…` (SQLite + PostgreSQL) |
| | Health-checked deploy, canary (re-checked at promotion), rollback refusing broken targets | Verified | `test_resilience` |
| | Inference API, prediction logging, label feedback | Verified, Measured | `test_ops`, `test_drift`, [performance.json](results/performance.json) |
| | Multi-worker serving | Measured (2 workers +43 % at 16 clients); **unsafe** until the scheduler has a leader lock | audit PERF-02 |
| | KServe / Kubernetes serving | Planned | — |
| **Pipelines** | Persisted DAG, retries, resume, lineage, schedules | Verified | `test_pipelines` |
| | Lost-worker recovery (stale-job reaper, every executor) | Verified | `test_resilience::test_stale_running_job_is_reaped…` |
| | Prefect / DVC | Deferred | ADR 0006 / 0004 |
| **Evaluation** | Deterministic, schema, JSON-field, tool-use, retrieval, groundedness, citation evaluators | Verified | `test_llmops`, `test_llm_benchmark` |
| | Bootstrap 95 % intervals on F1 (runs, evaluations, champion re-scores) | Verified | `test_benchmark_stats` |
| | Classic ML on 3 public datasets × 3 algorithms with intervals and CV | Measured | [ml-benchmark.json](results/ml-benchmark.json) |
| | Real-LLM benchmark harness (generated ground truth, Wilson/bootstrap intervals, latency, tokens) | Verified (plumbing, preflight); **results pending** | `test_llm_benchmark`, BEN-03 |
| | LLM providers (Anthropic, OpenAI-compatible/Ollama), LLM judge + calibration | Implemented, unvalidated | test doubles only; no real-model run yet |
| | Regression detection → incident; outages counted as errors, not scores | Verified | `test_llmops`, `test_agent_robustness::…planner_outage…` |
| **Observability** | OTel spans (API, worker, training, MLflow, inference, LLM, evals, Copilot), trace explorer | Verified, Measured | `test_observability`; trace list 2.1–2.7× faster (PERF-03) |
| | Prometheus metrics, Grafana dashboard | Verified (metrics); dashboard provisioned, not load-tested | `test_observability` |
| | Evidently | Deferred | ADR 0007 |
| **Drift** | KS/χ²/PSI, prediction drift, label-aware performance, incidents, approval-gated retraining | Verified | `test_drift`, case study 3 |
| **Copilot** | Permissioned tools, rules + LLM planners, budgets, cancellation | Verified | `test_copilot`, `test_agent_robustness` |
| | Actions bound to the caller's identity and role | Verified | SEC-06 tests |
| | Containment of malformed model output, tool exceptions, DB errors | Verified | AGT-01/02 tests |
| | Indirect prompt-injection framing; numeric claims checked against tool results | Verified (scripted adversarial model) | AGT-03/04 tests; real-model behaviour unmeasured |
| | LangGraph | Not used | ADR 0008 |
| **Frontend** | 15 modules + sign-in on the API; React Flow DAG/lineage; ECharts | Verified | `npm run e2e` |
| | Component tests (API client, sign-in, DAG, lineage, charts, brand background, UI) | Verified | `npm test` (24) |
| | Real user operations (two-person rollback, deploy/rollback, viewer read-only) | Verified | `npm run e2e:ops` (10 checks) |
| | Optional Higgsfield visual assets | Implemented (cost-gated tooling); no assets generated | `docs/guides/visual-assets.md` |
| **Ops** | Docker Compose (core + observability), CI (lint, tests on SQLite + PostgreSQL, migration drift, component tests, builds, security scans, e2e) | Verified in CI; images not built in the audit environment | `.github/workflows/ci.yml` |
| | Load testing harness | Measured | `make perf` |

Open items from the audit: RES-04, PERF-02, PERF-04, UX-01, OPS-01, BEN-02 (blocking rule) and the
pending real-LLM run. Details and recommendations: [audit report §2, §5](audit/AUDIT-REPORT.md).
