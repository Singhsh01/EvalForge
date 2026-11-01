# ADR 0010 — Observability: OpenTelemetry with a database exporter

Status: accepted (2026-11-01)

## Decision
* OpenTelemetry SDK spans for HTTP requests, background jobs (trace context propagated
  through the job payload, so worker spans join the request's trace), training stages,
  MLflow logging, inference, pipeline steps, LLM calls, evaluation cases, drift reports and
  Copilot tool calls.
* Spans are exported by a **batch processor** to the EvalForge database (the Trace explorer
  needs no external backend) and, when configured, to an OTLP collector → Jaeger.
  Synchronous export was tried first and rejected: it made request threads wait on the
  metadata store (and on SQLite's write lock).
* Redaction at attribute-set time and again at export: sensitive keys (password, token,
  api_key, authorization, cookie, …) are dropped; values matching key patterns
  (`sk-…`, `efk_…`, `Bearer …`, AWS keys) and e-mail addresses are masked.
* Prometheus metrics at `/metrics`; Grafana dashboard provisioned in the observability profile.

## Consequences
* Telemetry storage grows with traffic; a retention job (delete spans older than N days)
  is needed for long-running installs. Documented limitation.
