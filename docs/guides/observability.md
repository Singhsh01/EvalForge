# Observability guide

## Traces
* Every `/api/*` request is a server span (`http.route`, status, request id, user). Background
  jobs continue the request's trace: the trace context is stored in the job payload and
  restored in the worker, so `POST /api/runs` → `job.*` → `model.fit` → `mlflow.log_run` is one
  trace even with the Redis worker.
* Instrumented operations: training stages (span events), model fitting, MLflow logging,
  inference (`model.predict`), pipeline steps, drift reports, LLM calls (tokens), evaluation
  runs and cases, Copilot sessions and each tool call.
* Correlation ids on spans/rows: `job.id` (and `job.result.trace_id`), `eval.run_id`,
  `pipeline.run_id`, `copilot.session_id`, `model.name/version`, `deployment.id`, MLflow tag
  `evalforge.trace_id`. Responses carry `x-trace-id`.
* **Trace explorer** (UI) reads spans from the EvalForge database; set
  `EVALFORGE_OTLP_ENDPOINT` to also export to an OpenTelemetry Collector (→ Jaeger).

## Redaction
Keys matching password/secret/token/api key/authorization/cookie/credential/private are replaced
with `[REDACTED]`; values matching provider keys (`sk-…`), EvalForge tokens (`efk_…`), bearer
tokens, AWS access keys and e-mail addresses are masked — when set and again at export.

## Metrics (`GET /metrics`, Prometheus format)
`evalforge_http_requests_total`, `evalforge_http_request_duration_seconds`,
`evalforge_predictions_total`, `evalforge_inference_latency_seconds`, `evalforge_jobs_total`,
`evalforge_pipeline_step_retries_total`, `evalforge_llm_tokens_total`,
`evalforge_llm_errors_total`, `evalforge_copilot_tool_calls_total`, `evalforge_drift_alerts_total`.

## Optional stack
```bash
echo "EVALFORGE_OTLP_ENDPOINT=http://otel-collector:4318" >> .env
docker compose --profile observability up --build -d
# Jaeger http://localhost:16686 · Prometheus http://localhost:9090 · Grafana http://localhost:3001
```
Grafana is provisioned with an *EvalForge operations* dashboard (request rate, p95, error
ratio, predictions by version, inference p95, jobs, LLM tokens, tool calls, drift alerts,
pipeline retries).

## In-app health
*System health & audit* aggregates server spans (per-route p50/p95, 4xx/5xx), job outcomes,
pipeline retries, inference latency and span error counts for 1 h / 24 h / 7 d windows.
