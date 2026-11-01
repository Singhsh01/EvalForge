"""Prometheus metrics (scraped from ``GET /metrics``)."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

REGISTRY = CollectorRegistry(auto_describe=True)

HTTP_REQUESTS = Counter(
    "evalforge_http_requests_total",
    "HTTP requests",
    ["method", "route", "status"],
    registry=REGISTRY,
)
HTTP_LATENCY = Histogram(
    "evalforge_http_request_duration_seconds",
    "HTTP request latency",
    ["method", "route"],
    registry=REGISTRY,
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)
PREDICTIONS = Counter(
    "evalforge_predictions_total",
    "Model predictions served",
    ["model", "version", "environment", "role"],
    registry=REGISTRY,
)
INFERENCE_LATENCY = Histogram(
    "evalforge_inference_latency_seconds",
    "Model inference latency",
    ["model", "environment"],
    registry=REGISTRY,
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 1),
)
JOBS = Counter(
    "evalforge_jobs_total", "Background jobs finished", ["kind", "status"], registry=REGISTRY
)
JOB_RETRIES = Counter(
    "evalforge_pipeline_step_retries_total", "Pipeline step retries", ["step"], registry=REGISTRY
)
LLM_TOKENS = Counter(
    "evalforge_llm_tokens_total", "LLM tokens", ["model", "direction"], registry=REGISTRY
)
LLM_ERRORS = Counter("evalforge_llm_errors_total", "LLM call errors", ["model"], registry=REGISTRY)
TOOL_CALLS = Counter(
    "evalforge_copilot_tool_calls_total",
    "Copilot tool calls",
    ["tool", "status"],
    registry=REGISTRY,
)
DRIFT_ALERTS = Counter(
    "evalforge_drift_alerts_total",
    "Drift reports with drift detected",
    ["model"],
    registry=REGISTRY,
)


def render() -> bytes:
    return generate_latest(REGISTRY)
