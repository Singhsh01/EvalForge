"""OpenTelemetry tracing with a database exporter and redaction.

Spans come only from instrumented code paths (HTTP requests, jobs, training stages,
MLflow logging, inference, LLM calls, evaluation cases, Copilot tool calls). They
are exported to:

* the EvalForge ``spans`` table, which backs the Trace Explorer (always, when
  tracing is enabled), and
* an OTLP/HTTP collector when ``EVALFORGE_OTLP_ENDPOINT`` is set (Jaeger, Tempo,
  Grafana, ... via the OpenTelemetry Collector in the observability compose profile).

Redaction happens twice: attribute values are scrubbed when set through
:func:`span`, and every exported span is scrubbed again before it leaves the
process, so secrets set by third-party code are caught too.
"""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import Status, StatusCode

from evalforge.config import get_settings

log = logging.getLogger(__name__)

SENSITIVE_KEY = re.compile(
    r"pass(word)?|secret|token|api[_-]?key|authorization|cookie|"
    r"credential|private",
    re.I,
)
SENSITIVE_VALUE = [
    re.compile(r"\b(sk|efk|pk|rk)[-_][A-Za-z0-9_\-]{12,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]{8,}", re.I),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
]
REDACTED = "[REDACTED]"
_lock = threading.Lock()
_provider: TracerProvider | None = None


def redact_value(v: Any) -> Any:
    if isinstance(v, str):
        for pat in SENSITIVE_VALUE:
            v = pat.sub(REDACTED, v)
        return v[:4000]
    if isinstance(v, (list, tuple)):
        return [redact_value(x) for x in v]
    return v


def redact(attrs: dict[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in (attrs or {}).items():
        out[k] = REDACTED if SENSITIVE_KEY.search(k) else redact_value(v)
    return out


def _clean_attr(v: Any) -> Any:
    if isinstance(v, (str, bool, int, float)):
        return v
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    return str(v)


class DatabaseSpanExporter(SpanExporter):
    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        from evalforge.db import session_scope
        from evalforge.models import Span

        try:
            with session_scope() as db:
                for s in spans:
                    ctx = s.get_span_context()
                    if ctx is None or s.start_time is None or s.end_time is None:
                        continue
                    status = {StatusCode.OK: "ok", StatusCode.ERROR: "error"}.get(
                        s.status.status_code, "unset"
                    )
                    db.merge(
                        Span(
                            span_id=format(ctx.span_id, "016x"),
                            trace_id=format(ctx.trace_id, "032x"),
                            parent_id=format(s.parent.span_id, "016x") if s.parent else None,
                            name=s.name[:256],
                            kind=s.kind.name.lower(),
                            service=str(s.resource.attributes.get("service.name", "evalforge")),
                            start_time=datetime.fromtimestamp(s.start_time / 1e9, UTC),
                            duration_ms=(s.end_time - s.start_time) / 1e6,
                            status=status,
                            status_message=redact_value(s.status.description),
                            attributes=redact(dict(s.attributes or {})),
                            events=[
                                {
                                    "name": e.name,
                                    "attributes": redact(dict(e.attributes or {})),
                                    "time": datetime.fromtimestamp(
                                        e.timestamp / 1e9, UTC
                                    ).isoformat(),
                                }
                                for e in s.events
                            ],
                        )
                    )
            return SpanExportResult.SUCCESS
        except Exception:  # never let telemetry break the application
            log.debug("span export failed", exc_info=True)
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        return None


class RedactingExporter(SpanExporter):
    """Wraps another exporter (OTLP) and scrubs attributes before export."""

    def __init__(self, inner: SpanExporter) -> None:
        self.inner = inner

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        clean = [
            ReadableSpan(
                name=s.name,
                context=s.get_span_context(),
                parent=s.parent,
                resource=s.resource,
                attributes=redact(dict(s.attributes or {})),
                events=s.events,
                links=s.links,
                kind=s.kind,
                status=s.status,
                start_time=s.start_time,
                end_time=s.end_time,
                instrumentation_scope=s.instrumentation_scope,
            )
            for s in spans
        ]
        return self.inner.export(clean)

    def shutdown(self) -> None:
        self.inner.shutdown()


def setup() -> None:
    global _provider
    s = get_settings()
    with _lock:
        if _provider is not None or not s.tracing_enabled:
            return
        provider = TracerProvider(
            resource=Resource.create(
                {"service.name": s.service_name, "deployment.environment": s.env}
            )
        )
        # Always batched on a background thread: exporting synchronously would make
        # request threads wait on the metadata store (and, on SQLite, on its write lock).
        provider.add_span_processor(
            BatchSpanProcessor(DatabaseSpanExporter(), schedule_delay_millis=500)
        )
        if s.otlp_endpoint:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            provider.add_span_processor(
                BatchSpanProcessor(
                    RedactingExporter(
                        OTLPSpanExporter(endpoint=f"{s.otlp_endpoint.rstrip('/')}/v1/traces")
                    )
                )
            )
        _provider = provider


def reset() -> None:
    """Tests: drop the provider so the next setup() binds to the current database."""
    global _provider
    with _lock:
        if _provider is not None:
            _provider.shutdown()
        _provider = None


def flush() -> None:
    if _provider is not None:
        _provider.force_flush(5000)


def tracer() -> trace.Tracer:
    setup()
    if _provider is None:
        return trace.NoOpTracer()
    return _provider.get_tracer("evalforge")


@contextmanager
def span(
    name: str,
    attributes: dict[str, Any] | None = None,
    kind: trace.SpanKind = trace.SpanKind.INTERNAL,
) -> Iterator[Any]:
    with tracer().start_as_current_span(name, kind=kind) as sp:
        if attributes:
            sp.set_attributes(
                {k: _clean_attr(v) for k, v in redact(attributes).items() if v is not None}
            )
        try:
            yield _SafeSpan(sp)
        except Exception as exc:
            sp.record_exception(exc)
            sp.set_status(Status(StatusCode.ERROR, f"{type(exc).__name__}: {exc}"[:500]))
            raise


class _SafeSpan:
    def __init__(self, sp: Any) -> None:
        self._sp = sp

    def set_attributes(self, attrs: dict[str, Any]) -> None:
        self._sp.set_attributes(
            {k: _clean_attr(v) for k, v in redact(attrs).items() if v is not None}
        )

    def add_event(self, name: str, attrs: dict[str, Any] | None = None) -> None:
        self._sp.add_event(
            name, {k: _clean_attr(v) for k, v in redact(attrs).items() if v is not None}
        )

    def error(self, message: str) -> None:
        self._sp.set_status(Status(StatusCode.ERROR, redact_value(message)[:500]))


def current_trace_id() -> str | None:
    ctx = trace.get_current_span().get_span_context()
    return format(ctx.trace_id, "032x") if ctx and ctx.is_valid else None
