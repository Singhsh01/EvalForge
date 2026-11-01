"""FastAPI application factory."""

from __future__ import annotations

import contextlib
import logging
import os
import threading
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from sqlalchemy.exc import IntegrityError

from evalforge import __version__
from evalforge.api import (
    auth,
    copilot,
    datasets,
    llmops,
    monitoring,
    observability,
    ops,
    pipelines,
    runs,
    system,
)
from evalforge.config import get_settings
from evalforge.migrations import upgrade_to_head
from evalforge.observability import metrics, tracing
from evalforge.services import jobs
from evalforge.services.auth import ensure_bootstrap_users
from evalforge.services.tracking import UnsafeModelArtifact

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("evalforge")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    upgrade_to_head()
    ensure_bootstrap_users()
    tracing.setup()
    stop = threading.Event()
    if get_settings().scheduler_enabled and get_settings().executor != "sync":
        threading.Thread(
            target=_scheduler_loop, args=(stop,), daemon=True, name="evalforge-scheduler"
        ).start()
    n = jobs.recover_orphans() + jobs.reap_stale()
    if n:
        log.warning("marked %d interrupted job(s) as failed", n)
    yield
    stop.set()
    tracing.flush()


def _scheduler_loop(stop: threading.Event) -> None:
    from evalforge.services.pipelines import run_due_schedules

    while not stop.wait(get_settings().scheduler_interval_seconds):
        try:
            if reaped := jobs.reap_stale():
                log.warning("reaped %d stale job(s) whose worker was lost", reaped)
            started = run_due_schedules()
            if started:
                log.info("scheduler started pipeline runs %s", started)
        except Exception:  # keep the loop alive; failures are logged
            log.exception("scheduler tick failed")


def create_app() -> FastAPI:
    s = get_settings()
    app = FastAPI(
        title="EvalForge API",
        version=__version__,
        description="AI evaluation, MLOps and LLMOps control plane.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware, allow_origins=s.cors_origins, allow_methods=["*"], allow_headers=["*"]
    )

    # Rate limiting lives in the auth layer (services/auth.py): logins per IP and username,
    # authenticated calls per user, invalid tokens per IP. A header-keyed middleware here was
    # bypassable by rotating forged Authorization headers (audit SEC-01).

    @app.middleware("http")
    async def request_context(request: Request, call_next):  # type: ignore[no-untyped-def]
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        t0 = time.perf_counter()
        path = request.url.path
        traced = path.startswith("/api/") and not path.startswith(("/api/jobs", "/api/traces"))
        cm = (
            tracing.tracer().start_as_current_span(
                f"{request.method} {path}", kind=trace.SpanKind.SERVER
            )
            if traced
            else contextlib.nullcontext()
        )
        with cm as span:
            response = await call_next(request)
            route = getattr(request.scope.get("route"), "path", path)
            if span is not None:
                span.update_name(f"{request.method} {route}")
                principal = getattr(request.state, "principal", None)
                span.set_attributes(
                    {
                        "http.method": request.method,
                        "http.route": route,
                        "http.status_code": response.status_code,
                        "http.request_id": rid,
                        "enduser.id": principal.username if principal else "",
                    }
                )
                if response.status_code >= 500:
                    span.set_status(Status(StatusCode.ERROR, f"HTTP {response.status_code}"))
                response.headers["x-trace-id"] = tracing.current_trace_id() or ""
        ms = (time.perf_counter() - t0) * 1000
        metrics.HTTP_REQUESTS.labels(request.method, route, str(response.status_code)).inc()
        metrics.HTTP_LATENCY.labels(request.method, route).observe(ms / 1000)
        response.headers["x-request-id"] = rid
        response.headers["server-timing"] = f"app;dur={ms:.1f}"
        if not path.startswith("/api/jobs"):
            log.info("%s %s %s %.1fms rid=%s", request.method, path, response.status_code, ms, rid)
        return response

    @app.exception_handler(IntegrityError)
    async def integrity_conflict(request: Request, exc: IntegrityError) -> JSONResponse:
        # e.g. two concurrent deploys racing for the single active slot (audit RES-03)
        log.warning("integrity conflict on %s: %s", request.url.path, str(exc.orig)[:200])
        return JSONResponse(
            {"detail": "Conflicting concurrent change; reload and retry"}, status_code=409
        )

    @app.exception_handler(UnsafeModelArtifact)
    async def unsafe_artifact(request: Request, exc: UnsafeModelArtifact) -> JSONResponse:
        log.error("refused unsafe model artifact on %s: %s", request.url.path, exc)
        return JSONResponse({"detail": f"Model artifact refused: {exc}"}, status_code=409)

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics() -> Response:
        return Response(metrics.render(), media_type="text/plain; version=0.0.4")

    app.include_router(system.router)
    app.include_router(auth.router)
    app.include_router(ops.router)
    app.include_router(pipelines.router)
    app.include_router(observability.router)
    app.include_router(monitoring.router)
    app.include_router(llmops.router)
    app.include_router(copilot.router)
    app.include_router(datasets.router)
    app.include_router(runs.router)
    return app


app = create_app()
