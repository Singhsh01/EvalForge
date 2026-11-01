"""FastAPI application factory."""

from __future__ import annotations

import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from evalforge import __version__
from evalforge.api import datasets, runs, system
from evalforge.config import get_settings
from evalforge.migrations import upgrade_to_head
from evalforge.services import jobs

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("evalforge")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    upgrade_to_head()
    n = jobs.recover_orphans()
    if n:
        log.warning("marked %d interrupted job(s) as failed", n)
    yield


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

    @app.middleware("http")
    async def request_context(request: Request, call_next):  # type: ignore[no-untyped-def]
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        t0 = time.perf_counter()
        response = await call_next(request)
        ms = (time.perf_counter() - t0) * 1000
        response.headers["x-request-id"] = rid
        response.headers["server-timing"] = f"app;dur={ms:.1f}"
        if not request.url.path.startswith("/api/jobs"):  # polled frequently
            log.info(
                "%s %s %s %.1fms rid=%s",
                request.method,
                request.url.path,
                response.status_code,
                ms,
                rid,
            )
        return response

    app.include_router(system.router)
    app.include_router(datasets.router)
    app.include_router(runs.router)
    return app


app = create_app()
