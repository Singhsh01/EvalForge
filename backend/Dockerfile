FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy \
    MLFLOW_DISABLE_AGENT_HINT=1
COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project
COPY evalforge ./evalforge
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev
RUN useradd --create-home --uid 10001 evalforge && mkdir -p /data && chown evalforge /data
USER evalforge
ENV PATH="/app/.venv/bin:$PATH" EVALFORGE_DATA_DIR=/data
ARG GIT_COMMIT=unknown
ENV EVALFORGE_GIT_COMMIT=$GIT_COMMIT
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --retries=5 CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/api/health')"
CMD ["uvicorn", "evalforge.main:app", "--host", "0.0.0.0", "--port", "8000"]
