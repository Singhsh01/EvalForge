# ADR 0001 — Monorepo: FastAPI backend, Next.js frontend

Status: accepted (2026-11-01)

## Context
The platform's core is ML and evaluation code, which lives in the Python ecosystem
(scikit-learn, MLflow, Evidently, Ragas, LangGraph, Prefect). The UI needs rich,
interactive visualisation.

## Decision
`backend/` is a single FastAPI service (Pydantic v2, SQLAlchemy 2, Alembic) managed with
`uv` and a lockfile. `frontend/` is Next.js (App Router) + TypeScript + Tailwind v4.
The browser only talks to the Next server; `/api/*` is proxied to FastAPI by a rewrite.

## Consequences
* One language for all ML logic; the API, worker and Prefect flows share modules.
* No CORS or exposed backend port needed in the container profile.
* Two toolchains to maintain (uv, npm). Accepted.
