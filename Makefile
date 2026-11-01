# EvalForge developer commands. Lite profile needs only uv and Node 20+.
.PHONY: setup dev api web worker test lint demo compose-up compose-down mlflow-ui clean

setup:            ## install backend + frontend dependencies
	cd backend && uv sync
	cd frontend && npm ci

api:              ## FastAPI on :8000 (SQLite, embedded MLflow, in-process jobs)
	cd backend && uv run uvicorn evalforge.main:app --reload --port 8000

web:              ## Next.js on :3000, proxies /api to :8000
	cd frontend && npm run dev

dev:              ## api + web together (Ctrl-C stops both)
	$(MAKE) -j2 api web

worker:           ## RQ worker (only when EVALFORGE_EXECUTOR=rq)
	cd backend && uv run python -m evalforge.worker

mlflow-ui:        ## browse the embedded MLflow store on :5000
	cd backend && uv run mlflow ui --backend-store-uri sqlite:///var/mlflow.db --port 5000

demo:             ## Demonstration: public dataset -> 3 runs -> comparison (API must be running)
	cd backend && uv run python -m evalforge.demo.run_demo --out ../docs/results

test:
	cd backend && uv run pytest
	cd frontend && npm run typecheck

lint:
	cd backend && uv run ruff check evalforge tests && uv run ruff format --check evalforge tests

compose-up:
	GIT_COMMIT=$$(git rev-parse HEAD 2>/dev/null || echo unknown) docker compose up --build -d

compose-down:
	docker compose down

clean:
	rm -rf backend/var frontend/.next
