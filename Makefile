# EvalForge developer commands. Lite profile needs only uv and Node 20+.
LLM_MODEL ?= openai:qwen2.5:3b

.PHONY: perf llm-api llm-benchmark setup dev api web worker test lint demo demo-enterprise e2e compose-up compose-down observability mlflow-ui clean

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

demo-enterprise:  ## Case studies 1-3 end to end through the API (API + worker must be running)
	cd backend && uv run python -m evalforge.demo.enterprise_demo --out ../docs/results

perf:             ## Load test a running API (throughput, p50/p95/p99, memory) -> docs/results/performance.json
	cd backend && uv run python -m evalforge.benchmarks.load_test --out ../docs/results/performance.json $(ARGS)

llm-api:          ## API with local Ollama as the OpenAI-compatible provider (run instead of `make api`)
	cd backend && EVALFORGE_OPENAI_BASE_URL=http://localhost:11434/v1 EVALFORGE_LLM_TIMEOUT_SECONDS=180 uv run uvicorn evalforge.main:app --port 8000

llm-benchmark:    ## Real-LLM benchmark through the API (LLM_MODEL=openai:qwen2.5:3b; add ARGS=--quick for a smoke run)
	cd backend && uv run python -m evalforge.benchmarks.llm_benchmark --model $(LLM_MODEL) --out ../docs/results $(ARGS)

e2e:              ## Browser smoke test + real user operations (API + web running, demo data loaded)
	cd frontend && npm run e2e && npm run e2e:ops

test:
	cd backend && uv run pytest
	cd frontend && npm run typecheck && npm test

lint:
	cd backend && uv run ruff check evalforge tests && uv run ruff format --check evalforge tests

compose-up:
	GIT_COMMIT=$$(git rev-parse HEAD 2>/dev/null || echo unknown) docker compose up --build -d

observability:    ## full stack + OTel Collector, Jaeger, Prometheus, Grafana
	docker compose --profile observability up --build -d

compose-down:
	docker compose down

clean:
	rm -rf backend/var frontend/.next
