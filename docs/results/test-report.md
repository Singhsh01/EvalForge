# Test report

Branch `feature/audit-hardening`, run on 2026-11-01 in a Linux dev container
(Python 3.13 via uv, pytest 9.2.0, Node 22.23, PostgreSQL 16). Every result below was produced
by the command next to it. Nothing is estimated.

| Check | Command | Result |
|---|---|---|
| Backend tests on SQLite | `cd backend && uv run pytest` | **97 passed**, 0 failed, 0 skipped (172 s) |
| Backend tests on PostgreSQL 16 | `EVALFORGE_TEST_DATABASE_URL=postgresql+psycopg://…/evalforge_test uv run pytest` | **97 passed**, 0 failed, 0 skipped (183 s) |
| Lint / format / static security | `ruff check` · `ruff format --check` · `bandit` | pass · 72 files formatted · no findings |
| Migration drift (ORM vs migrations 0001–0005) | CI script, on SQLite and PostgreSQL | in sync on both |
| Frontend type check / component tests / build | `npx tsc --noEmit` · `npm test` · `npm run build` | pass · **24 passed** · pass (19 routes) |
| Dependency audit | `npm audit` (all deps) | 0 vulnerabilities |
| Browser e2e smoke | `npm run e2e` | **pass**: sign-in, 15 module pages, live Copilot investigation, auth redirect |
| Browser e2e operations | `npm run e2e:ops`, fresh database, default rate limits | **10/10**: two-person production rollback across two sessions, staging deploy + rollback, viewer read-only |
| Enterprise case studies through the API | `make demo-enterprise` | pass; same point results as before the audit |

Previous release (`feature/enterprise-ai-operations`): 55/55 backend tests on both databases, no
component tests, smoke e2e only. The audit added 42 backend tests (security, Copilot robustness,
resilience, statistics, benchmark harness), 24 component tests and the operations e2e. See
[the audit report](../audit/AUDIT-REPORT.md).

## Case-study numbers

Identical on both profiles; source: [enterprise-demo.json](enterprise-demo.json).

| Case | Measured |
|---|---|
| 1 · ML lifecycle | Champion holdout F1 0.951; challenger Δ −0.001 vs champion; canary received 101 of 400 requests (25 % target); canary rolled back |
| 3 · Drift incident | No drift on in-distribution traffic. After a calibration shift, F1 0.951 → 0.869 (max PSI 2.29), so the verdict was `performance_degradation`. Retrained candidate scored 0.949 vs the champion's 0.922 on the v2 holdout. F1 0.992 on recalibrated traffic after release |
| 2 · Agent regression | Copilot investigation suite pass rate 100 % → 60 % (tool disabled) → 100 % (fixed config) |

## Scope of these results

- **Small holdout.** The breast-cancer holdout has 114 rows, so F1 differences under about
  0.02 are within noise.
- **No live-LLM run.** LLM provider calls, prompt evaluation and the LLM judge are tested
  against a deterministic test double. No API key was available, so there are no live-model
  numbers.
- **Docker images not built locally.** Docker Hub is unreachable from the dev container. CI
  builds the images.
- **Higgsfield workflow: guards only.** The scripts' guards were exercised by hand: missing
  CLI, generation without `--approved-credits`, and a cost call without login (fails, nothing
  charged). The optimizer was checked on a locally synthesised image, which was not committed.
  No Higgsfield asset has been generated.
