# Database migrations

* Alembic scripts live in `backend/evalforge/migrations/versions` and run automatically on API
  and worker start-up (`upgrade_to_head()`), against SQLite (lite) or PostgreSQL (full).
* `0001_initial_schema` — datasets, versions, jobs, training runs, evaluations, audit events.
* `0002_enterprise_operations` — users, tokens, model versions, approvals, deployments,
  prediction logs, pipeline runs/steps/schedules, prompts/versions, eval suites/runs/case
  results, LLM calls, spans, drift reports, incidents/events, agent configs, Copilot sessions;
  adds the `pipeline` and `eval_suite` job kinds (PostgreSQL `ALTER TYPE … ADD VALUE`).
* New status values are stored as strings validated in the service layer, so adding a state
  does not require an enum migration.
* **Drift check:** CI upgrades a fresh database and asserts that the ORM metadata and the
  migrations produce no autogenerate diff.
* Create a migration after changing `models.py`:
  ```bash
  cd backend
  uv run python -c "from alembic import command; from evalforge.migrations import alembic_config as c; command.revision(c(), message='describe change', autogenerate=True)"
  ```
  Review the generated script (enum and SQLite batch operations especially) before committing.
* Found during this work: a 17-character status (`awaiting_approval`) in a VARCHAR(16) column
  passed on SQLite (no length enforcement) and failed on PostgreSQL — why the test suite runs on both.
