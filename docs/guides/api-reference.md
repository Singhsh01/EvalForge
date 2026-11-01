# API reference

Generated from the running API (`GET /openapi.json`, 83 operations). Interactive docs: `http://localhost:8000/docs`.

All endpoints except `/api/health*`, `/api/meta`, `/api/auth/login` and `/metrics` require `Authorization: Bearer <token>` from `POST /api/auth/login`. Required roles are listed in [security.md](security.md#roles).

## Auth

| Method | Path | Operation |
|---|---|---|
| `POST` | `/api/auth/login` | Login |
| `POST` | `/api/auth/logout` | Logout |
| `GET` | `/api/auth/me` | Me |
| `GET` | `/api/auth/users` | Users |
| `POST` | `/api/auth/users` | Create User |
| `POST` | `/api/auth/tokens/revoke` | Revoke |

## System

| Method | Path | Operation |
|---|---|---|
| `GET` | `/api/health` | Health |
| `GET` | `/api/health/ready` | Ready |
| `GET` | `/api/meta` | Meta |
| `GET` | `/api/overview` | Overview |
| `POST` | `/api/demo/breast-cancer` | Load Demo |

## Datasets

| Method | Path | Operation |
|---|---|---|
| `GET` | `/api/datasets` | List Datasets |
| `POST` | `/api/datasets` | Create Dataset |
| `GET` | `/api/datasets/{dataset_id}` | Get Dataset |
| `POST` | `/api/datasets/{dataset_id}/versions` | Add Version |
| `GET` | `/api/datasets/{dataset_id}/versions/{version}` | Get Version |
| `GET` | `/api/datasets/{dataset_id}/versions/{version}/preview` | Preview |

## Experiments

| Method | Path | Operation |
|---|---|---|
| `POST` | `/api/runs` | Launch Run |
| `GET` | `/api/runs` | List Runs |
| `GET` | `/api/runs/compare` | Compare Runs |
| `GET` | `/api/runs/{run_id}` | Get Run |
| `POST` | `/api/runs/{run_id}/evaluations` | Launch Evaluation |
| `GET` | `/api/runs/{run_id}/evaluations` | List Evaluations |
| `GET` | `/api/jobs/{job_id}` | Get Job |

## Registry & Deployment

| Method | Path | Operation |
|---|---|---|
| `GET` | `/api/registry/models` | List Models |
| `POST` | `/api/registry/versions` | Register |
| `GET` | `/api/registry/versions/{mv_id}` | Get Version |
| `POST` | `/api/registry/versions/{mv_id}/gate` | Run Gate |
| `GET` | `/api/registry/versions/{mv_id}/card` | Card |
| `POST` | `/api/registry/versions/{mv_id}/request-promotion` | Request Promotion |
| `GET` | `/api/approvals` | List Approvals |
| `POST` | `/api/approvals/{approval_id}/decision` | Decide Approval |
| `GET` | `/api/deployments` | List Deployments |
| `POST` | `/api/deployments` | Create Deployment |
| `POST` | `/api/deployments/rollback` | Rollback |
| `POST` | `/api/deployments/promote-canary` | Promote Canary |
| `POST` | `/api/deployments/{dep_id}/health` | Recheck Health |
| `POST` | `/api/serve/{model_name}/predict` | Predict |
| `POST` | `/api/serve/feedback` | Feedback |

## Pipelines

| Method | Path | Operation |
|---|---|---|
| `GET` | `/api/pipelines` | Definitions |
| `GET` | `/api/pipelines/runs` | List Runs |
| `POST` | `/api/pipelines/runs` | Trigger |
| `GET` | `/api/pipelines/runs/{run_id}` | Get Run |
| `POST` | `/api/pipelines/runs/{run_id}/resume` | Resume |
| `GET` | `/api/pipelines/schedules` | Schedules |
| `POST` | `/api/pipelines/schedules` | Create Schedule |
| `POST` | `/api/pipelines/schedules/{schedule_id}/toggle` | Toggle |
| `POST` | `/api/pipelines/schedules/tick` | Tick |

## Llmops

| Method | Path | Operation |
|---|---|---|
| `GET` | `/api/llm/providers` | List Providers |
| `GET` | `/api/prompts` | List Prompts |
| `POST` | `/api/prompts` | Create Prompt |
| `POST` | `/api/prompts/{name}/versions` | Add Version |
| `GET` | `/api/prompts/{name}/diff` | Diff |
| `POST` | `/api/prompts/versions/{pv_id}/request-release` | Request Release |
| `GET` | `/api/evals/suites` | List Suites |
| `POST` | `/api/evals/suites` | Create Suite |
| `GET` | `/api/evals/suites/{suite_id}` | Get Suite |
| `POST` | `/api/evals/runs` | Launch |
| `GET` | `/api/evals/runs` | List Runs |
| `GET` | `/api/evals/runs/{run_id}` | Get Run |
| `GET` | `/api/evals/compare` | Compare |
| `GET` | `/api/evals/runs/{run_id}/judge-calibration` | Calibration |

## Observability

| Method | Path | Operation |
|---|---|---|
| `GET` | `/api/traces` | List Traces |
| `GET` | `/api/traces/{trace_id}` | Get Trace |
| `GET` | `/api/observability/summary` | Summary |
| `GET` | `/api/analytics/cost-performance` | Cost Performance |
| `GET` | `/api/audit` | Audit |

## Monitoring

| Method | Path | Operation |
|---|---|---|
| `POST` | `/api/drift/reports` | Create Report |
| `GET` | `/api/drift/reports` | List Reports |
| `GET` | `/api/drift/reports/{report_id}` | Get Report |
| `GET` | `/api/incidents` | List Incidents |
| `GET` | `/api/incidents/{incident_id}` | Get Incident |
| `POST` | `/api/incidents/{incident_id}/status` | Set Status |
| `POST` | `/api/incidents/{incident_id}/retraining-plan` | Plan |
| `POST` | `/api/incidents/{incident_id}/request-retraining` | Request Retraining |

## Copilot

| Method | Path | Operation |
|---|---|---|
| `GET` | `/api/copilot/tools` | Tools |
| `GET` | `/api/copilot/configs` | Configs |
| `POST` | `/api/copilot/configs` | Create Config |
| `GET` | `/api/copilot/sessions` | Sessions |
| `POST` | `/api/copilot/sessions` | Ask |
| `GET` | `/api/copilot/sessions/{sid}` | Session |
| `POST` | `/api/copilot/sessions/{sid}/cancel` | Cancel |
| `POST` | `/api/incidents/{incident_id}/investigate` | Investigate |
