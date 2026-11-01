# AI MLOps Copilot guide

Ask in the UI (*MLOps Copilot*) or `POST /api/copilot/sessions {"question": "...", "wait": true}`.

| Question type | Intent | Tools used (rules planner) |
|---|---|---|
| "Why did my latest model perform worse?" | model_regression | query_mlflow_runs → compare_runs → diff_dataset_versions |
| "Which dataset changes are responsible…?" | dataset_changes | list_dataset_versions, diff_dataset_versions, query_mlflow_runs |
| "Compare the last five experiments and recommend…" | recommend | query_mlflow_runs, recommend_candidate (Pareto: CV F1, holdout F1, p95) |
| "Why did this AI agent fail its evaluation?" | agent_eval_failure | list_eval_runs → compare_eval_runs → diff_agent_configs |
| "Investigate the latest deployment failure." | deployment_failure | get_deployments, get_model_versions, search_traces |
| "Investigate the drift incident…" | drift | list_incidents, get_drift_reports |
| pipeline failures | pipeline_failure | get_pipeline_runs |

* **Planners:** `rules` (default, deterministic, offline) or `llm:<provider>:<model>` set in an
  agent config. The LLM planner chooses tools turn by turn via a JSON protocol and falls back
  to rules when no provider is configured.
* **Evidence:** a finding is kept only if every identifier it cites appeared in a tool result.
* **Budgets:** `max_steps` / `timeout_seconds` per config, capped by
  `EVALFORGE_COPILOT_MAX_STEPS` / `EVALFORGE_COPILOT_TIMEOUT_SECONDS`; sessions can be cancelled.
* **Actions:** `request_evaluation` is the only action tool; it requires `allow_actions` and
  files an approval. Retraining, promotion and deployment are never executed by the agent.
* **Incidents:** *Investigate with Copilot* stores findings, hypotheses and recommendations on
  the incident and links the session.
* **Evaluate the agent:** create an `agent` suite and run it against any config version
  (case study 2).
