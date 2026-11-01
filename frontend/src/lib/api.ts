// Typed client for the EvalForge FastAPI backend (same-origin via Next rewrites).

export type Check = {
  id: string; title: string; severity: "error" | "warning" | "info";
  passed: boolean; message: string; details: Record<string, unknown>;
};

export type VersionSummary = {
  id: string; version: number; sha256: string; source_filename: string; file_format: string;
  size_bytes: number; n_rows: number; n_columns: number; task_type: string | null;
  status: "validated" | "rejected"; note: string; created_at: string;
  checks_failed: number; checks_warned: number;
};

export type ColumnProfile = {
  kind: string; missing: number; missing_ratio: number; unique: number;
  stats?: Record<string, number | null>;
  histogram?: { counts: number[]; edges: (number | null)[] };
  top_values?: { value: string; count: number }[];
};

export type VersionDetail = VersionSummary & {
  dataset_id: string;
  schema: { name: string; dtype: string; kind: string; nullable: boolean }[];
  profile: {
    n_rows: number; n_columns: number; duplicate_rows: number; memory_bytes: number;
    columns: Record<string, ColumnProfile>;
    target?: { column: string; classes: { label: string; count: number }[] };
  };
  quality_checks: Check[];
  split: { strategy: string; test_size: number; seed: number; n_train: number; n_test: number;
           train_test_overlap_rows: number } | null;
};

export type Dataset = {
  id: string; name: string; description: string; target_column: string; created_at: string;
  versions: VersionSummary[];
};

export type Job = {
  id: string; kind: string; status: "queued" | "running" | "succeeded" | "failed";
  stage: string; progress: number; error: string | null;
  logs: { ts: string; stage: string; message: string; traceback?: string }[];
  result: Record<string, unknown> | null; created_at: string;
  started_at: string | null; finished_at: string | null;
};

export type Run = {
  id: string; job_id: string; dataset_version_id: string; dataset_name: string | null;
  dataset_version_number: number | null; algorithm: string;
  hyperparameters: Record<string, unknown>; seed: number; cv_folds: number;
  status: Job["status"]; metrics: Record<string, number> | null;
  mlflow_experiment_id: string | null; mlflow_run_id: string | null; model_uri: string | null;
  training_seconds: number | null; git_commit: string | null; error: string | null;
  created_at: string; finished_at: string | null;
};

export type RunDetail = Run & {
  report: {
    labels: string[]; positive_label: string | null; averaging: string; n_samples: number;
    confusion_matrix: number[][];
    per_class: Record<string, Record<string, number>>;
    roc_curve?: { fpr: number[]; tpr: number[] };
    feature_importance: { feature: string; importance: number; std: number }[];
  } | null;
  environment: Record<string, unknown> | null;
  feature_columns: string[] | null;
  dropped_columns: { column: string; reason: string }[] | null;
  job: Job | null;
};

export type Evaluation = {
  id: string; run_id: string; dataset_version_id: string; split: string;
  status: Job["status"]; metrics: Record<string, number> | null; error: string | null;
  report: { unseen_labels?: string[]; n_samples?: number } | null; created_at: string;
};

export type Param = {
  name: string; type: "int" | "float" | "choice"; default: unknown;
  min: number | null; max: number | null; choices: unknown[] | null;
  nullable: boolean; log_scale: boolean; help: string;
};
export type Algorithm = { key: string; label: string; description: string; tags: string[]; params: Param[] };
export type Meta = {
  version: string; executor: string; database: string; mlflow_tracking_uri: string;
  algorithms: Algorithm[]; limits: { max_upload_mb: number; max_rows: number; max_columns: number };
};

export type Compare = {
  runs: Run[]; metric_keys: string[]; param_keys: string[]; differing_params: string[];
  same_holdout: boolean; best: Record<string, string[]>; warnings: string[];
};

export type Overview = {
  counts: { datasets: number; dataset_versions: number; runs: number;
            runs_by_status: Record<string, number>; active_jobs: number;
            model_versions?: number; champions?: number; active_deployments?: number;
            production_deployments?: number; rollbacks?: number; pipeline_runs?: number;
            drift_reports?: number; incidents_open?: number; incident_retrains?: number;
            eval_runs?: number; pending_approvals?: number; copilot_sessions?: number };
  total_training_seconds: number;
  best_by_dataset: { dataset: string; run_id: string; algorithm: string; dataset_version: number;
                     f1: number | null; roc_auc: number | null; n_runs: number }[];
  recent_runs: { id: string; algorithm: string; status: string; dataset: string;
                 dataset_version: number; f1: number | null; created_at: string }[];
  recent_activity: { action: string; entity_type: string; entity_id: string;
                     data: Record<string, unknown>; created_at: string }[];
};

export class ApiError extends Error {
  constructor(public status: number, message: string) { super(message); }
}

export const TOKEN_KEY = "evalforge.token";
export const USER_KEY = "evalforge.user";

export function getToken(): string | null {
  try { return localStorage.getItem(TOKEN_KEY); } catch { return null; }
}
export function getUser(): { username: string; role: string } | null {
  try { return JSON.parse(localStorage.getItem(USER_KEY) ?? "null"); } catch { return null; }
}
export function setSession(token: string | null, user?: { username: string; role: string }) {
  try {
    if (token) { localStorage.setItem(TOKEN_KEY, token); localStorage.setItem(USER_KEY, JSON.stringify(user)); }
    else { localStorage.removeItem(TOKEN_KEY); localStorage.removeItem(USER_KEY); }
  } catch { /* storage unavailable: session lasts for this page only */ }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const token = getToken();
  const headers = new Headers(init?.headers);
  if (token) headers.set("authorization", `Bearer ${token}`);
  const res = await fetch(path, { ...init, headers, cache: "no-store" });
  if (res.status === 401 && typeof window !== "undefined" && !path.endsWith("/auth/login")) {
    setSession(null);
    if (!window.location.pathname.startsWith("/login")) window.location.href = "/login";
  }
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const body = await res.json();
      const d = body.detail;
      msg = typeof d === "string" ? d : Array.isArray(d) ? d.map((e) => e.msg).join("; ")
          : d?.message ?? JSON.stringify(d);
    } catch { /* non-JSON error */ }
    throw new ApiError(res.status, msg);
  }
  return res.json() as Promise<T>;
}

const json = (body: unknown): RequestInit => ({
  method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body),
});

/* eslint-disable @typescript-eslint/no-explicit-any */
export type J = Record<string, any>;
const get = <T = J>(p: string) => request<T>(p);
const post = <T = J>(p: string, body?: unknown) => request<T>(p, json(body ?? {}));

export const api = {
  login: (username: string, password: string) =>
    request<{ token: string; user: { username: string; role: string } }>("/api/auth/login", json({ username, password })),
  logout: () => post("/api/auth/logout"),
  me: () => get<{ username: string; role: string }>("/api/auth/me"),
  users: () => get<J[]>("/api/auth/users"),
  // registry & deployments
  models: () => get<J[]>("/api/registry/models"),
  modelVersion: (id: string) => get(`/api/registry/versions/${id}`),
  register: (run_id: string) => post("/api/registry/versions", { run_id }),
  gate: (id: string) => post(`/api/registry/versions/${id}/gate`),
  card: (id: string) => get(`/api/registry/versions/${id}/card`),
  requestPromotion: (id: string) => post(`/api/registry/versions/${id}/request-promotion`),
  approvals: (status?: string) => get<J[]>(`/api/approvals${status ? `?status=${status}` : ""}`),
  decide: (id: string, approve: boolean, reason: string) => post(`/api/approvals/${id}/decision`, { approve, reason }),
  deployments: (model?: string) => get<J[]>(`/api/deployments${model ? `?model_name=${encodeURIComponent(model)}` : ""}`),
  deploy: (model_version_id: string, environment: string, traffic_percent: number) =>
    post("/api/deployments", { model_version_id, environment, traffic_percent }),
  rollback: (model_name: string, environment: string) => post("/api/deployments/rollback", { model_name, environment }),
  promoteCanary: (model_name: string, environment: string) => post("/api/deployments/promote-canary", { model_name, environment }),
  health: (id: string) => post(`/api/deployments/${id}/health`),
  predict: (model: string, env: string, instances: J[]) => post(`/api/serve/${encodeURIComponent(model)}/predict?environment=${env}`, { instances }),
  // pipelines
  pipelineDefs: () => get<J[]>("/api/pipelines"),
  pipelineRuns: () => get<J[]>("/api/pipelines/runs"),
  pipelineRun: (id: string) => get(`/api/pipelines/runs/${id}`),
  triggerPipeline: (config: J) => post("/api/pipelines/runs", { pipeline: "continuous_training", config }),
  resumePipeline: (id: string) => post(`/api/pipelines/runs/${id}/resume`),
  schedules: () => get<J[]>("/api/pipelines/schedules"),
  createSchedule: (config: J, interval_minutes: number) => post("/api/pipelines/schedules", { config, interval_minutes }),
  toggleSchedule: (id: string) => post(`/api/pipelines/schedules/${id}/toggle`),
  // llmops
  providers: () => get("/api/llm/providers"),
  prompts: () => get<J[]>("/api/prompts"),
  createPrompt: (body: J) => post("/api/prompts", body),
  addPromptVersion: (name: string, body: J) => post(`/api/prompts/${name}/versions`, body),
  promptDiff: (name: string, a: number, b: number) => get(`/api/prompts/${name}/diff?a=${a}&b=${b}`),
  requestRelease: (pvId: string, eval_run_id: string) => post(`/api/prompts/versions/${pvId}/request-release`, { eval_run_id }),
  suites: () => get<J[]>("/api/evals/suites"),
  suite: (id: string) => get(`/api/evals/suites/${id}`),
  createSuite: (body: J) => post("/api/evals/suites", body),
  evalRuns: () => get<J[]>("/api/evals/runs"),
  evalRun: (id: string) => get(`/api/evals/runs/${id}`),
  launchEval: (suite_id: string, subject: J) => post("/api/evals/runs", { suite_id, subject }),
  compareEvals: (a: string, b: string) => get(`/api/evals/compare?a=${a}&b=${b}`),
  // observability
  traces: (q = "") => get<J[]>(`/api/traces${q}`),
  trace: (id: string) => get(`/api/traces/${id}`),
  obsSummary: (hours = 24) => get(`/api/observability/summary?hours=${hours}`),
  costPerf: () => get("/api/analytics/cost-performance"),
  audit: (q = "") => get<J[]>(`/api/audit${q}`),
  // monitoring
  driftReports: () => get<J[]>("/api/drift/reports"),
  driftReport: (id: string) => get(`/api/drift/reports/${id}`),
  createDrift: (body: J) => post("/api/drift/reports", body),
  incidents: () => get<J[]>("/api/incidents"),
  incident: (id: string) => get(`/api/incidents/${id}`),
  investigate: (id: string) => post(`/api/incidents/${id}/investigate`),
  retrainPlan: (id: string, dataset_version_id?: string) => post(`/api/incidents/${id}/retraining-plan`, { dataset_version_id }),
  requestRetraining: (id: string, body: J) => post(`/api/incidents/${id}/request-retraining`, body),
  incidentStatus: (id: string, status: string, note: string) => post(`/api/incidents/${id}/status`, { status, note }),
  // copilot
  copilotTools: () => get<J[]>("/api/copilot/tools"),
  agentConfigs: () => get<J[]>("/api/copilot/configs"),
  createAgentConfig: (config: J, note: string) => post("/api/copilot/configs", { config, note }),
  ask: (question: string, agent_config_id?: string) => post("/api/copilot/sessions", { question, agent_config_id }),
  sessions: () => get<J[]>("/api/copilot/sessions"),
  copilotSession: (id: string) => get(`/api/copilot/sessions/${id}`),
  cancelSession: (id: string) => post(`/api/copilot/sessions/${id}/cancel`),

  meta: () => request<Meta>("/api/meta"),
  overview: () => request<Overview>("/api/overview"),
  datasets: () => request<Dataset[]>("/api/datasets"),
  dataset: (id: string) => request<Dataset>(`/api/datasets/${id}`),
  version: (id: string, v: number) => request<VersionDetail>(`/api/datasets/${id}/versions/${v}`),
  preview: (id: string, v: number) =>
    request<{ columns: string[]; rows: unknown[][] }>(`/api/datasets/${id}/versions/${v}/preview?limit=25`),
  createDataset: (form: FormData) => request<Dataset>("/api/datasets", { method: "POST", body: form }),
  addVersion: (id: string, form: FormData) =>
    request<VersionDetail>(`/api/datasets/${id}/versions`, { method: "POST", body: form }),
  loadDemo: () => request<{ dataset_id: string; created: boolean }>("/api/demo/breast-cancer", { method: "POST" }),
  runs: (datasetId?: string) => request<Run[]>(`/api/runs${datasetId ? `?dataset_id=${datasetId}` : ""}`),
  run: (id: string) => request<RunDetail>(`/api/runs/${id}`),
  launchRun: (body: { dataset_version_id: string; algorithm: string;
                      hyperparameters: Record<string, unknown>; seed: number; cv_folds: number }) =>
    request<{ job: Job; run: Run }>("/api/runs", json(body)),
  compare: (ids: string[]) => request<Compare>(`/api/runs/compare?ids=${ids.join(",")}`),
  evaluations: (runId: string) => request<Evaluation[]>(`/api/runs/${runId}/evaluations`),
  evaluate: (runId: string, versionId: string) =>
    request<{ job: Job }>(`/api/runs/${runId}/evaluations`, json({ dataset_version_id: versionId })),
};
