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
            runs_by_status: Record<string, number>; active_jobs: number };
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

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, { ...init, cache: "no-store" });
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

export const api = {
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
