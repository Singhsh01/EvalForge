import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export const cn = (...c: ClassValue[]) => twMerge(clsx(c));

export const ALGO_LABEL: Record<string, string> = {
  logistic_regression: "Logistic Regression",
  random_forest: "Random Forest",
  gradient_boosting: "Hist. Gradient Boosting",
};

export const METRIC_LABEL: Record<string, string> = {
  accuracy: "Accuracy", balanced_accuracy: "Balanced acc.", precision: "Precision",
  recall: "Recall", f1: "F1", f1_weighted: "F1 (weighted)", roc_auc: "ROC-AUC",
  roc_auc_ovr: "ROC-AUC (OvR)", log_loss: "Log loss", cv_f1_macro_mean: "CV F1 mean",
  cv_f1_macro_std: "CV F1 std", training_seconds: "Train time (s)",
  latency_single_p50_ms: "Latency p50 (ms)", latency_single_p95_ms: "Latency p95 (ms)",
  latency_batch_ms_per_row: "Batch ms/row", generalization_gap_f1: "Train–test F1 gap",
  train_accuracy: "Train accuracy", train_f1: "Train F1",
};

export const QUALITY_METRICS = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "roc_auc_ovr"];

export function num(v: number | null | undefined, digits = 4): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  if (Math.abs(v) >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
  return v.toFixed(digits);
}

export function bytes(n: number): string {
  const u = ["B", "KB", "MB", "GB"]; let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(i ? 1 : 0)} ${u[i]}`;
}

export function ago(iso: string): string {
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return new Date(iso).toLocaleDateString();
}
