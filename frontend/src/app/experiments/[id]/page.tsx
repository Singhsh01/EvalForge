"use client";

import { ConfusionMatrix, ImportanceBars, RocCurve } from "@/components/charts";
import { PageHeader } from "@/components/shell";
import { Button, Empty, ErrorNote, Mono, Panel, Status } from "@/components/ui";
import { api } from "@/lib/api";
import { ALGO_LABEL, METRIC_LABEL, num } from "@/lib/format";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { motion } from "motion/react";
import Link from "next/link";
import { use, useState } from "react";

const HEADLINE = ["f1", "precision", "recall", "accuracy", "roc_auc", "roc_auc_ovr", "log_loss"];

export default function RunPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const qc = useQueryClient();
  const run = useQuery({
    queryKey: ["run", id], queryFn: () => api.run(id),
    refetchInterval: (q) => (q.state.data && ["queued", "running"].includes(q.state.data.status) ? 1000 : false),
  });
  const r = run.data;
  const ds = useQuery({
    queryKey: ["dataset-of-run", r?.dataset_name],
    queryFn: async () => (await api.datasets()).find((d) => d.name === r?.dataset_name) ?? null,
    enabled: !!r?.dataset_name,
  });
  const evals = useQuery({
    queryKey: ["evals", id], queryFn: () => api.evaluations(id), enabled: r?.status === "succeeded",
    refetchInterval: (q) => (q.state.data?.some((e) => e.status === "queued" || e.status === "running") ? 1000 : false),
  });
  const [evalVersion, setEvalVersion] = useState("");
  const evaluate = useMutation({
    mutationFn: () => api.evaluate(id, evalVersion),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["evals", id] }),
  });

  if (run.error) return <ErrorNote error={run.error} />;
  if (!r) return null;
  const m = r.metrics ?? {};
  const job = r.job;
  const active = r.status === "queued" || r.status === "running";
  const otherVersions = ds.data?.versions.filter((v) => v.status === "validated") ?? [];

  return (
    <>
      <PageHeader
        title={`${ALGO_LABEL[r.algorithm] ?? r.algorithm}`}
        description={<>Run <Mono>{r.id}</Mono> on <Link className="text-ember hover:underline" href={`/datasets/${ds.data?.id ?? ""}`}>{r.dataset_name} v{r.dataset_version_number}</Link></>}
        action={<Status value={r.status} />}
      />

      {(active || r.status === "failed") && job && (
        <Panel className="mb-6" title={active ? "Training in progress" : "Run failed"} subtitle={job.error ?? `Stage: ${job.stage.replaceAll("_", " ")}`}>
          <div className="mb-4 h-1.5 overflow-hidden rounded-full bg-white/5" role="progressbar" aria-valuenow={Math.round(job.progress * 100)} aria-valuemin={0} aria-valuemax={100}>
            <motion.div className="h-full bg-ember" animate={{ width: `${job.progress * 100}%` }} transition={{ duration: 0.4 }} />
          </div>
          <ol className="space-y-1 text-xs">
            {job.logs.map((l, i) => (
              <li key={i} className="flex gap-3"><span className="w-20 shrink-0 tabular-nums text-white/35">{new Date(l.ts).toLocaleTimeString()}</span>
                <span className="text-white/70">{l.message}</span></li>
            ))}
          </ol>
          {r.status === "failed" && job.logs.at(-1)?.traceback && (
            <details className="mt-3"><summary className="cursor-pointer text-xs text-white/50">Traceback</summary>
              <pre className="mt-2 max-h-72 overflow-auto rounded-lg bg-ink-950/70 p-3 text-[11px] text-white/60">{job.logs.at(-1)!.traceback}</pre></details>
          )}
        </Panel>
      )}

      {r.status === "succeeded" && (
        <>
          <div className="mb-6 grid grid-cols-2 gap-3 sm:grid-cols-4 xl:grid-cols-7">
            {HEADLINE.filter((k) => m[k] !== undefined).map((k) => (
              <div key={k} className="glass rounded-xl px-4 py-3">
                <div className="text-xs text-white/50">{METRIC_LABEL[k]}</div>
                <div className="mt-1 text-xl font-semibold tabular-nums text-white">{num(m[k], 4)}</div>
              </div>
            ))}
          </div>
          <p className="-mt-3 mb-6 text-xs text-white/40">
            Measured on {r.report?.n_samples} holdout rows. {r.report?.averaging === "binary"
              ? <>Precision, recall and F1 treat “{r.report?.positive_label}” as the positive class.</>
              : <>Precision, recall and F1 are macro-averaged across classes.</>}
          </p>

          <div className="grid gap-6 xl:grid-cols-2">
            <Panel title="Confusion matrix" subtitle="Holdout predictions; diagonal cells are correct">
              {r.report && <ConfusionMatrix matrix={r.report.confusion_matrix} labels={r.report.labels} />}
            </Panel>
            <Panel title="ROC curve" subtitle={r.report?.roc_curve ? "Holdout, positive class probability" : undefined}>
              {r.report?.roc_curve ? <RocCurve fpr={r.report.roc_curve.fpr} tpr={r.report.roc_curve.tpr} auc={m.roc_auc} />
                : <Empty title="Not applicable">ROC curves are shown for binary targets.</Empty>}
            </Panel>
            <Panel title="Feature importance" subtitle="Permutation importance: drop in macro F1 on the holdout when a feature is shuffled (5 repeats, ± std)">
              {r.report && <ImportanceBars rows={r.report.feature_importance} />}
            </Panel>
            <Panel title="Generalisation and operations">
              <dl className="grid grid-cols-2 gap-x-6 gap-y-3 text-sm">
                {["train_f1", "generalization_gap_f1", "cv_f1_macro_mean", "cv_f1_macro_std", "training_seconds", "latency_single_p50_ms", "latency_single_p95_ms", "latency_batch_ms_per_row"]
                  .filter((k) => m[k] !== undefined).map((k) => (
                    <div key={k} className="flex justify-between gap-3 border-b border-white/5 pb-2">
                      <dt className="text-white/55">{METRIC_LABEL[k]}</dt><dd className="tabular-nums text-white">{num(m[k], 4)}</dd>
                    </div>
                  ))}
              </dl>
              <p className="mt-3 text-xs text-white/40">Latency is measured in-process on this machine (single-row predict calls over 50 holdout rows), not over the network.</p>
            </Panel>
          </div>

          <div className="mt-6 grid gap-6 xl:grid-cols-3">
            <Panel title="Evaluations" className="xl:col-span-2" subtitle="Re-score this run's MLflow model on any version of the same dataset">
              <ul className="mb-4 divide-y divide-white/5 text-sm">
                {evals.data?.map((e) => {
                  const v = ds.data?.versions.find((x) => x.id === e.dataset_version_id);
                  return (
                    <li key={e.id} className="flex flex-wrap items-center gap-x-6 gap-y-1 py-2.5">
                      <span className="w-56 text-white">{e.split === "holdout" ? "Holdout split" : "All labelled rows"} · v{v?.version}
                        {e.split === "full" && e.dataset_version_id === r.dataset_version_id &&
                          <span className="block text-xs text-warn">includes training rows, so optimistic</span>}</span>
                      <Status value={e.status} />
                      {e.metrics && ["f1", "accuracy", "roc_auc"].map((k) => e.metrics![k] !== undefined && (
                        <span key={k} className="tabular-nums text-white/70">{METRIC_LABEL[k]} {num(e.metrics![k])}</span>))}
                      {e.error && <span className="text-xs text-fail">{e.error}</span>}
                    </li>
                  );
                })}
              </ul>
              <form className="flex flex-wrap gap-2" onSubmit={(e) => { e.preventDefault(); evaluate.mutate(); }}>
                <select className="rounded-lg border border-white/10 bg-ink-950/60 px-3 py-2 text-sm" value={evalVersion} onChange={(e) => setEvalVersion(e.target.value)} required>
                  <option value="">Choose a dataset version…</option>
                  {otherVersions.map((v) => <option key={v.id} value={v.id}>v{v.version} · {v.n_rows} rows{v.id === r.dataset_version_id ? " (training version)" : ""}</option>)}
                </select>
                <Button type="submit" variant="outline" loading={evaluate.isPending} disabled={!evalVersion}>Run evaluation</Button>
              </form>
              <ErrorNote error={evaluate.error} />
            </Panel>

            <Panel title="Reproducibility">
              <dl className="space-y-2.5 text-sm">
                <Row k="MLflow run" v={<Mono>{r.mlflow_run_id}</Mono>} />
                <Row k="Model URI" v={<Mono className="break-all">{r.model_uri}</Mono>} />
                <Row k="Source commit" v={<Mono>{r.git_commit?.slice(0, 12)}</Mono>} />
                <Row k="Seed" v={r.seed} />
                <Row k="Hyperparameters" v={<Mono className="break-all">{JSON.stringify(r.hyperparameters)}</Mono>} />
                <Row k="Features" v={`${r.feature_columns?.length} used${r.dropped_columns?.length ? `, dropped ${r.dropped_columns.map((d) => `${d.column} (${d.reason})`).join(", ")}` : ""}`} />
                {r.environment && <Row k="Environment" v={<Mono>{`py ${r.environment.python} · sklearn ${r.environment.sklearn} · mlflow ${r.environment.mlflow}`}</Mono>} />}
              </dl>
            </Panel>
          </div>
        </>
      )}
    </>
  );
}

function Row({ k, v }: { k: string; v: React.ReactNode }) {
  return <div><dt className="text-xs text-white/45">{k}</dt><dd className="mt-0.5 text-white/85">{v}</dd></div>;
}
