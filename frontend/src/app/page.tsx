"use client";

import { ForgeLine, type Stage } from "@/components/forge-line";
import { PageHeader } from "@/components/shell";
import { Button, Empty, ErrorNote, Panel, Stat, Status } from "@/components/ui";
import { api } from "@/lib/api";
import { ago, ALGO_LABEL, num } from "@/lib/format";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { useRouter } from "next/navigation";

const ACTION_LABEL: Record<string, string> = {
  "dataset.created": "Dataset registered",
  "dataset_version.created": "Dataset version validated",
  "run.launched": "Training run launched",
  "run.completed": "Training run logged to MLflow",
};

export default function OverviewPage() {
  const qc = useQueryClient();
  const router = useRouter();
  const ov = useQuery({ queryKey: ["overview"], queryFn: api.overview, refetchInterval: 5000 });
  const demo = useMutation({
    mutationFn: api.loadDemo,
    onSuccess: (r) => { qc.invalidateQueries(); router.push(`/datasets/${r.dataset_id}`); },
  });

  const c = ov.data?.counts;
  const hasRuns = (c?.runs_by_status?.succeeded ?? 0) > 0;
  const stages: Stage[] = [
    { key: "data", label: "Data", detail: c ? `${c.dataset_versions} validated versions` : "Upload & validate", state: c?.dataset_versions ? "live" : "ready" },
    { key: "train", label: "Training", detail: "3 algorithms, CPU", state: c?.runs ? "live" : "ready" },
    { key: "track", label: "Tracking", detail: "MLflow runs & models", state: hasRuns ? "live" : "ready" },
    { key: "eval", label: "Evaluation", detail: "Fixed holdout + re-scoring", state: hasRuns ? "live" : "ready" },
    { key: "registry", label: "Registry", detail: "Aliases & approvals", state: "planned" },
    { key: "deploy", label: "Deployment", detail: "Serving & rollback", state: "planned" },
    { key: "monitor", label: "Monitoring", detail: "Latency, errors, quality", state: "planned" },
    { key: "drift", label: "Drift", detail: "Evidently reports", state: "planned" },
    { key: "retrain", label: "Retraining", detail: "Prefect flow", state: "planned" },
    { key: "reeval", label: "Re-evaluation", detail: "Quality gates", state: "planned" },
  ];

  return (
    <>
      <PageHeader
        title="Operations overview"
        description="Every number on this page is computed from runs and datasets stored by the platform. Stages light up once they hold real records."
        action={c && c.datasets === 0 ? (
          <Button onClick={() => demo.mutate()} loading={demo.isPending}>Load the breast-cancer demo dataset</Button>
        ) : (
          <Link href="/experiments"><Button>Launch a training run</Button></Link>
        )}
      />
      <ErrorNote error={ov.error ?? demo.error} />

      <Panel className="mb-6 px-6 py-6" title="Model lifecycle"
        subtitle="Filled: running with data · ringed: built, waiting for data · dim: scheduled for a later phase">
        <ForgeLine stages={stages} />
      </Panel>

      <div className="mb-6 grid grid-cols-2 gap-4 md:grid-cols-4">
        <Panel><Stat label="Datasets" value={c?.datasets ?? "—"} sub={`${c?.dataset_versions ?? 0} versions`} /></Panel>
        <Panel><Stat label="Training runs" value={c?.runs ?? "—"} sub={`${c?.runs_by_status?.succeeded ?? 0} succeeded · ${c?.runs_by_status?.failed ?? 0} failed`} /></Panel>
        <Panel><Stat label="Active jobs" value={c?.active_jobs ?? "—"} sub="queued or running" /></Panel>
        <Panel><Stat label="Compute spent" value={ov.data ? `${num(ov.data.total_training_seconds, 1)} s` : "—"} sub="total model fitting time" /></Panel>
      </div>

      <div className="grid gap-6 xl:grid-cols-5">
        <Panel className="xl:col-span-3" title="Best model per dataset" subtitle="Ranked by holdout F1 on the dataset's fixed split">
          {ov.data?.best_by_dataset.length ? (
            <table className="w-full text-sm">
              <thead className="text-left text-xs text-white/45">
                <tr><th className="pb-2 font-normal">Dataset</th><th className="pb-2 font-normal">Algorithm</th>
                  <th className="pb-2 text-right font-normal">F1</th><th className="pb-2 text-right font-normal">ROC-AUC</th>
                  <th className="pb-2 text-right font-normal">Runs</th></tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {ov.data.best_by_dataset.map((b) => (
                  <tr key={b.dataset}>
                    <td className="py-2.5 text-white">{b.dataset} <span className="text-white/40">v{b.dataset_version}</span></td>
                    <td><Link className="text-ember hover:underline" href={`/experiments/${b.run_id}`}>{ALGO_LABEL[b.algorithm] ?? b.algorithm}</Link></td>
                    <td className="text-right tabular-nums">{num(b.f1)}</td>
                    <td className="text-right tabular-nums">{num(b.roc_auc)}</td>
                    <td className="text-right tabular-nums text-white/60">{b.n_runs}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : <Empty title="No trained models yet">Upload a dataset or load the demo, then launch a run in the Experiment lab.</Empty>}
        </Panel>

        <Panel className="xl:col-span-2" title="Recent activity" subtitle="From the audit log">
          {ov.data?.recent_activity.length ? (
            <ul className="space-y-3">
              {ov.data.recent_activity.slice(0, 8).map((a, i) => (
                <li key={i} className="flex items-baseline justify-between gap-3 text-sm">
                  <span className="text-white/80">{ACTION_LABEL[a.action] ?? a.action}
                    {typeof a.data.f1 === "number" && <span className="text-white/45"> · F1 {num(a.data.f1 as number)}</span>}
                    {typeof a.data.algorithm === "string" && <span className="text-white/45"> · {ALGO_LABEL[a.data.algorithm as string]}</span>}
                  </span>
                  <span className="shrink-0 text-xs text-white/35">{ago(a.created_at)}</span>
                </li>
              ))}
            </ul>
          ) : <Empty title="Nothing has happened yet" />}
        </Panel>

        <Panel className="xl:col-span-5" title="Recent runs">
          {ov.data?.recent_runs.length ? (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[640px] text-sm">
                <thead className="text-left text-xs text-white/45"><tr>
                  <th className="pb-2 font-normal">Run</th><th className="pb-2 font-normal">Algorithm</th>
                  <th className="pb-2 font-normal">Dataset</th><th className="pb-2 font-normal">Status</th>
                  <th className="pb-2 text-right font-normal">F1</th><th className="pb-2 text-right font-normal">Started</th></tr></thead>
                <tbody className="divide-y divide-white/5">
                  {ov.data.recent_runs.map((r) => (
                    <tr key={r.id}>
                      <td className="py-2.5"><Link className="font-mono text-xs text-ember hover:underline" href={`/experiments/${r.id}`}>{r.id.slice(0, 8)}</Link></td>
                      <td>{ALGO_LABEL[r.algorithm] ?? r.algorithm}</td>
                      <td className="text-white/70">{r.dataset} v{r.dataset_version}</td>
                      <td><Status value={r.status} /></td>
                      <td className="text-right tabular-nums">{num(r.f1)}</td>
                      <td className="text-right text-xs text-white/40">{ago(r.created_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : <Empty title="No runs yet" />}
        </Panel>
      </div>
    </>
  );
}
