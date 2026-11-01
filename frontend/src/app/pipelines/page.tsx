"use client";

import { Dag } from "@/components/dag";
import { PageHeader } from "@/components/shell";
import { Button, Empty, ErrorNote, Field, inputCls, JsonBlock, Mono, Panel, Status, Th } from "@/components/ui";
import { api, type J } from "@/lib/api";
import { ALGO_LABEL, ago, cn } from "@/lib/format";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";

export default function PipelinesPage() { return <Suspense><Pipelines /></Suspense>; }

function Pipelines() {
  const qc = useQueryClient();
  const params = useSearchParams();
  const defs = useQuery({ queryKey: ["pipeline-defs"], queryFn: api.pipelineDefs });
  const runs = useQuery({ queryKey: ["pipeline-runs"], queryFn: api.pipelineRuns,
    refetchInterval: (q) => (q.state.data?.some((r) => ["queued", "running"].includes(r.status)) ? 1500 : 8000) });
  const datasets = useQuery({ queryKey: ["datasets"], queryFn: api.datasets });
  const schedules = useQuery({ queryKey: ["schedules"], queryFn: api.schedules });
  const [sel, setSel] = useState<string | null>(params.get("run"));
  const [step, setStep] = useState<string | undefined>();
  useEffect(() => { if (!sel && runs.data?.length) setSel(runs.data[0].id); }, [runs.data, sel]);
  const run = runs.data?.find((r) => r.id === sel);
  const versions = (datasets.data ?? []).flatMap((d) => d.versions.filter((v) => v.status === "validated").map((v) => ({ ...v, name: d.name })));
  const [form, setForm] = useState<J>({ algorithm: "logistic_regression", cv_folds: 5, deploy_to_staging: true, request_promotion: true });
  const [interval, setInterval_] = useState(1440);
  useEffect(() => { if (!form.dataset_version_id && versions.length) setForm((f) => ({ ...f, dataset_version_id: versions.at(-1)!.id })); }, [versions, form.dataset_version_id]);

  const trigger = useMutation({ mutationFn: () => api.triggerPipeline(form), onSuccess: (r: J) => { setSel(r.id); qc.invalidateQueries(); } });
  const resume = useMutation({ mutationFn: (id: string) => api.resumePipeline(id), onSuccess: () => qc.invalidateQueries() });
  const schedule = useMutation({ mutationFn: () => api.createSchedule(form, interval), onSuccess: () => qc.invalidateQueries() });
  const toggle = useMutation({ mutationFn: (id: string) => api.toggleSchedule(id), onSuccess: () => qc.invalidateQueries() });
  const def = defs.data?.[0];
  const nodes = (run?.steps ?? def?.steps ?? []).map((s: J) => ({ ...s, depends_on: s.depends_on ?? [] }));
  const st = run?.steps?.find((s: J) => s.name === step);

  return (
    <>
      <PageHeader title="Pipeline studio"
        description="Continuous training as a persisted DAG: validate → train → evaluate → register → quality gate → staging → promotion request. Steps retry transient failures, failed runs resume from the failing step, and every run records its lineage." />
      <ErrorNote error={runs.error ?? trigger.error ?? resume.error ?? schedule.error} />
      <div className="grid gap-6 xl:grid-cols-3">
        <Panel className="xl:col-span-2" title={run ? `Run ${run.id.slice(0, 8)}` : "continuous_training"}
          subtitle={run ? <span className="flex flex-wrap items-center gap-2"><Status value={run.status} /> {run.trigger} by {run.triggered_by} · {ago(run.created_at)} · config <Mono>{run.config_hash.slice(0, 10)}</Mono></span> : "Pipeline definition"}
          action={run?.status === "failed" ? <Button onClick={() => resume.mutate(run.id)} loading={resume.isPending}>Resume from failed step</Button> : null}>
          <Dag nodes={nodes} onSelect={setStep} selected={step} />
          {run?.error && <p className="mt-3 text-sm text-fail">{run.error}</p>}
          {st && (
            <div className="mt-4 border-t border-white/5 pt-4">
              <div className="mb-2 text-sm text-white">{st.name.replaceAll("_", " ")} <span className="text-white/45">· attempts {st.attempts}</span></div>
              {st.error && <p className="mb-2 text-sm text-fail">{st.error}</p>}
              <JsonBlock value={st.output} />
            </div>
          )}
          {run && (
            <div className="mt-4 border-t border-white/5 pt-4">
              <div className="mb-2 text-sm font-medium text-white">Lineage</div>
              <dl className="grid grid-cols-2 gap-x-6 gap-y-1 text-xs sm:grid-cols-3">
                {Object.entries(run.lineage ?? {}).map(([k, v]) => (
                  <div key={k}><dt className="text-white/40">{k}</dt><dd className="truncate font-mono text-white/75" title={String(v)}>{String(v)}</dd></div>
                ))}
              </dl>
            </div>
          )}
        </Panel>
        <Panel title="Run the pipeline">
          <div className="space-y-3">
            <Field label="Dataset version">
              <select className={inputCls} value={form.dataset_version_id ?? ""} onChange={(e) => setForm({ ...form, dataset_version_id: e.target.value })}>
                {versions.map((v) => <option key={v.id} value={v.id}>{v.name} v{v.version}</option>)}
              </select>
            </Field>
            <Field label="Algorithm">
              <select className={inputCls} value={form.algorithm} onChange={(e) => setForm({ ...form, algorithm: e.target.value })}>
                {Object.entries(ALGO_LABEL).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
              </select>
            </Field>
            <label className="flex items-center gap-2 text-sm text-white/70"><input type="checkbox" checked={form.deploy_to_staging} onChange={(e) => setForm({ ...form, deploy_to_staging: e.target.checked })} /> Deploy to staging after the gate</label>
            <label className="flex items-center gap-2 text-sm text-white/70"><input type="checkbox" checked={form.request_promotion} onChange={(e) => setForm({ ...form, request_promotion: e.target.checked })} /> Request promotion approval</label>
            <Button className="w-full" disabled={!form.dataset_version_id} loading={trigger.isPending} onClick={() => trigger.mutate()}>Run pipeline</Button>
            <div className="flex items-end gap-2 border-t border-white/5 pt-3">
              <Field label="Schedule every (minutes)"><input type="number" min={5} className={inputCls} value={interval} onChange={(e) => setInterval_(Number(e.target.value))} /></Field>
              <Button variant="outline" onClick={() => schedule.mutate()} loading={schedule.isPending}>Add schedule</Button>
            </div>
            {schedules.data?.map((s) => (
              <div key={s.id} className="flex items-center justify-between text-xs text-white/60">
                <span>Every {s.interval_minutes} min · last {s.last_run_at ? ago(s.last_run_at) : "never"}</span>
                <button className="text-ember" onClick={() => toggle.mutate(s.id)}>{s.enabled ? "Pause" : "Resume"}</button>
              </div>
            ))}
          </div>
        </Panel>
      </div>
      <Panel className="mt-6" title="Run history">
        {!runs.data?.length ? <Empty title="No pipeline runs yet" /> : (
          <table className="w-full text-sm">
            <thead><tr><Th>Run</Th><Th>Status</Th><Th>Trigger</Th><Th>Model</Th><Th>Failed step</Th><Th right>Started</Th></tr></thead>
            <tbody className="divide-y divide-white/5">
              {runs.data.map((r) => (
                <tr key={r.id} className={cn("cursor-pointer", r.id === sel && "bg-white/[0.04]")} onClick={() => { setSel(r.id); setStep(undefined); }}>
                  <td className="py-2 font-mono text-xs text-ember">{r.id.slice(0, 8)}</td>
                  <td><Status value={r.status} /></td>
                  <td className="text-white/60">{r.trigger}</td>
                  <td className="text-white/70">{r.lineage?.name ? `${r.lineage.name} v${r.lineage.version}` : "—"}</td>
                  <td className="text-fail">{r.steps.find((s: J) => s.status === "failed")?.name ?? ""}</td>
                  <td className="text-right text-xs text-white/45">{ago(r.created_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Panel>
    </>
  );
}
