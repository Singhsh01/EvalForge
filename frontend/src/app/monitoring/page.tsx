"use client";

import { PageHeader } from "@/components/shell";
import { Button, Empty, ErrorNote, Field, inputCls, Panel, Pill, Th } from "@/components/ui";
import { api, type J } from "@/lib/api";
import { ago, cn, num } from "@/lib/format";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

const VERDICT: Record<string, { label: string; tone: "pass" | "warn" | "fail" }> = {
  no_drift: { label: "No drift", tone: "pass" },
  statistical_drift_without_labels: { label: "Statistical drift · no labels", tone: "warn" },
  statistical_drift_performance_stable: { label: "Statistical drift · quality stable", tone: "warn" },
  performance_degradation: { label: "Performance degradation", tone: "fail" },
};

export default function MonitoringPage() {
  const qc = useQueryClient();
  const reports = useQuery({ queryKey: ["drift"], queryFn: api.driftReports });
  const models = useQuery({ queryKey: ["models"], queryFn: api.models });
  const [sel, setSel] = useState<string | null>(null);
  const [form, setForm] = useState<J>({ source: "predictions", environment: "production", hours: 24 });
  const versions = (models.data ?? []).flatMap((m) => m.versions as J[]);
  useEffect(() => { if (!form.model_version_id && versions.length) setForm((f) => ({ ...f, model_version_id: (versions.find((v) => v.is_champion) ?? versions[0]).id })); }, [versions, form.model_version_id]);
  useEffect(() => { if (!sel && reports.data?.length) setSel(reports.data[0].id); }, [reports.data, sel]);
  const detail = useQuery({ queryKey: ["drift", sel], queryFn: () => api.driftReport(sel!), enabled: !!sel });
  const create = useMutation({ mutationFn: () => api.createDrift(form), onSuccess: (r: J) => { setSel(r.id); qc.invalidateQueries(); } });
  const [feat, setFeat] = useState<string | null>(null);
  const r = detail.data;
  const f = r?.features?.find((x: J) => x.feature === feat) ?? r?.features?.[0];

  return (
    <>
      <PageHeader title="Drift monitoring"
        description="Compares a deployed model's training data with logged production inputs (or a new dataset version). Statistical drift, prediction drift and label-confirmed performance degradation are reported separately; only the last one justifies retraining on its own." />
      <ErrorNote error={reports.error ?? create.error} />
      <div className="grid gap-6 xl:grid-cols-3">
        <Panel title="New drift report">
          <div className="space-y-3">
            <Field label="Model version">
              <select className={inputCls} value={form.model_version_id ?? ""} onChange={(e) => setForm({ ...form, model_version_id: e.target.value })}>
                {versions.map((v) => <option key={v.id} value={v.id}>{v.name} v{v.version}{v.is_champion ? " (champion)" : ""}</option>)}
              </select>
            </Field>
            <Field label="Current data">
              <select className={inputCls} value={form.source} onChange={(e) => setForm({ ...form, source: e.target.value })}>
                <option value="predictions">Logged predictions</option><option value="dataset_version">A dataset version</option>
              </select>
            </Field>
            {form.source === "predictions" ? (
              <div className="grid grid-cols-2 gap-3">
                <Field label="Environment"><select className={inputCls} value={form.environment} onChange={(e) => setForm({ ...form, environment: e.target.value })}><option>production</option><option>staging</option></select></Field>
                <Field label="Window (hours)"><input type="number" className={inputCls} value={form.hours} onChange={(e) => setForm({ ...form, hours: Number(e.target.value) })} /></Field>
              </div>
            ) : (
              <Field label="Dataset version id"><input className={inputCls} value={form.dataset_version_id ?? ""} onChange={(e) => setForm({ ...form, dataset_version_id: e.target.value })} /></Field>
            )}
            <Button className="w-full" loading={create.isPending} onClick={() => create.mutate()}>Run drift analysis</Button>
          </div>
          <div className="mt-6 border-t border-white/5 pt-4">
            {!reports.data?.length ? <Empty title="No reports yet" /> : (
              <ul className="space-y-1">
                {reports.data.map((x) => (
                  <li key={x.id}><button onClick={() => setSel(x.id)} className={cn("flex w-full items-center justify-between rounded-lg px-2 py-1.5 text-left text-sm", x.id === sel ? "bg-white/[0.06]" : "hover:bg-white/[0.03]")}>
                    <span className="text-white/80">{x.model_name} · {ago(x.created_at)}</span><Pill tone={VERDICT[x.summary.verdict]?.tone ?? "warn"}>{Math.round(x.summary.drift_share * 100)}%</Pill>
                  </button></li>
                ))}
              </ul>
            )}
          </div>
        </Panel>
        {r && (
          <Panel className="xl:col-span-2" title={<span className="flex items-center gap-2">{r.model_name} <Pill tone={VERDICT[r.summary.verdict]?.tone}>{VERDICT[r.summary.verdict]?.label}</Pill></span>}
            subtitle={`Reference: dataset v${r.reference.dataset_version} training split (${r.reference.rows} rows) · current: ${r.source === "predictions" ? `${r.current.environment} predictions, last ${r.current.window_hours} h` : `dataset v${r.current.version}`} (${r.current.rows} rows)`}>
            <div className="mb-5 grid grid-cols-2 gap-3 md:grid-cols-4">
              <Box label="Features drifted" value={`${r.summary.drifted_features.length}/${r.features.length}`} />
              <Box label="Prediction drift" value={r.prediction_drift.drifted ? `yes · PSI ${num(r.prediction_drift.psi, 2)}` : `no · PSI ${num(r.prediction_drift.psi, 2)}`} />
              <Box label="Labelled samples" value={r.performance ? r.performance.labelled_samples : "none"} />
              <Box label="F1 (baseline → now)" value={r.performance ? `${num(r.performance.baseline_f1, 3)} → ${num(r.performance.current.f1, 3)}` : "unknown"} />
            </div>
            <h3 className="mb-2 text-sm font-medium text-white">Feature heatmap · PSI</h3>
            <div className="mb-5 grid grid-cols-[repeat(auto-fill,minmax(150px,1fr))] gap-1.5">
              {r.features.map((x: J) => (
                <button key={x.feature} onClick={() => setFeat(x.feature)} title={`${x.feature}: PSI ${x.psi}, p=${x.p_value.toExponential(2)}`}
                  className={cn("truncate rounded-md px-2 py-1.5 text-left text-[11px]", f?.feature === x.feature && "ring-1 ring-white/50")}
                  style={{ background: `rgb(242 122 133 / ${Math.min(0.08 + x.psi / 1.2, 0.85)})` }}>
                  <span className="text-white">{x.feature}</span> <span className="tabular-nums text-white/70">{num(x.psi, 2)}</span>
                </button>
              ))}
            </div>
            {f && (
              <>
                <h3 className="mb-1 text-sm font-medium text-white">{f.feature} <span className="text-xs font-normal text-white/50">{f.test} statistic {num(f.statistic, 3)} · p {f.p_value.toExponential(2)} · PSI {num(f.psi, 3)} {f.drifted ? "· drifted" : ""}</span></h3>
                <div className="h-56">
                  <ResponsiveContainer>
                    <BarChart data={f.histogram ? f.histogram.reference.map((v: number, i: number) => ({ bin: num(f.histogram.edges[i], 2), reference: v, current: f.histogram.current[i] }))
                      : f.distribution.categories.map((c: string, i: number) => ({ bin: c, reference: f.distribution.reference[i], current: f.distribution.current[i] }))} barGap={0}>
                      <CartesianGrid stroke="rgb(255 255 255 / .06)" vertical={false} />
                      <XAxis dataKey="bin" stroke="rgb(255 255 255 / .4)" fontSize={10} interval="preserveStartEnd" />
                      <YAxis stroke="rgb(255 255 255 / .4)" fontSize={10} width={36} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
                      <Tooltip formatter={(v) => `${(Number(v) * 100).toFixed(1)}%`} />
                      <Legend wrapperStyle={{ fontSize: 12 }} />
                      <Bar dataKey="reference" name="Training reference" fill="#8fb4dc" />
                      <Bar dataKey="current" name="Current" fill="#f5b04c" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </>
            )}
            <table className="mt-4 w-full text-sm">
              <thead><tr><Th>Feature</Th><Th>Test</Th><Th right>Statistic</Th><Th right>p-value</Th><Th right>PSI</Th><Th>Drifted</Th></tr></thead>
              <tbody className="divide-y divide-white/5">
                {r.features.slice(0, 12).map((x: J) => (
                  <tr key={x.feature}><td className="py-1.5">{x.feature}</td><td className="text-white/60">{x.test}</td><td className="text-right tabular-nums">{num(x.statistic, 3)}</td><td className="text-right tabular-nums">{x.p_value.toExponential(1)}</td><td className="text-right tabular-nums">{num(x.psi, 3)}</td><td>{x.drifted ? <Pill tone="fail">yes</Pill> : <Pill>no</Pill>}</td></tr>
                ))}
              </tbody>
            </table>
            {r.summary.verdict !== "no_drift" && <p className="mt-4 text-sm text-white/60">An incident was opened or updated for this report. <Link href="/incidents" className="text-ember">Open incidents</Link></p>}
          </Panel>
        )}
      </div>
    </>
  );
}

function Box({ label, value }: { label: string; value: React.ReactNode }) {
  return <div className="rounded-xl bg-white/[0.03] px-3 py-2"><div className="text-xs text-white/45">{label}</div><div className="mt-0.5 text-sm font-medium tabular-nums text-white">{value}</div></div>;
}
