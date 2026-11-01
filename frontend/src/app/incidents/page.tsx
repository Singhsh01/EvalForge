"use client";

import { PageHeader } from "@/components/shell";
import { Button, Empty, ErrorNote, Field, inputCls, JsonBlock, Notice, Panel, Pill, Status } from "@/components/ui";
import { api, type J } from "@/lib/api";
import { ago, cn } from "@/lib/format";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";

export default function IncidentsPage() { return <Suspense><Incidents /></Suspense>; }

const SEV: Record<string, "fail" | "warn" | "neutral"> = { high: "fail", medium: "warn", low: "neutral" };

function Incidents() {
  const qc = useQueryClient();
  const params = useSearchParams();
  const list = useQuery({ queryKey: ["incidents"], queryFn: api.incidents, refetchInterval: 6000 });
  const [sel, setSel] = useState<string | null>(params.get("id"));
  useEffect(() => { if (!sel && list.data?.length) setSel(list.data[0].id); }, [list.data, sel]);
  const inc = useQuery({ queryKey: ["incident", sel], queryFn: () => api.incident(sel!), enabled: !!sel, refetchInterval: 4000 });
  const datasets = useQuery({ queryKey: ["datasets"], queryFn: api.datasets });
  const [dv, setDv] = useState("");
  const [override, setOverride] = useState("");
  const [note, setNote] = useState("");
  const [msg, setMsg] = useState<string | null>(null);
  const i = inc.data;
  const done = () => qc.invalidateQueries();

  const investigate = useMutation({ mutationFn: () => api.investigate(sel!), onSuccess: done });
  const plan = useMutation({ mutationFn: () => api.retrainPlan(sel!, dv || undefined) });
  const retrain = useMutation({ mutationFn: () => api.requestRetraining(sel!, { dataset_version_id: dv || undefined, override_reason: override || undefined }),
    onSuccess: () => { setMsg("Retraining requested. It starts only after an approver accepts it."); done(); } });
  const status = useMutation({ mutationFn: (s: string) => api.incidentStatus(sel!, s, note || s), onSuccess: done });
  const modelVersions = (datasets.data ?? []).find((d) => d.name === i?.details?.model_name)?.versions ?? [];

  return (
    <>
      <PageHeader title="Incident investigation"
        description="Incidents are opened automatically by drift reports and evaluation regressions. Each keeps an event timeline, the Copilot's evidence-backed findings, and the approval-gated actions taken to resolve it." />
      <ErrorNote error={list.error ?? investigate.error ?? retrain.error ?? plan.error} />
      {msg && <div className="mb-4"><Notice>{msg}</Notice></div>}
      <div className="grid gap-6 xl:grid-cols-5">
        <Panel className="xl:col-span-2" title="Incidents">
          {!list.data?.length ? <Empty title="No incidents">Drift reports and evaluation regressions open incidents here.</Empty> : (
            <ul className="space-y-1">
              {list.data.map((x) => (
                <li key={x.id}>
                  <button onClick={() => { setSel(x.id); plan.reset(); setMsg(null); }} className={cn("w-full rounded-lg px-2 py-2 text-left", x.id === sel ? "bg-white/[0.06]" : "hover:bg-white/[0.03]")}>
                    <div className="flex items-center gap-2"><Pill tone={SEV[x.severity]}>{x.severity}</Pill><Status value={x.status} /><span className="text-xs text-white/40">{x.kind.replaceAll("_", " ")} · {ago(x.created_at)}</span></div>
                    <div className="mt-1 text-sm text-white/85">{x.title}</div>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </Panel>
        {i && (
          <div className="space-y-6 xl:col-span-3">
            <Panel title={i.title} subtitle={`${i.kind.replaceAll("_", " ")} · source ${i.source_type} ${String(i.source_id).slice(0, 8)}`}
              action={<Button onClick={() => investigate.mutate()} loading={investigate.isPending}>Investigate with Copilot</Button>}>
              <h3 className="mb-2 text-sm font-medium text-white">Timeline</h3>
              <ol className="relative mb-4 space-y-3 border-l border-white/10 pl-4">
                {i.events.map((e: J) => (
                  <li key={e.id} className="relative">
                    <span className={cn("absolute -left-[21px] top-1.5 size-2.5 rounded-full", e.kind === "detected" ? "bg-fail" : e.kind === "findings" ? "bg-ember" : e.kind === "action" ? "bg-steel" : "bg-white/40")} />
                    <div className="text-xs text-white/40">{e.kind} · {e.actor} · {ago(e.created_at)}</div>
                    <div className="text-sm text-white/80">{e.message}</div>
                  </li>
                ))}
              </ol>
              {i.details?.affected_features?.length > 0 && <p className="text-xs text-white/55">Affected features: {i.details.affected_features.join(", ")}</p>}
              {i.details?.pipeline_run_id && <p className="mt-1 text-xs"><Link className="text-ember" href={`/pipelines?run=${i.details.pipeline_run_id}`}>Retraining pipeline {String(i.details.pipeline_run_id).slice(0, 8)}</Link></p>}
              {i.details?.current_run_id && <p className="mt-1 text-xs"><Link className="text-ember" href={`/evaluations?run=${i.details.current_run_id}`}>Regressed evaluation run</Link></p>}
            </Panel>

            {i.findings && (
              <Panel title="Copilot findings" subtitle={`Session ${String(i.findings.session_id).slice(0, 8)} · every claim cites identifiers returned by tools`}
                action={<Link className="text-sm text-ember" href={`/copilot?session=${i.findings.session_id}`}>Open session</Link>}>
                <Section title="Evidence" items={i.findings.findings.map((f: J) => ({ text: f.claim, refs: f.evidence }))} />
                <Section title="Hypotheses" items={i.findings.hypotheses.map((h: J) => ({ text: `(${h.confidence}) ${h.hypothesis}`, refs: h.evidence }))} />
                <Section title="Proposed fixes" items={i.findings.recommendations.map((r: J) => ({ text: r.action + (r.requires_approval ? " — requires approval" : ""), refs: r.evidence ?? [] }))} />
                {i.findings.unsupported?.length > 0 && <p className="mt-2 text-xs text-warn">{i.findings.unsupported.length} unsupported claim(s) were excluded.</p>}
              </Panel>
            )}

            {i.details?.model_version_id && (
              <Panel title="Controlled retraining" subtitle="A plan states whether the evidence justifies retraining; the request still needs an approver.">
                <div className="flex flex-wrap items-end gap-2">
                  <Field label="Training data version">
                    <select className={inputCls} value={dv} onChange={(e) => setDv(e.target.value)}>
                      <option value="">Latest validated</option>
                      {modelVersions.map((v) => <option key={v.id} value={v.id}>v{v.version} · {v.n_rows} rows</option>)}
                    </select>
                  </Field>
                  <Button variant="outline" onClick={() => plan.mutate()} loading={plan.isPending}>Evaluate plan</Button>
                </div>
                {plan.data && (
                  <div className="mt-3 space-y-2">
                    <p className={cn("text-sm", plan.data.justified ? "text-pass" : "text-warn")}>{plan.data.justified ? "Retraining is justified." : "Retraining is not justified yet."} <span className="text-white/60">{plan.data.reasons.join(" ")}</span></p>
                    {!plan.data.justified && <input className={inputCls} placeholder="Override reason (optional, recorded)" value={override} onChange={(e) => setOverride(e.target.value)} />}
                    <Button onClick={() => retrain.mutate()} loading={retrain.isPending} disabled={!plan.data.justified && !override}>Request retraining approval</Button>
                    <JsonBlock value={plan.data.config} />
                  </div>
                )}
              </Panel>
            )}

            <Panel title="Status">
              <div className="flex flex-wrap items-center gap-2">
                <input className={cn(inputCls, "max-w-md")} placeholder="Note for the timeline" value={note} onChange={(e) => setNote(e.target.value)} />
                {["mitigating", "resolved"].map((s) => <Button key={s} variant="outline" onClick={() => status.mutate(s)}>Mark {s}</Button>)}
              </div>
            </Panel>
          </div>
        )}
      </div>
    </>
  );
}

function Section({ title, items }: { title: string; items: { text: string; refs: string[] }[] }) {
  if (!items.length) return null;
  return (
    <div className="mb-3">
      <h3 className="mb-1 text-sm font-medium text-white">{title}</h3>
      <ul className="space-y-1.5">
        {items.map((it, k) => (
          <li key={k} className="text-sm text-white/80">{it.text}
            <span className="ml-2 inline-flex flex-wrap gap-1">{it.refs.slice(0, 4).map((r) => <code key={r} className="rounded bg-white/5 px-1 font-mono text-[10px] text-white/50">{r.slice(0, 8)}</code>)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
