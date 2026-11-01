"use client";

import { PageHeader } from "@/components/shell";
import { Button, Empty, ErrorNote, JsonBlock, Notice, Panel, Pill, Status, Th } from "@/components/ui";
import { api, type J } from "@/lib/api";
import { ago, cn, num } from "@/lib/format";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";

export default function DeploymentsPage() {
  const qc = useQueryClient();
  const deps = useQuery({ queryKey: ["deployments"], queryFn: () => api.deployments() });
  const models = useMemo(() => [...new Set((deps.data ?? []).map((d) => d.model_name as string))], [deps.data]);
  const [model, setModel] = useState("");
  useEffect(() => { if (!model && models.length) setModel(models[0]); }, [models, model]);
  const rows = (deps.data ?? []).filter((d) => d.model_name === model);
  const [note, setNote] = useState<string | null>(null);

  const act = useMutation({
    mutationFn: ({ kind, env }: { kind: string; env: string }) =>
      kind === "rollback" ? api.rollback(model, env) : api.promoteCanary(model, env),
    onSuccess: (r: J, { env }) => {
      qc.invalidateQueries();
      setNote(r.approval ? `Request filed (approval ${String(r.approval.id).slice(0, 8)}). Production changes run after an approver accepts.` : `Done in ${env}.`);
    },
  });

  return (
    <>
      <PageHeader title="Deployment center"
        description="Active releases per environment, canary traffic splits, release health checks and the full deployment history. Production rollouts, canary promotions and rollbacks all go through approval."
        action={models.length > 1 ? (
          <select className="rounded-lg border border-white/10 bg-ink-950 px-3 py-2 text-sm" value={model} onChange={(e) => setModel(e.target.value)}>
            {models.map((m) => <option key={m}>{m}</option>)}
          </select>
        ) : null} />
      <ErrorNote error={deps.error ?? act.error} />
      {note && <div className="mb-4"><Notice>{note}</Notice></div>}
      {!rows.length && deps.isSuccess ? <Empty title="Nothing deployed yet">Deploy a registered version from the Model registry.</Empty> : (
        <>
          <div className="mb-6 grid gap-6 lg:grid-cols-2">
            {(["staging", "production"] as const).map((env) => {
              const active = rows.filter((d) => d.environment === env && d.status === "active");
              const stable = active.find((d) => d.role === "stable");
              const canary = active.find((d) => d.role === "canary");
              return (
                <Panel key={env} title={env === "production" ? "Production" : "Staging"}
                  subtitle={stable ? `v${stable.version} stable${canary ? ` · v${canary.version} canary` : ""}` : "No active deployment"}>
                  {stable && (
                    <>
                      <div className="mb-3 flex h-3 overflow-hidden rounded-full" aria-label="Traffic split">
                        <div className="bg-steel" style={{ flex: 100 - (canary?.traffic_percent ?? 0) }} />
                        {canary && <div className="bg-ember" style={{ flex: canary.traffic_percent }} />}
                      </div>
                      <div className="mb-4 flex gap-4 text-xs text-white/60">
                        <span><span className="mr-1 inline-block size-2 rounded-full bg-steel" />v{stable.version} {100 - (canary?.traffic_percent ?? 0)}%</span>
                        {canary && <span><span className="mr-1 inline-block size-2 rounded-full bg-ember" />v{canary.version} {canary.traffic_percent}%</span>}
                      </div>
                      {[stable, canary].filter(Boolean).map((d) => (
                        <div key={d!.id} className="mb-2 flex flex-wrap items-center gap-2 text-sm">
                          <span className="text-white">v{d!.version} {d!.role}</span>
                          {d!.health?.healthy ? <Pill tone="pass">healthy</Pill> : <Pill tone="fail">unhealthy</Pill>}
                          <span className="text-xs text-white/45">p95 {num(d!.health?.p95_ms, 2)} ms · smoke accuracy {num(d!.health?.smoke_accuracy, 2)} · by {d!.deployed_by} {ago(d!.created_at)}</span>
                        </div>
                      ))}
                      <div className="mt-4 flex gap-2 border-t border-white/5 pt-4">
                        {canary && <Button variant="outline" onClick={() => act.mutate({ kind: "promote", env })}>Promote canary to 100%</Button>}
                        <Button variant="outline" onClick={() => act.mutate({ kind: "rollback", env })}>{canary ? "Roll back canary" : "Roll back"}</Button>
                      </div>
                    </>
                  )}
                </Panel>
              );
            })}
          </div>
          <Playground model={model} />
          <Panel className="mt-6" title="Deployment history">
            <div className="overflow-x-auto">
              <table className="w-full min-w-[720px] text-sm">
                <thead><tr><Th>When</Th><Th>Env</Th><Th>Version</Th><Th>Role / traffic</Th><Th>Status</Th><Th>Health</Th><Th>By</Th><Th>Note</Th></tr></thead>
                <tbody className="divide-y divide-white/5">
                  {rows.map((d) => (
                    <tr key={d.id}>
                      <td className="py-2 text-xs text-white/50">{ago(d.created_at)}</td>
                      <td>{d.environment}</td><td>v{d.version}</td>
                      <td>{d.role} · {d.traffic_percent}%</td>
                      <td><Status value={d.status} /></td>
                      <td>{d.health ? (d.health.healthy ? <Pill tone="pass">ok</Pill> : <Pill tone="fail">{Object.entries(d.health.checks ?? {}).filter(([, v]) => !v).map(([k]) => k).join(", ")}</Pill>) : "—"}</td>
                      <td className="text-white/60">{d.deployed_by}</td>
                      <td className="text-xs text-white/50">{d.note}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
        </>
      )}
    </>
  );
}

function Playground({ model }: { model: string }) {
  const [env, setEnv] = useState("production");
  const [text, setText] = useState("");
  const versions = useQuery({ queryKey: ["models"], queryFn: api.models });
  const ds = useQuery({ queryKey: ["datasets"], queryFn: api.datasets });
  useEffect(() => {
    if (text || !model || !ds.data) return;
    const d = ds.data.find((x) => x.name === model);
    if (!d) return;
    const v = d.versions.at(-1)!;
    api.preview(d.id, v.version).then((p) => {
      const rows = p.rows.slice(0, 2).map((r) => Object.fromEntries(p.columns.map((c, i) => [c, r[i]]).filter(([c]) => c !== d.target_column)));
      setText(JSON.stringify(rows, null, 1));
    }).catch(() => undefined);
  }, [model, ds.data, text]);
  const run = useMutation({ mutationFn: () => api.predict(model, env, JSON.parse(text)) });
  void versions;
  return (
    <Panel title="Inference playground" subtitle="Calls the live versioned inference API; every prediction is logged for drift monitoring.">
      <div className="grid gap-4 lg:grid-cols-2">
        <div>
          <textarea aria-label="Instances JSON" className="h-56 w-full rounded-lg border border-white/10 bg-ink-950/70 p-3 font-mono text-[11px] text-white/80" value={text} onChange={(e) => setText(e.target.value)} />
          <div className="mt-2 flex gap-2">
            <select className="rounded-lg border border-white/10 bg-ink-950 px-3 py-2 text-sm" value={env} onChange={(e) => setEnv(e.target.value)}>
              <option>production</option><option>staging</option>
            </select>
            <Button loading={run.isPending} onClick={() => run.mutate()}>Predict</Button>
          </div>
        </div>
        <div>
          <ErrorNote error={run.error} />
          {run.data && (
            <>
              <p className={cn("mb-2 text-sm text-white/70")}>Served by v{run.data.version} ({run.data.role}) in {num(run.data.latency_ms, 2)} ms · request {String(run.data.request_id).slice(0, 10)}</p>
              <JsonBlock value={run.data.predictions} />
            </>
          )}
        </div>
      </div>
    </Panel>
  );
}
