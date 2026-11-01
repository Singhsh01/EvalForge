"use client";

import { PageHeader } from "@/components/shell";
import { Button, Empty, ErrorNote, JsonBlock, Panel, Pill, Status } from "@/components/ui";
import { api, type J } from "@/lib/api";
import { ago, cn, num } from "@/lib/format";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AnimatePresence, motion } from "motion/react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";

const EXAMPLES = [
  "Why did my latest model perform worse?",
  "Which dataset changes are responsible for this regression?",
  "Compare the last five experiments and recommend a candidate model.",
  "Why did this AI agent fail its evaluation?",
  "Investigate the latest deployment failure.",
  "Investigate the drift incident in production.",
];

export default function CopilotPage() { return <Suspense><Copilot /></Suspense>; }

function Copilot() {
  const qc = useQueryClient();
  const params = useSearchParams();
  const [q, setQ] = useState("");
  const [sel, setSel] = useState<string | null>(params.get("session"));
  const configs = useQuery({ queryKey: ["agent-configs"], queryFn: api.agentConfigs });
  const [cfg, setCfg] = useState("");
  useEffect(() => { if (!cfg && configs.data?.length) setCfg(configs.data.filter((c) => c.name === "copilot").at(-1)?.id ?? configs.data[0].id); }, [configs.data, cfg]);
  const sessions = useQuery({ queryKey: ["copilot-sessions"], queryFn: api.sessions, refetchInterval: 5000 });
  const s = useQuery({ queryKey: ["copilot", sel], queryFn: () => api.copilotSession(sel!), enabled: !!sel,
    refetchInterval: (x) => (x.state.data?.status === "running" ? 700 : false) });
  const ask = useMutation({ mutationFn: () => api.ask(q, cfg || undefined), onSuccess: (r: J) => { setSel(r.id); qc.invalidateQueries({ queryKey: ["copilot-sessions"] }); } });
  const cancel = useMutation({ mutationFn: () => api.cancelSession(sel!), onSuccess: () => qc.invalidateQueries() });
  const d = s.data;
  const rep = d?.report;

  return (
    <>
      <PageHeader title="AI MLOps Copilot"
        description="Ask an engineering question. The Copilot plans an investigation, gathers evidence only through permissioned tools over real EvalForge data, and returns findings that cite the identifiers they rest on. It cannot change anything: actions become approval requests." />
      <ErrorNote error={ask.error ?? configs.error} />
      <div className="grid gap-6 xl:grid-cols-3">
        <div className="space-y-6 xl:col-span-2">
          <Panel>
            <form onSubmit={(e) => { e.preventDefault(); if (q.trim()) ask.mutate(); }} className="flex flex-col gap-3 sm:flex-row">
              <input aria-label="Question" value={q} onChange={(e) => setQ(e.target.value)} placeholder="Ask about experiments, data, deployments, drift or agent evaluations…"
                className="flex-1 rounded-xl border border-white/10 bg-ink-950/60 px-4 py-3 text-[15px] text-white placeholder:text-white/30 focus:border-ember/60 focus:outline-none" />
              <Button type="submit" loading={ask.isPending} className="px-5">Investigate</Button>
            </form>
            <div className="mt-3 flex flex-wrap gap-2">
              {EXAMPLES.map((e) => <button key={e} onClick={() => setQ(e)} className="rounded-full border border-white/10 px-3 py-1 text-xs text-white/60 hover:border-white/25 hover:text-white">{e}</button>)}
            </div>
            <div className="mt-3 flex items-center gap-2 text-xs text-white/45">
              Agent configuration
              <select className="rounded-md border border-white/10 bg-ink-950 px-2 py-1" value={cfg} onChange={(e) => setCfg(e.target.value)}>
                {configs.data?.map((c) => <option key={c.id} value={c.id}>{c.name} v{c.version} · {c.config.planner} · {c.config.tools_enabled.length} tools{c.note ? ` — ${c.note}` : ""}</option>)}
              </select>
            </div>
          </Panel>

          {d && (
            <Panel title={d.question} subtitle={<span className="flex flex-wrap items-center gap-2"><Status value={d.status} /> {rep?.intent && <Pill tone="steel">{rep.intent.replaceAll("_", " ")}</Pill>} planner {d.planner} · {d.tool_calls} tool calls · {num(d.duration_ms, 0)} ms</span>}
              action={d.status === "running" ? <Button variant="outline" onClick={() => cancel.mutate()}>Cancel</Button> : d.trace_id ? <Link className="text-sm text-ember" href={`/traces?id=${d.trace_id}`}>Open trace</Link> : null}>
              <h3 className="mb-2 text-sm font-medium text-white">Investigation steps</h3>
              <ol className="mb-5 space-y-1.5">
                <AnimatePresence initial={false}>
                  {(d.steps ?? []).map((c: J, k: number) => (
                    <motion.li key={k} initial={{ opacity: 0, x: -6 }} animate={{ opacity: 1, x: 0 }} className="flex items-start gap-3 text-sm">
                      <span className={cn("mt-1.5 size-2 shrink-0 rounded-full", c.error ? "bg-fail" : "bg-pass")} />
                      <div className="min-w-0">
                        <span className="font-mono text-[13px] text-white">{c.tool}</span>
                        <span className="ml-2 font-mono text-[11px] text-white/45">{JSON.stringify(c.args)}</span>
                        <span className="ml-2 text-[11px] text-white/35">{num(c.duration_ms, 1)} ms</span>
                        {c.error && <div className="text-xs text-fail">{c.error}</div>}
                      </div>
                    </motion.li>
                  ))}
                </AnimatePresence>
                {d.status === "running" && <li className="text-sm text-ember">Working…</li>}
              </ol>
              {rep && (
                <>
                  <Block title="Findings" items={rep.findings.map((f: J) => ({ text: f.claim, refs: f.evidence }))} />
                  <Block title="Hypotheses" items={rep.hypotheses.map((h: J) => ({ text: h.hypothesis, refs: h.evidence, tag: h.confidence }))} />
                  <Block title="Recommendations" items={rep.recommendations.map((r: J) => ({ text: r.action, refs: r.evidence ?? [], tag: r.requires_approval ? "needs approval" : undefined }))} />
                  {rep.gaps?.length > 0 && <Block title="Evidence gaps" items={rep.gaps.map((g: string) => ({ text: g, refs: [] }))} />}
                  {rep.unsupported?.length > 0 && (
                    <div className="mt-3 rounded-lg border border-warn/30 bg-warn/5 p-3 text-xs text-warn">
                      {rep.unsupported.length} claim(s) were rejected because their citations were not in the evidence: {rep.unsupported.map((u: J) => `“${u.claim ?? u.hypothesis}” (${u.reason})`).join("; ")}
                    </div>
                  )}
                  {!rep.findings.length && d.status !== "running" && <Empty title="No supported findings">{d.error ?? "The tools returned no evidence for this question."}</Empty>}
                </>
              )}
            </Panel>
          )}
        </div>
        <Panel title="Recent sessions">
          {!sessions.data?.length ? <Empty title="No sessions yet" /> : (
            <ul className="space-y-1">
              {sessions.data.map((x) => (
                <li key={x.id}><button onClick={() => setSel(x.id)} className={cn("w-full rounded-lg px-2 py-2 text-left", x.id === sel ? "bg-white/[0.06]" : "hover:bg-white/[0.03]")}>
                  <div className="truncate text-sm text-white/85">{x.question}</div>
                  <div className="mt-0.5 flex items-center gap-2 text-xs text-white/40"><Status value={x.status} /> {x.tool_calls} calls · {ago(x.created_at)}{x.incident_id && " · incident"}</div>
                </button></li>
              ))}
            </ul>
          )}
          <details className="mt-4 border-t border-white/5 pt-3"><summary className="cursor-pointer text-xs text-white/45">Raw report</summary>{rep && <JsonBlock value={rep} />}</details>
        </Panel>
      </div>
    </>
  );
}

function Block({ title, items }: { title: string; items: { text: string; refs: string[]; tag?: string }[] }) {
  if (!items.length) return null;
  return (
    <div className="mb-4">
      <h3 className="mb-1.5 text-sm font-medium text-white">{title}</h3>
      <ul className="space-y-2">
        {items.map((it, k) => (
          <li key={k} className="rounded-lg bg-white/[0.03] px-3 py-2 text-sm text-white/85">
            {it.tag && <Pill tone={it.tag === "high" ? "ember" : it.tag === "needs approval" ? "warn" : "neutral"}>{it.tag}</Pill>} {it.text}
            {it.refs.length > 0 && <div className="mt-1 flex flex-wrap gap-1">{it.refs.slice(0, 6).map((r) => <code key={r} title={r} className="rounded bg-white/5 px-1.5 font-mono text-[10px] text-white/50">{r.slice(0, 10)}</code>)}</div>}
          </li>
        ))}
      </ul>
    </div>
  );
}
