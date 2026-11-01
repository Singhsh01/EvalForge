"use client";

import { PageHeader } from "@/components/shell";
import { Empty, ErrorNote, Panel, Stat, Th } from "@/components/ui";
import { api, getUser, type J } from "@/lib/api";
import { ago, num } from "@/lib/format";
import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

export default function SystemPage() {
  const [hours, setHours] = useState(24);
  const [action, setAction] = useState("");
  const s = useQuery({ queryKey: ["obs", hours], queryFn: () => api.obsSummary(hours), refetchInterval: 10000 });
  const audit = useQuery({ queryKey: ["audit", action], queryFn: () => api.audit(`?limit=150${action ? `&action=${encodeURIComponent(action)}` : ""}`) });
  const isAdmin = getUser()?.role === "admin";
  const users = useQuery({ queryKey: ["users"], queryFn: api.users, enabled: isAdmin });
  const d = s.data;
  return (
    <>
      <PageHeader title="System health & audit"
        description="Request rates, latencies and error rates from server spans, background-job outcomes, pipeline retries and inference latency, plus the immutable audit log of who did what."
        action={<select className="rounded-lg border border-white/10 bg-ink-950 px-3 py-2 text-sm" value={hours} onChange={(e) => setHours(Number(e.target.value))}>{[1, 24, 168].map((h) => <option key={h} value={h}>Last {h === 168 ? "7 days" : `${h} h`}</option>)}</select>} />
      <ErrorNote error={s.error ?? audit.error} />
      {d && (
        <div className="mb-6 grid grid-cols-2 gap-4 md:grid-cols-5">
          <Panel><Stat label="Requests" value={d.http.requests} sub={`${(d.http.server_error_rate * 100).toFixed(2)}% server errors`} /></Panel>
          <Panel><Stat label="Worker failures" value={d.worker_failures} sub="failed background jobs" /></Panel>
          <Panel><Stat label="Pipeline retries" value={d.pipeline_step_retries} sub="extra step attempts" /></Panel>
          <Panel><Stat label="Predictions" value={d.inference.predictions} sub={`p95 ${num(d.inference.p95_ms, 2)} ms`} /></Panel>
          <Panel><Stat label="Spans" value={d.spans.total} sub={`${d.spans.errors} errors`} /></Panel>
        </div>
      )}
      <div className="grid gap-6 xl:grid-cols-2">
        <Panel title="API routes">
          {!d?.http.routes.length ? <Empty title="No traffic in this window" /> : (
            <table className="w-full text-sm">
              <thead><tr><Th>Route</Th><Th right>Req</Th><Th right>p50</Th><Th right>p95</Th><Th right>4xx</Th><Th right>5xx</Th></tr></thead>
              <tbody className="divide-y divide-white/5">
                {d.http.routes.map((r: J) => <tr key={r.route}><td className="max-w-64 truncate py-1.5 font-mono text-[11px]">{r.route}</td><td className="text-right">{r.requests}</td><td className="text-right tabular-nums">{num(r.p50_ms, 1)}</td><td className="text-right tabular-nums">{num(r.p95_ms, 1)}</td><td className="text-right text-warn">{r.client_errors || ""}</td><td className="text-right text-fail">{r.server_errors || ""}</td></tr>)}
              </tbody>
            </table>
          )}
          {(d?.jobs?.length ?? 0) > 0 && <div className="mt-4 text-xs text-white/55">Jobs: {d!.jobs.map((j: J) => `${j.kind} ${j.status} ×${j.count}`).join(" · ")}</div>}
        </Panel>
        <Panel title="Audit log" action={<input aria-label="Filter actions" placeholder="action prefix, e.g. approval." className="rounded-lg border border-white/10 bg-ink-950/60 px-2 py-1 text-xs" value={action} onChange={(e) => setAction(e.target.value)} />}>
          <ul className="max-h-[480px] divide-y divide-white/5 overflow-y-auto">
            {audit.data?.map((e) => (
              <li key={e.id} className="py-2 text-sm">
                <span className="font-mono text-[12px] text-ember">{e.action}</span> <span className="text-white/70">by {e.actor}</span> <span className="text-xs text-white/40">· {e.entity_type} {String(e.entity_id).slice(0, 8)} · {ago(e.created_at)}</span>
              </li>
            ))}
          </ul>
        </Panel>
        {isAdmin && (
          <Panel title="Users">
            <table className="w-full text-sm"><thead><tr><Th>User</Th><Th>Role</Th><Th>Active</Th></tr></thead>
              <tbody className="divide-y divide-white/5">{users.data?.map((u) => <tr key={u.id}><td className="py-1.5">{u.username}</td><td>{u.role}</td><td>{u.active ? "yes" : "no"}</td></tr>)}</tbody></table>
          </Panel>
        )}
      </div>
    </>
  );
}
