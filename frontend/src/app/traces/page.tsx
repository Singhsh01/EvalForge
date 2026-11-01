"use client";

import { PageHeader } from "@/components/shell";
import { Empty, ErrorNote, JsonBlock, Panel, Pill, Th } from "@/components/ui";
import { api, type J } from "@/lib/api";
import { ago, cn, num } from "@/lib/format";
import { useQuery } from "@tanstack/react-query";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useMemo, useState } from "react";

export default function TracesPage() { return <Suspense><Traces /></Suspense>; }

const KIND_COLOR: Record<string, string> = { server: "bg-steel", consumer: "bg-[#c9a7f5]", internal: "bg-ember" };

function Traces() {
  const params = useSearchParams();
  const [errorsOnly, setErrorsOnly] = useState(false);
  const [q, setQ] = useState("");
  const list = useQuery({ queryKey: ["traces", errorsOnly, q], queryFn: () => api.traces(`?limit=60${errorsOnly ? "&status=error" : ""}${q ? `&q=${encodeURIComponent(q)}` : ""}`), refetchInterval: 8000 });
  const [sel, setSel] = useState<string | null>(params.get("id"));
  useEffect(() => { if (!sel && list.data?.length) setSel(list.data[0].trace_id); }, [list.data, sel]);
  const trace = useQuery({ queryKey: ["trace", sel], queryFn: () => api.trace(sel!), enabled: !!sel });
  const [span, setSpan] = useState<J | null>(null);

  const ordered = useMemo(() => {
    const spans: J[] = trace.data?.spans ?? [];
    const kids: Record<string, J[]> = {};
    spans.forEach((s) => { (kids[s.parent_id ?? "root"] ??= []).push(s); });
    const out: (J & { depth: number })[] = [];
    const walk = (id: string, depth: number) => (kids[id] ?? []).sort((a, b) => a.offset_ms - b.offset_ms).forEach((s) => { out.push({ ...s, depth }); walk(s.span_id, depth + 1); });
    walk("root", 0);
    const ids = new Set(out.map((s) => s.span_id));
    spans.filter((s) => !ids.has(s.span_id)).forEach((s) => out.push({ ...s, depth: 0 }));  // orphans
    return out;
  }, [trace.data]);
  const total = trace.data?.duration_ms || 1;

  return (
    <>
      <PageHeader title="Trace explorer"
        description="OpenTelemetry spans recorded by the API, background workers, training, MLflow logging, inference, LLM calls, evaluations and Copilot tool calls. Secrets and e-mail addresses are redacted before storage." />
      <ErrorNote error={list.error ?? trace.error} />
      <div className="grid gap-6 xl:grid-cols-5">
        <Panel className="xl:col-span-2" title="Recent traces"
          action={<label className="flex items-center gap-2 text-xs text-white/60"><input type="checkbox" checked={errorsOnly} onChange={(e) => setErrorsOnly(e.target.checked)} /> errors only</label>}>
          <input aria-label="Filter by span name" placeholder="Filter by name (e.g. POST, eval, copilot)" className="mb-3 w-full rounded-lg border border-white/10 bg-ink-950/60 px-3 py-2 text-sm" value={q} onChange={(e) => setQ(e.target.value)} />
          {!list.data?.length ? <Empty title="No traces match" /> : (
            <table className="w-full text-sm">
              <thead><tr><Th>Root span</Th><Th right>Spans</Th><Th right>ms</Th><Th right>When</Th></tr></thead>
              <tbody className="divide-y divide-white/5">
                {list.data.map((t) => (
                  <tr key={t.trace_id} onClick={() => { setSel(t.trace_id); setSpan(null); }} className={cn("cursor-pointer", t.trace_id === sel && "bg-white/[0.05]")}>
                    <td className="max-w-56 truncate py-1.5"><span className={cn(t.has_error ? "text-fail" : "text-white/85")}>{t.name}</span></td>
                    <td className="text-right tabular-nums text-white/55">{t.span_count}</td>
                    <td className="text-right tabular-nums text-white/55">{num(t.duration_ms, 1)}</td>
                    <td className="text-right text-xs text-white/40">{ago(t.start_time)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Panel>
        <Panel className="xl:col-span-3" title={sel ? `Trace ${sel.slice(0, 12)}…` : "Waterfall"}
          subtitle={trace.data ? `${trace.data.spans.length} spans · ${num(trace.data.duration_ms, 1)} ms` : undefined}>
          {trace.data && (
            <>
              {Object.keys(trace.data.correlations).length > 0 && (
                <div className="mb-3 flex flex-wrap gap-2">{Object.entries(trace.data.correlations).map(([k, v]) => <Pill key={k} tone="steel">{k}: {String(v).slice(0, 14)}</Pill>)}</div>
              )}
              <div className="space-y-0.5" role="list" aria-label="Span waterfall">
                {ordered.map((s) => (
                  <button key={s.span_id} role="listitem" onClick={() => setSpan(s)} className={cn("grid w-full grid-cols-[minmax(0,2fr)_minmax(0,3fr)] items-center gap-3 rounded px-1 py-1 text-left hover:bg-white/[0.04]", span?.span_id === s.span_id && "bg-white/[0.06]")}>
                    <span className="truncate text-xs" style={{ paddingLeft: s.depth * 12 }}>
                      <span className={s.status === "error" ? "text-fail" : "text-white/85"}>{s.name}</span>
                    </span>
                    <span className="relative h-4">
                      <span className={cn("absolute top-1 h-2 rounded-sm", s.status === "error" ? "bg-fail" : KIND_COLOR[s.kind] ?? "bg-ember")}
                        style={{ left: `${(s.offset_ms / total) * 100}%`, width: `max(2px, ${(s.duration_ms / total) * 100}%)` }} />
                      <span className="absolute right-0 top-0 text-[10px] tabular-nums text-white/40">{num(s.duration_ms, 1)} ms</span>
                    </span>
                  </button>
                ))}
              </div>
              {span && (
                <div className="mt-4 border-t border-white/5 pt-3">
                  <div className="mb-1 text-sm text-white">{span.name} <span className="text-xs text-white/40">{span.kind} · {span.service}</span></div>
                  {span.status_message && <p className="mb-2 text-sm text-fail">{span.status_message}</p>}
                  <JsonBlock value={{ attributes: span.attributes, events: span.events }} />
                </div>
              )}
            </>
          )}
        </Panel>
      </div>
    </>
  );
}
