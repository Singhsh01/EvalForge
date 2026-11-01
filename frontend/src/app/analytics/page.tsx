"use client";

import { PageHeader } from "@/components/shell";
import { Empty, ErrorNote, Panel, Th } from "@/components/ui";
import { api, type J } from "@/lib/api";
import { ALGO_LABEL, num } from "@/lib/format";
import { useQuery } from "@tanstack/react-query";
import { AXIS, EChart, LEGEND, TOOLTIP } from "@/components/echart";

const C: Record<string, string> = { logistic_regression: "#8fb4dc", random_forest: "#5fd3a2", gradient_boosting: "#f5b04c" };

export default function AnalyticsPage() {
  const cp = useQuery({ queryKey: ["cost-perf"], queryFn: api.costPerf });
  const obs = useQuery({ queryKey: ["obs", 168], queryFn: () => api.obsSummary(168) });
  const training: J[] = cp.data?.training ?? [];
  const algos = [...new Set(training.map((t) => t.algorithm))];
  return (
    <>
      <PageHeader title="Cost & performance analytics"
        description="Quality against latency and compute for every measured run, evaluation cost and token use, and live serving latency. Costs appear only where a provider price is configured; nothing is estimated." />
      <ErrorNote error={cp.error} />
      <div className="grid gap-6 xl:grid-cols-2">
        <Panel title="Quality vs. inference latency" subtitle="Holdout F1 against single-row p95 latency; bubble size = training seconds">
          {!training.length ? <Empty title="No finished runs" /> : (
            <EChart height={288} ariaLabel="Holdout F1 against p95 latency per training run" option={{
              grid: { left: 48, right: 16, top: 32, bottom: 40 }, legend: LEGEND,
              tooltip: { ...TOOLTIP, formatter: (p: { data: J }) => `${p.data.algorithm} · ${String(p.data.run_id).slice(0, 8)}<br/>F1 ${num(p.data.f1, 4)} · p95 ${num(p.data.latency_p95_ms, 2)} ms<br/>train ${num(p.data.training_seconds, 2)} s` },
              xAxis: { type: "value", name: "p95 latency (ms)", nameLocation: "middle", nameGap: 26, scale: true, ...AXIS },
              yAxis: { type: "value", name: "F1", scale: true, max: 1, ...AXIS },
              series: algos.map((a) => ({ type: "scatter", name: ALGO_LABEL[a] ?? a, itemStyle: { color: C[a] ?? "#ddd", opacity: 0.85 },
                symbolSize: (_: unknown, p: { data: J }) => 10 + Math.min(30, Math.sqrt(p.data.training_seconds ?? 0) * 20),
                data: training.filter((t) => t.algorithm === a).map((t) => ({ ...t, value: [t.latency_p95_ms, t.f1] })) })),
            }} />
          )}
        </Panel>
        <Panel title="LLM usage (7 days)" subtitle="From recorded provider calls">
          {!obs.data?.llm?.length ? <Empty title="No LLM calls recorded">Configure a provider key to evaluate prompts or use the LLM planner.</Empty> : (
            <table className="w-full text-sm">
              <thead><tr><Th>Model</Th><Th right>Calls</Th><Th right>Tokens in/out</Th><Th right>Cost</Th><Th right>Avg ms</Th><Th right>Errors</Th></tr></thead>
              <tbody className="divide-y divide-white/5">
                {obs.data.llm.map((m: J) => <tr key={m.model}><td className="py-1.5">{m.model}</td><td className="text-right">{m.calls}</td><td className="text-right tabular-nums">{m.input_tokens}/{m.output_tokens}</td><td className="text-right tabular-nums">{m.cost_usd ? `$${m.cost_usd}` : "not priced"}</td><td className="text-right tabular-nums">{num(m.avg_latency_ms, 0)}</td><td className="text-right">{m.errors}</td></tr>)}
              </tbody>
            </table>
          )}
        </Panel>
        <Panel title="Evaluation runs">
          {!cp.data?.evaluations?.length ? <Empty title="No evaluation runs" /> : (
            <table className="w-full text-sm">
              <thead><tr><Th>Suite</Th><Th>Subject</Th><Th right>Pass rate</Th><Th right>p95 ms</Th><Th right>Tokens</Th><Th right>Cost</Th></tr></thead>
              <tbody className="divide-y divide-white/5">
                {cp.data.evaluations.map((e: J) => <tr key={e.eval_run_id}><td className="py-1.5">{e.suite}</td><td className="max-w-48 truncate font-mono text-[11px] text-white/60">{e.subject}</td><td className="text-right">{Math.round(e.pass_rate * 100)}%</td><td className="text-right tabular-nums">{num(e.p95_ms, 0)}</td><td className="text-right">{e.tokens}</td><td className="text-right">{e.cost_usd ? `$${e.cost_usd}` : "—"}</td></tr>)}
              </tbody>
            </table>
          )}
        </Panel>
        <Panel title="Serving latency (active deployments)">
          {!cp.data?.serving?.length ? <Empty title="No active deployments" /> : (
            <table className="w-full text-sm">
              <thead><tr><Th>Model</Th><Th>Env</Th><Th>Role</Th><Th right>Predictions</Th><Th right>p50 ms</Th><Th right>p95 ms</Th></tr></thead>
              <tbody className="divide-y divide-white/5">
                {cp.data.serving.map((s: J) => <tr key={s.deployment_id}><td className="py-1.5">{s.model} v{s.version}</td><td>{s.environment}</td><td>{s.role}</td><td className="text-right">{s.predictions}</td><td className="text-right tabular-nums">{num(s.p50_ms, 3)}</td><td className="text-right tabular-nums">{num(s.p95_ms, 3)}</td></tr>)}
              </tbody>
            </table>
          )}
        </Panel>
      </div>
    </>
  );
}
