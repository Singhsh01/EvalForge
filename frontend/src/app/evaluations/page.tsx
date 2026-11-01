"use client";

import { SERIES } from "@/components/charts";
import { PageHeader } from "@/components/shell";
import { Button, Empty, ErrorNote, Field, inputCls, JsonBlock, Panel, Pill, Status, Th } from "@/components/ui";
import { api, type J } from "@/lib/api";
import { ago, cn, num } from "@/lib/format";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import { AXIS, EChart, LEGEND, TOOLTIP } from "@/components/echart";

const TEMPLATE = JSON.stringify({
  name: "my-suite", kind: "prompt", description: "",
  cases: [{ id: "c1", inputs: { question: "…" }, expected_contains: ["…"] }],
  evaluators: [{ type: "contains" }, { type: "json_schema" }],
  gates: { pass_rate: 0.9, regression_tolerance: 0.05 },
}, null, 1);

export default function EvalPage() { return <Suspense><Evaluations /></Suspense>; }

function Evaluations() {
  const qc = useQueryClient();
  const params = useSearchParams();
  const suites = useQuery({ queryKey: ["suites"], queryFn: api.suites });
  const runs = useQuery({ queryKey: ["eval-runs"], queryFn: api.evalRuns,
    refetchInterval: (q) => (q.state.data?.some((r) => ["queued", "running"].includes(r.status)) ? 1500 : 10000) });
  const prompts = useQuery({ queryKey: ["prompts"], queryFn: api.prompts });
  const configs = useQuery({ queryKey: ["agent-configs"], queryFn: api.agentConfigs });
  const providers = useQuery({ queryKey: ["providers"], queryFn: api.providers });
  const [sel, setSel] = useState<string | null>(params.get("run"));
  const [cmp, setCmp] = useState<string[]>([]);
  const [suiteId, setSuiteId] = useState("");
  const [subject, setSubject] = useState<J>({});
  const [draft, setDraft] = useState(TEMPLATE);
  const suite = suites.data?.find((s) => s.id === suiteId);
  useEffect(() => { if (!suiteId && suites.data?.length) setSuiteId(suites.data[0].id); }, [suites.data, suiteId]);
  useEffect(() => {
    if (!suite) return;
    setSubject(suite.kind === "agent" ? { type: "agent", agent_config_id: configs.data?.at(-1)?.id }
      : suite.kind === "rag" ? { type: "rag", top_k: 3 } : { type: "prompt", prompt_version_id: prompts.data?.[0]?.versions.at(-1)?.id, model: "anthropic:" });
  }, [suite, configs.data, prompts.data]);

  const create = useMutation({ mutationFn: () => api.createSuite(JSON.parse(draft)), onSuccess: (s: J) => { setSuiteId(s.id); qc.invalidateQueries(); } });
  const launch = useMutation({ mutationFn: () => api.launchEval(suiteId, subject), onSuccess: (r: J) => { setSel(r.id); qc.invalidateQueries(); } });
  const detail = useQuery({ queryKey: ["eval-run", sel], queryFn: () => api.evalRun(sel!), enabled: !!sel,
    refetchInterval: (q) => (q.state.data && ["queued", "running"].includes(q.state.data.status) ? 1000 : false) });
  const compare = useQuery({ queryKey: ["eval-cmp", cmp], queryFn: () => api.compareEvals(cmp[0], cmp[1]), enabled: cmp.length === 2 });
  const r = detail.data;

  return (
    <>
      <PageHeader title="Evaluation workbench"
        description="Versioned suites scored with deterministic checks, RAG retrieval and grounding checks, agent tool-use checks and rubric-based LLM judges. Every run is compared with the suite's previous run; regressions open an incident." />
      <ErrorNote error={suites.error ?? launch.error ?? create.error} />
      <div className="grid gap-6 xl:grid-cols-3">
        <Panel title="Run a suite">
          {!suites.data?.length ? <Empty title="No suites yet">Create one below.</Empty> : (
            <div className="space-y-3">
              <Field label="Suite">
                <select className={inputCls} value={suiteId} onChange={(e) => setSuiteId(e.target.value)}>
                  {suites.data.map((s) => <option key={s.id} value={s.id}>{s.name} v{s.version} · {s.kind} · {s.n_cases} cases</option>)}
                </select>
              </Field>
              {subject.type === "agent" && (
                <Field label="Agent configuration">
                  <select className={inputCls} value={subject.agent_config_id ?? ""} onChange={(e) => setSubject({ ...subject, agent_config_id: e.target.value })}>
                    {configs.data?.map((c) => <option key={c.id} value={c.id}>{c.name} v{c.version} {c.note && `— ${c.note}`}</option>)}
                  </select>
                </Field>
              )}
              {subject.type === "prompt" && (
                <>
                  <Field label="Prompt version">
                    <select className={inputCls} value={subject.prompt_version_id ?? ""} onChange={(e) => setSubject({ ...subject, prompt_version_id: e.target.value })}>
                      {prompts.data?.flatMap((p) => p.versions.map((v: J) => <option key={v.id} value={v.id}>{p.name} v{v.version} ({v.status})</option>))}
                    </select>
                  </Field>
                  <Field label="Model" hint={`provider:model — configured: ${(providers.data?.providers ?? []).filter((p: J) => p.configured).map((p: J) => p.provider).join(", ") || "none (set an API key)"}`}>
                    <input className={inputCls} value={subject.model ?? ""} onChange={(e) => setSubject({ ...subject, model: e.target.value })} />
                  </Field>
                </>
              )}
              {subject.type === "rag" && (
                <div className="grid grid-cols-2 gap-3">
                  <Field label="top_k"><input type="number" min={1} max={20} className={inputCls} value={subject.top_k} onChange={(e) => setSubject({ ...subject, top_k: Number(e.target.value) })} /></Field>
                  <Field label="Generator (optional)" hint="empty = retrieval only"><input className={inputCls} value={subject.model ?? ""} onChange={(e) => setSubject({ ...subject, model: e.target.value || undefined })} /></Field>
                </div>
              )}
              <Button className="w-full" loading={launch.isPending} onClick={() => launch.mutate()}>Launch evaluation</Button>
            </div>
          )}
          <details className="mt-5 border-t border-white/5 pt-4">
            <summary className="cursor-pointer text-sm text-white/70">Create a suite version</summary>
            <textarea aria-label="Suite JSON" className="mt-2 h-56 w-full rounded-lg border border-white/10 bg-ink-950/70 p-2 font-mono text-[11px]" value={draft} onChange={(e) => setDraft(e.target.value)} />
            <Button variant="outline" className="mt-2" loading={create.isPending} onClick={() => create.mutate()}>Save suite version</Button>
          </details>
        </Panel>
        <Panel className="xl:col-span-2" title="Runs" subtitle="Tick two runs of the same suite to compare them"
          action={cmp.length === 2 ? <Button variant="ghost" onClick={() => setCmp([])}>Clear comparison</Button> : null}>
          {!runs.data?.length ? <Empty title="No evaluation runs yet" /> : (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[680px] text-sm">
                <thead><tr><Th /><Th>Suite</Th><Th>Subject</Th><Th>Status</Th><Th right>Pass rate</Th><Th right>p95 ms</Th><Th right>Tokens</Th><Th>Regression</Th><Th right>When</Th></tr></thead>
                <tbody className="divide-y divide-white/5">
                  {runs.data.map((x) => (
                    <tr key={x.id} className={cn("cursor-pointer", x.id === sel && "bg-white/[0.04]")} onClick={() => setSel(x.id)}>
                      <td className="py-2" onClick={(e) => e.stopPropagation()}><input type="checkbox" aria-label="compare" checked={cmp.includes(x.id)} onChange={() => setCmp((c) => c.includes(x.id) ? c.filter((i) => i !== x.id) : [...c, x.id].slice(-2))} /></td>
                      <td>{x.suite}</td>
                      <td className="max-w-56 truncate font-mono text-[11px] text-white/60" title={x.subject_key}>{x.subject_key}</td>
                      <td><Status value={x.status} /></td>
                      <td className="text-right tabular-nums">{x.summary ? `${Math.round(x.summary.pass_rate * 100)}%` : "—"}</td>
                      <td className="text-right tabular-nums text-white/60">{num(x.summary?.latency_ms?.p95, 1)}</td>
                      <td className="text-right tabular-nums text-white/60">{x.summary ? x.summary.input_tokens + x.summary.output_tokens : "—"}</td>
                      <td>{x.regression?.detected ? <Pill tone="fail">regressed</Pill> : x.regression ? <Pill tone="pass">stable</Pill> : <span className="text-white/30">baseline</span>}</td>
                      <td className="text-right text-xs text-white/45">{ago(x.created_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Panel>
      </div>

      {compare.data && <ComparePanel data={compare.data} />}

      {r && (
        <Panel className="mt-6" title={`${r.suite} · ${r.subject_key}`}
          subtitle={r.summary ? `${r.summary.passed}/${r.summary.cases} passed · ${r.summary.errors} errors · gates ${r.summary.gates_passed ? "passed" : "failed"}${r.summary.cost_usd ? ` · $${r.summary.cost_usd}` : ""}` : r.status}
          action={r.trace_id ? <Link className="text-sm text-ember" href={`/traces?id=${r.trace_id}`}>Open trace</Link> : null}>
          {r.regression?.detected && (
            <p className="mb-4 rounded-lg border border-fail/30 bg-fail/10 px-3 py-2 text-sm text-fail">
              Regression vs baseline {String(r.regression.baseline_run_id).slice(0, 8)}: {Object.entries(r.regression.regressed_metrics).map(([k, v]) => `${k} ${Number(v) > 0 ? "+" : ""}${Number(v).toFixed(2)}`).join(", ")} · newly failing: {r.regression.newly_failing_cases.map((c: J) => c.case_id).join(", ")}
            </p>
          )}
          <ScoreHeatmap results={r.results ?? []} />
          <div className="overflow-x-auto">
            <table className="w-full min-w-[720px] text-sm">
              <thead><tr><Th>Case</Th><Th>Result</Th>{Object.keys(r.results?.[0]?.scores ?? {}).map((k) => <Th key={k} right>{k}</Th>)}<Th right>ms</Th><Th>Details</Th></tr></thead>
              <tbody className="divide-y divide-white/5">
                {r.results?.map((c: J) => (
                  <tr key={c.case_id} className="align-top">
                    <td className="py-2 font-mono text-xs">{c.case_id}</td>
                    <td>{c.error ? <Pill tone="fail">error</Pill> : c.passed ? <Pill tone="pass">pass</Pill> : <Pill tone="fail">fail</Pill>}</td>
                    {Object.entries((c.scores ?? {}) as Record<string, J>).map(([k, s]: [string, J]) => <td key={k} className={cn("text-right tabular-nums", s.passed ? "text-white/80" : "text-fail")}>{num(s.score, 2)}</td>)}
                    <td className="text-right tabular-nums text-white/50">{num(c.latency_ms, 0)}</td>
                    <td className="max-w-md text-xs text-white/55">
                      {c.error ?? Object.entries((c.scores ?? {}) as Record<string, J>).filter(([, s]: [string, J]) => !s.passed).map(([k, s]: [string, J]) => (
                        <div key={k}><b className="text-white/70">{k}:</b> {JSON.stringify(s.detail).slice(0, 240)}</div>
                      ))}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <details className="mt-3"><summary className="cursor-pointer text-xs text-white/45">Summary JSON</summary><JsonBlock value={r.summary} /></details>
        </Panel>
      )}
    </>
  );
}

function ComparePanel({ data }: { data: J }) {
  const metrics = Object.keys({ ...(data.a.summary?.evaluator_means ?? {}), pass_rate: 1 });
  const chart = metrics.map((m) => ({ metric: m, a: m === "pass_rate" ? data.a.summary?.pass_rate : data.a.summary?.evaluator_means?.[m], b: m === "pass_rate" ? data.b.summary?.pass_rate : data.b.summary?.evaluator_means?.[m] }));
  return (
    <Panel className="mt-6" title="Comparison" subtitle={`A: ${data.a.subject_key}   ·   B: ${data.b.subject_key}`}>
      <div className="grid gap-6 lg:grid-cols-2">
        <EChart height={256} ariaLabel="Evaluation metrics, run A vs run B" option={{
          grid: { left: 36, right: 8, top: 32, bottom: 28 }, legend: LEGEND,
          tooltip: { ...TOOLTIP, trigger: "axis", valueFormatter: (v: unknown) => num(Number(v), 3) },
          xAxis: { type: "category", data: chart.map((c) => c.metric), ...AXIS },
          yAxis: { type: "value", min: 0, max: 1, ...AXIS },
          series: [
            { type: "bar", name: "A", data: chart.map((c) => c.a), itemStyle: { color: SERIES[1], borderRadius: [3, 3, 0, 0] } },
            { type: "bar", name: "B", data: chart.map((c) => c.b), itemStyle: { color: SERIES[0], borderRadius: [3, 3, 0, 0] } },
          ],
        }} />
        <table className="text-sm">
          <thead><tr><Th>Case</Th><Th>A</Th><Th>B</Th></tr></thead>
          <tbody className="divide-y divide-white/5">
            {data.cases.map((c: J) => (
              <tr key={c.case_id} className={cn(c.a_passed && !c.b_passed && "bg-fail/5")}>
                <td className="py-1.5 font-mono text-xs">{c.case_id}</td>
                <td>{c.a_passed ? <Pill tone="pass">pass</Pill> : <Pill tone="fail">fail</Pill>}</td>
                <td>{c.b_passed ? <Pill tone="pass">pass</Pill> : <Pill tone="fail">fail</Pill>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

function ScoreHeatmap({ results }: { results: J[] }) {
  if (!results.length) return null;
  const evals = Object.keys(results[0]?.scores ?? {});
  if (!evals.length) return null;
  const cases = results.map((c) => c.case_id);
  const data = results.flatMap((c, y) => evals.map((e, x) => [x, y, c.error ? 0 : Number(c.scores?.[e]?.score ?? 0)]));
  return (
    <div className="mb-4">
      <EChart height={Math.max(140, cases.length * 28 + 60)} ariaLabel="Score heatmap: cases by evaluator" option={{
        grid: { left: 110, right: 70, top: 8, bottom: 32 },
        tooltip: { ...TOOLTIP, formatter: (p: { value: number[] }) => `${cases[p.value[1]]} · ${evals[p.value[0]]}: ${num(p.value[2], 2)}` },
        xAxis: { type: "category", data: evals, ...AXIS, splitArea: { show: false } },
        yAxis: { type: "category", data: cases, ...AXIS, axisLabel: { ...AXIS.axisLabel, fontFamily: "IBM Plex Mono" } },
        visualMap: { min: 0, max: 1, calculable: false, orient: "vertical", right: 0, top: "middle", itemHeight: 90,
          textStyle: { color: "rgba(255,255,255,0.6)" }, inRange: { color: ["#f27a85", "#f0c35b", "#5fd3a2"] } },
        series: [{ type: "heatmap", data, label: { show: true, color: "#0c1122", fontSize: 11, formatter: (p: { value: number[] }) => p.value[2].toFixed(2) },
          itemStyle: { borderColor: "#0c1122", borderWidth: 2, borderRadius: 4 } }],
      }} />
    </div>
  );
}
