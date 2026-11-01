"use client";

import { GroupedMetricBars, SERIES } from "@/components/charts";
import { PageHeader } from "@/components/shell";
import { ErrorNote, Panel } from "@/components/ui";
import { api } from "@/lib/api";
import { ALGO_LABEL, cn, METRIC_LABEL, num, QUALITY_METRICS } from "@/lib/format";
import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense } from "react";

export default function ComparePage() {
  return <Suspense><Compare /></Suspense>;
}

function Compare() {
  const ids = (useSearchParams().get("ids") ?? "").split(",").filter(Boolean);
  const q = useQuery({ queryKey: ["compare", ids], queryFn: () => api.compare(ids), enabled: ids.length >= 2 });
  const c = q.data;
  const name = (id: string) => {
    const r = c?.runs.find((x) => x.id === id);
    return r ? `${ALGO_LABEL[r.algorithm] ?? r.algorithm} · ${r.id.slice(0, 6)}` : id;
  };
  const quality = c?.metric_keys.filter((k) => QUALITY_METRICS.includes(k)) ?? [];
  const chart = quality.map((k) => ({ metric: METRIC_LABEL[k] ?? k, ...Object.fromEntries(c!.runs.map((r) => [r.id, r.metrics?.[k] ?? 0])) }));
  const shown = c?.metric_keys.filter((k) => METRIC_LABEL[k]) ?? [];

  return (
    <>
      <PageHeader title="Run comparison" description="Best values are highlighted per metric, respecting whether higher or lower is better. Ties highlight every winner." />
      <ErrorNote error={q.error} />
      {c?.warnings.map((w) => <p key={w} className="mb-4 rounded-lg border border-warn/30 bg-warn/10 px-3 py-2 text-sm text-warn">{w}</p>)}
      {c && (
        <div className="grid gap-6">
          <Panel title="Holdout quality" subtitle={c.same_holdout ? "All runs scored on the same holdout rows" : undefined}>
            <GroupedMetricBars data={chart} series={c.runs.map((r) => ({ key: r.id, name: name(r.id) }))} />
          </Panel>
          <Panel title="Metrics and parameters">
            <div className="overflow-x-auto">
              <table className="w-full min-w-[640px] text-sm">
                <thead><tr className="text-left">
                  <th className="pb-3 font-normal text-white/45">Metric</th>
                  {c.runs.map((r, i) => (
                    <th key={r.id} className="pb-3 text-right font-medium">
                      <Link href={`/experiments/${r.id}`} className="hover:underline" style={{ color: SERIES[i % SERIES.length] }}>{name(r.id)}</Link>
                    </th>
                  ))}
                </tr></thead>
                <tbody className="divide-y divide-white/5">
                  {shown.map((k) => (
                    <tr key={k}>
                      <td className="py-2 text-white/60">{METRIC_LABEL[k]}</td>
                      {c.runs.map((r) => {
                        const best = c.best[k]?.includes(r.id);
                        return <td key={r.id} className={cn("text-right tabular-nums", best ? "font-semibold text-ember" : "text-white/80")}>{num(r.metrics?.[k])}</td>;
                      })}
                    </tr>
                  ))}
                  <tr><td colSpan={c.runs.length + 1} className="pt-5 pb-1 text-xs text-white/45">Parameters (differences highlighted)</td></tr>
                  {c.param_keys.map((k) => {
                    const differs = c.differing_params.includes(k);
                    return (
                      <tr key={k}>
                        <td className={cn("py-2", differs ? "text-white" : "text-white/45")}>{k}</td>
                        {c.runs.map((r) => {
                          const v = k === "algorithm" ? r.algorithm : k === "seed" ? r.seed : k === "cv_folds" ? r.cv_folds : r.hyperparameters[k];
                          return <td key={r.id} className={cn("text-right font-mono text-xs", differs ? "text-steel" : "text-white/40")}>{v === undefined ? "—" : String(v)}</td>;
                        })}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </Panel>
        </div>
      )}
    </>
  );
}
