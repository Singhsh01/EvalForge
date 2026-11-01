"use client";

import { AXIS, EChart, LEGEND, SERIES, TOOLTIP } from "@/components/echart";
import { num } from "@/lib/format";

export { SERIES };

export function ConfusionMatrix({ matrix, labels }: { matrix: number[][]; labels: string[] }) {
  const max = Math.max(...matrix.flat(), 1);
  const total = matrix.flat().reduce((a, b) => a + b, 0);
  return (
    <div className="overflow-x-auto">
      <table className="mx-auto border-separate border-spacing-1 text-sm">
        <thead>
          <tr>
            <th className="p-1 text-right text-xs font-normal text-white/40">actual ↓ / predicted →</th>
            {labels.map((l) => <th key={l} className="px-2 pb-1 text-xs font-medium text-white/60">{l}</th>)}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={labels[i]}>
              <th className="pr-2 text-right text-xs font-medium text-white/60">{labels[i]}</th>
              {row.map((v, j) => {
                const a = v / max;
                return (
                  <td key={j} className="h-16 min-w-20 rounded-lg text-center tabular-nums"
                    style={{ background: i === j ? `rgb(95 211 162 / ${0.08 + a * 0.5})` : `rgb(242 122 133 / ${v ? 0.08 + a * 0.6 : 0.03})` }}
                    title={`${labels[i]} predicted as ${labels[j]}: ${v}`}>
                    <div className="text-base font-semibold text-white">{v}</div>
                    <div className="text-[11px] text-white/50">{((v / total) * 100).toFixed(1)}%</div>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function RocCurve({ fpr, tpr, auc }: { fpr: number[]; tpr: number[]; auc?: number }) {
  return (
    <EChart height={256} ariaLabel={`ROC curve, AUC ${num(auc, 3)}`} option={{
      grid: { left: 40, right: 12, top: 28, bottom: 36 },
      legend: LEGEND, tooltip: { ...TOOLTIP, trigger: "axis", valueFormatter: (v: unknown) => num(Number(v), 3) },
      xAxis: { type: "value", min: 0, max: 1, name: "False positive rate", nameLocation: "middle", nameGap: 24, ...AXIS },
      yAxis: { type: "value", min: 0, max: 1, ...AXIS },
      series: [
        { name: `TPR${auc ? ` (AUC ${num(auc, 3)})` : ""}`, type: "line", step: "end", showSymbol: false,
          data: fpr.map((x, i) => [x, tpr[i]]), lineStyle: { width: 2 }, areaStyle: { opacity: 0.08 } },
        { name: "chance", type: "line", showSymbol: false, data: [[0, 0], [1, 1]],
          lineStyle: { type: "dashed", color: "rgba(255,255,255,0.25)", width: 1 }, tooltip: { show: false } },
      ],
    }} />
  );
}

export function ImportanceBars({ rows }: { rows: { feature: string; importance: number; std: number }[] }) {
  const data = rows.slice(0, 12).reverse();
  return (
    <EChart height={Math.max(180, data.length * 26 + 40)} ariaLabel="Permutation feature importance" option={{
      grid: { left: 170, right: 24, top: 8, bottom: 24 },
      tooltip: { ...TOOLTIP, trigger: "axis", formatter: (p: { dataIndex: number }[]) => {
        const r = data[p[0].dataIndex]; return `${r.feature}<br/>ΔF1 ${num(r.importance, 4)} ± ${num(r.std, 4)}`; } },
      xAxis: { type: "value", ...AXIS },
      yAxis: { type: "category", data: data.map((r) => r.feature), ...AXIS, axisLabel: { ...AXIS.axisLabel, color: "rgba(255,255,255,0.75)" } },
      series: [{ type: "bar", data: data.map((r) => ({ value: r.importance, itemStyle: { color: r.importance > 0 ? "#8fb4dc" : "rgba(255,255,255,0.2)", borderRadius: [0, 4, 4, 0] } })) },
        { type: "scatter", symbol: "rect", symbolSize: [1, 10], data: data.map((r, i) => [r.importance + r.std, i]), itemStyle: { color: "rgba(255,255,255,0.45)" }, tooltip: { show: false } }],
    }} />
  );
}

export function Histogram({ counts, edges }: { counts: number[]; edges: (number | null)[] }) {
  return (
    <EChart height={80} ariaLabel="Value distribution" option={{
      grid: { left: 0, right: 0, top: 4, bottom: 0 },
      tooltip: { ...TOOLTIP, trigger: "axis", formatter: (p: { dataIndex: number; value: number }[]) =>
        `${num(edges[p[0].dataIndex], 3)} – ${num(edges[p[0].dataIndex + 1], 3)}<br/>${p[0].value} rows` },
      xAxis: { type: "category", show: false, data: counts.map((_, i) => i) },
      yAxis: { type: "value", show: false },
      series: [{ type: "bar", data: counts, barCategoryGap: "8%", itemStyle: { color: "#8fb4dc", borderRadius: [2, 2, 0, 0] } }],
    }} />
  );
}

export function GroupedMetricBars({ data, series }: { data: Record<string, number | string>[]; series: { key: string; name: string }[] }) {
  const vals = data.flatMap((d) => series.map((s) => Number(d[s.key]) || 0));
  return (
    <EChart height={288} ariaLabel="Holdout metrics by run" option={{
      grid: { left: 44, right: 8, top: 36, bottom: 28 },
      legend: LEGEND, tooltip: { ...TOOLTIP, trigger: "axis", valueFormatter: (v: unknown) => num(Number(v), 4) },
      xAxis: { type: "category", data: data.map((d) => d.metric), ...AXIS },
      yAxis: { type: "value", min: Math.max(0, Math.floor((Math.min(...vals) - 0.02) * 100) / 100), max: 1, ...AXIS },
      series: series.map((s) => ({ type: "bar", name: s.name, data: data.map((d) => d[s.key]), itemStyle: { borderRadius: [3, 3, 0, 0] } })),
    }} />
  );
}
