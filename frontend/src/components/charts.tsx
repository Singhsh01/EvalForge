"use client";

import { num } from "@/lib/format";
import {
  Bar, BarChart, CartesianGrid, Cell, ErrorBar, Legend, Line, LineChart, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";

const AXIS = { stroke: "rgb(255 255 255 / 0.35)", fontSize: 11 };
const GRID = "rgb(255 255 255 / 0.06)";
export const SERIES = ["#f5b04c", "#8fb4dc", "#5fd3a2", "#c9a7f5", "#f27a85", "#7fd6e0", "#e6e6e6", "#b9c76a"];

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
                const diag = i === j;
                return (
                  <td key={j}
                    className="h-16 min-w-20 rounded-lg text-center tabular-nums"
                    style={{
                      background: diag ? `rgb(95 211 162 / ${0.08 + a * 0.5})` : `rgb(242 122 133 / ${v ? 0.08 + a * 0.6 : 0.03})`,
                    }}
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
  const data = fpr.map((x, i) => ({ fpr: x, tpr: tpr[i] }));
  return (
    <div className="h-64">
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 8, right: 12, bottom: 18, left: 0 }}>
          <CartesianGrid stroke={GRID} />
          <XAxis type="number" dataKey="fpr" domain={[0, 1]} {...AXIS} label={{ value: "False positive rate", position: "insideBottom", offset: -10, fill: "rgb(255 255 255 / .45)", fontSize: 11 }} />
          <YAxis type="number" domain={[0, 1]} {...AXIS} width={36} />
          <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="rgb(255 255 255 / 0.2)" strokeDasharray="4 4" />
          <Tooltip formatter={(v) => num(Number(v), 3)} labelFormatter={(v) => `FPR ${num(Number(v), 3)}`} />
          <Line type="stepAfter" dataKey="tpr" name={`TPR${auc ? ` (AUC ${num(auc, 3)})` : ""}`} stroke="#f5b04c" strokeWidth={2} dot={false} />
          <Legend verticalAlign="top" height={24} wrapperStyle={{ fontSize: 12 }} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function ImportanceBars({ rows }: { rows: { feature: string; importance: number; std: number }[] }) {
  const data = rows.slice(0, 12).map((r) => ({ ...r, err: [r.std, r.std] }));
  return (
    <div style={{ height: Math.max(160, data.length * 26 + 30) }}>
      <ResponsiveContainer>
        <BarChart data={data} layout="vertical" margin={{ left: 8, right: 16 }}>
          <CartesianGrid stroke={GRID} horizontal={false} />
          <XAxis type="number" {...AXIS} />
          <YAxis type="category" dataKey="feature" width={170} {...AXIS} tick={{ fill: "rgb(255 255 255 / .7)", fontSize: 11 }} />
          <Tooltip formatter={(v) => num(Number(v), 4)} cursor={{ fill: "rgb(255 255 255 / 0.04)" }} />
          <Bar dataKey="importance" name="F1 drop when shuffled" radius={[0, 4, 4, 0]}>
            {data.map((d) => <Cell key={d.feature} fill={d.importance > 0 ? "#8fb4dc" : "rgb(255 255 255 / 0.2)"} />)}
            <ErrorBar dataKey="err" width={4} stroke="rgb(255 255 255 / 0.5)" direction="x" />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function Histogram({ counts, edges }: { counts: number[]; edges: (number | null)[] }) {
  const data = counts.map((c, i) => ({ c, x: edges[i] ?? 0, x2: edges[i + 1] ?? 0 }));
  return (
    <div className="h-20">
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 2, right: 0, bottom: 0, left: 0 }} barCategoryGap={1}>
          <Tooltip cursor={{ fill: "rgb(255 255 255 / 0.05)" }}
            formatter={(v) => [v, "rows"]}
            labelFormatter={(_, p) => p?.[0] ? `${num(p[0].payload.x, 3)} – ${num(p[0].payload.x2, 3)}` : ""} />
          <Bar dataKey="c" fill="#8fb4dc" radius={[2, 2, 0, 0]} isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function GroupedMetricBars({ data, series }: { data: Record<string, number | string>[]; series: { key: string; name: string }[] }) {
  return (
    <div className="h-72">
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid stroke={GRID} vertical={false} />
          <XAxis dataKey="metric" {...AXIS} tick={{ fill: "rgb(255 255 255 / .7)", fontSize: 11 }} />
          <YAxis domain={["dataMin - 0.02", 1]} {...AXIS} width={44} tickFormatter={(v) => Number(v).toFixed(2)} allowDataOverflow />
          <Tooltip formatter={(v) => num(Number(v), 4)} cursor={{ fill: "rgb(255 255 255 / 0.04)" }} />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          {series.map((s, i) => <Bar key={s.key} dataKey={s.key} name={s.name} fill={SERIES[i % SERIES.length]} radius={[3, 3, 0, 0]} />)}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
