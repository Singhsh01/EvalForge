"use client";

import { cn } from "@/lib/format";
import { motion } from "motion/react";

export type DagNode = { name: string; depends_on: string[]; status?: string; attempts?: number; detail?: string };

const COLOR: Record<string, string> = {
  succeeded: "border-pass/60 bg-pass/10 text-pass",
  running: "border-ember bg-ember/15 text-ember",
  failed: "border-fail/70 bg-fail/10 text-fail",
  waiting_approval: "border-warn/70 bg-warn/10 text-warn",
  skipped: "border-white/10 bg-white/[0.02] text-white/30",
  pending: "border-white/15 bg-white/[0.03] text-white/50",
};

/** Layered DAG layout: each node's column = longest dependency chain to it. */
export function Dag({ nodes, onSelect, selected }: { nodes: DagNode[]; onSelect?: (n: string) => void; selected?: string }) {
  const depth: Record<string, number> = {};
  const d = (n: DagNode): number => {
    if (depth[n.name] !== undefined) return depth[n.name];
    depth[n.name] = n.depends_on.length ? 1 + Math.max(...n.depends_on.map((p) => d(nodes.find((x) => x.name === p)!))) : 0;
    return depth[n.name];
  };
  nodes.forEach(d);
  const cols = Math.max(...Object.values(depth), 0) + 1;
  const W = 100, H = 58, GX = 12;
  const rows: Record<number, number> = {};
  const pos = Object.fromEntries(nodes.map((n) => {
    const c = depth[n.name]; const r = rows[c] ?? 0; rows[c] = r + 1;
    return [n.name, { x: c * (W + GX), y: r * (H + 24) }];
  }));
  const height = (Math.max(...Object.values(rows), 1)) * (H + 24);
  return (
    <div className="overflow-x-auto pb-2">
      <div className="relative" style={{ width: cols * (W + GX) - GX, height }}>
        <svg className="absolute inset-0" width={cols * (W + GX)} height={height} aria-hidden>
          {nodes.flatMap((n) => n.depends_on.map((p) => {
            const a = pos[p], b = pos[n.name];
            const done = n.status && n.status !== "pending" && n.status !== "skipped";
            return (
              <path key={`${p}-${n.name}`} d={`M${a.x + W},${a.y + H / 2} C${a.x + W + GX / 2},${a.y + H / 2} ${b.x - GX / 2},${b.y + H / 2} ${b.x},${b.y + H / 2}`}
                fill="none" stroke={done ? "rgb(245 176 76 / .6)" : "rgb(255 255 255 / .15)"} strokeWidth={1.5} />
            );
          }))}
        </svg>
        {nodes.map((n, i) => (
          <motion.button key={n.name} type="button" onClick={() => onSelect?.(n.name)}
            initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.04 }}
            className={cn("absolute rounded-xl border px-3 py-2 text-left", COLOR[n.status ?? "pending"] ?? COLOR.pending,
              selected === n.name && "ring-2 ring-white/40")}
            style={{ left: pos[n.name].x, top: pos[n.name].y, width: W, height: H }}>
            <div className="truncate text-[12px] font-medium text-white" title={n.name}>{n.name.replaceAll("_", " ")}</div>
            <div className="mt-0.5 flex items-center gap-1.5 text-[11px]">
              {n.status === "running" && <span className="size-1.5 animate-pulse rounded-full bg-current" />}
              {(n.status ?? "pending").replaceAll("_", " ")}{n.attempts && n.attempts > 1 ? ` · ${n.attempts} attempts` : ""}
            </div>
          </motion.button>
        ))}
      </div>
    </div>
  );
}
