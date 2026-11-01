"use client";

import "@xyflow/react/dist/style.css";
import {
  Background, BackgroundVariant, Controls, Handle, MarkerType, Position, ReactFlow,
  type Edge, type Node, type NodeProps,
} from "@xyflow/react";
import { useReducedMotion } from "motion/react";
import { useMemo } from "react";
import { cn } from "@/lib/format";

export type DagNode = { name: string; depends_on: string[]; status?: string; attempts?: number; detail?: string };

const COLOR: Record<string, string> = {
  succeeded: "border-pass/60 bg-pass/10 text-pass",
  running: "border-ember bg-ember/15 text-ember",
  failed: "border-fail/70 bg-fail/10 text-fail",
  waiting_approval: "border-warn/70 bg-warn/10 text-warn",
  skipped: "border-white/10 bg-white/[0.02] text-white/30",
  pending: "border-white/15 bg-white/[0.03] text-white/50",
};
const EDGE: Record<string, string> = { succeeded: "#5fd3a2", running: "#f5b04c", failed: "#f27a85", waiting_approval: "#f0c35b" };

type StepData = { node: DagNode; selected: boolean };
const W = 128, H = 54, GX = 26, GY = 18;

function StepNode({ data }: NodeProps<Node<StepData>>) {
  const n = data.node;
  const status = n.status ?? "pending";
  return (
    <div className={cn("glass-node rounded-xl border px-3 py-2 text-left backdrop-blur-md transition-shadow",
      COLOR[status] ?? COLOR.pending, data.selected && "ring-2 ring-white/50 shadow-[0_0_24px_rgba(255,255,255,0.12)]")}
      style={{ width: W, height: H }}>
      <Handle type="target" position={Position.Left} className="!size-1.5 !border-0 !bg-white/30" />
      <div className="truncate text-[12px] font-medium text-white" title={n.name}>{n.name.replaceAll("_", " ")}</div>
      <div className="mt-0.5 flex items-center gap-1.5 text-[11px]">
        {status === "running" && <span className="size-1.5 animate-pulse rounded-full bg-current" />}
        {status.replaceAll("_", " ")}{n.attempts && n.attempts > 1 ? ` · ${n.attempts} attempts` : ""}
      </div>
      <Handle type="source" position={Position.Right} className="!size-1.5 !border-0 !bg-white/30" />
    </div>
  );
}
const nodeTypes = { step: StepNode };

/** Layered layout: column = longest dependency chain to the node. */
function layout(nodes: DagNode[]) {
  const depth: Record<string, number> = {};
  const d = (n: DagNode): number => {
    if (depth[n.name] !== undefined) return depth[n.name];
    const parents = n.depends_on.map((p) => nodes.find((x) => x.name === p)).filter(Boolean) as DagNode[];
    depth[n.name] = parents.length ? 1 + Math.max(...parents.map(d)) : 0;
    return depth[n.name];
  };
  nodes.forEach(d);
  const rows: Record<number, number> = {};
  return Object.fromEntries(nodes.map((n) => {
    const c = depth[n.name]; const r = rows[c] ?? 0; rows[c] = r + 1;
    return [n.name, { x: c * (W + GX), y: r * (H + GY) }];
  }));
}

/** Interactive pipeline DAG (React Flow): pan/zoom, click a step to inspect it, live edges while running. */
export function Dag({ nodes, onSelect, selected }: { nodes: DagNode[]; onSelect?: (n: string) => void; selected?: string }) {
  const reduce = useReducedMotion();
  const { rfNodes, rfEdges } = useMemo(() => {
    const pos = layout(nodes);
    const rfNodes: Node<StepData>[] = nodes.map((n) => ({
      id: n.name, type: "step", position: pos[n.name], data: { node: n, selected: selected === n.name },
      draggable: false, connectable: false, ariaLabel: `${n.name} ${n.status ?? "pending"}`,
    }));
    const rfEdges: Edge[] = nodes.flatMap((n) => n.depends_on.map((p) => {
      const parent = nodes.find((x) => x.name === p);
      const live = n.status === "running";
      const color = EDGE[parent?.status ?? ""] ?? "rgba(255,255,255,0.18)";
      return {
        id: `${p}->${n.name}`, source: p, target: n.name, type: "smoothstep", animated: live && !reduce,
        style: { stroke: color, strokeWidth: 1.5 }, markerEnd: { type: MarkerType.ArrowClosed, color, width: 14, height: 14 },
      };
    }));
    return { rfNodes, rfEdges };
  }, [nodes, selected, reduce]);
  const pos = layout(nodes);
  const rows = new Set(Object.values(pos).map((p) => p.y)).size;
  const height = Math.max(140, Math.min(360, rows * (H + GY) + 70));
  return (
    <div className="flow-dark w-full overflow-hidden rounded-xl border border-white/5 bg-black/20" style={{ height }}>
      <ReactFlow nodes={rfNodes} edges={rfEdges} nodeTypes={nodeTypes} fitView fitViewOptions={{ padding: 0.06, maxZoom: 1.25 }}
        onNodeClick={(_, n) => onSelect?.(n.id)} proOptions={{ hideAttribution: true }}
        nodesConnectable={false} elementsSelectable minZoom={0.4} maxZoom={1.6} panOnScroll={false} zoomOnScroll={false}
        colorMode="dark">
        <Background variant={BackgroundVariant.Dots} gap={18} size={1} color="rgba(255,255,255,0.06)" />
        <Controls showInteractive={false} showZoom={false} position="top-left" orientation="horizontal" />
      </ReactFlow>
    </div>
  );
}

type LinNode = { id: string; kind: string; label: string; sub?: string; href?: string };
const KIND_COLOR: Record<string, string> = {
  dataset: "#8fb4dc", run: "#f5b04c", mlflow: "#5fd3a2", model: "#c3a6ff", deployment: "#f27a85", approval: "#f0c35b", commit: "#9aa3b8",
};

function LineageNode({ data }: NodeProps<Node<LinNode>>) {
  const c = KIND_COLOR[data.kind] ?? "#fff";
  const body = (
    <div className="rounded-xl border bg-white/[0.04] px-3 py-2 backdrop-blur-md" style={{ width: 160, borderColor: `${c}80` }}>
      <Handle type="target" position={Position.Left} className="!size-1.5 !border-0" style={{ background: c }} />
      <div className="text-[10px] uppercase tracking-wider" style={{ color: c }}>{data.kind}</div>
      <div className="truncate text-[12px] font-medium text-white" title={data.label}>{data.label}</div>
      {data.sub && <div className="truncate font-mono text-[10px] text-white/45" title={data.sub}>{data.sub}</div>}
      <Handle type="source" position={Position.Right} className="!size-1.5 !border-0" style={{ background: c }} />
    </div>
  );
  return data.href ? <a href={data.href}>{body}</a> : body;
}
const lineageTypes = { lin: LineageNode };

/** Provenance chain of a pipeline run: data → run → MLflow → model version → deployment → approval. */
export function LineageGraph({ lineage }: { lineage: Record<string, unknown> }) {
  const s = (k: string) => (lineage[k] === undefined || lineage[k] === null ? undefined : String(lineage[k]));
  const chain: LinNode[] = [];
  if (s("dataset_version_id")) chain.push({ id: "dataset", kind: "dataset", label: s("dataset") ? `${s("dataset")}${s("dataset_version") ? ` v${s("dataset_version")}` : ""}` : "dataset version", sub: s("dataset_sha256")?.slice(0, 16) ?? s("dataset_version_id")?.slice(0, 8), href: "/datasets" });
  if (s("run_id")) chain.push({ id: "run", kind: "run", label: "training run", sub: `${s("run_id")!.slice(0, 8)}${s("git_commit") ? ` · git ${s("git_commit")!.slice(0, 7)}` : ""}`, href: `/experiments/${s("run_id")}` });
  if (s("mlflow_run_id")) chain.push({ id: "mlflow", kind: "mlflow", label: "MLflow run", sub: s("mlflow_run_id")!.slice(0, 12) });
  if (s("model_version_id")) chain.push({ id: "model", kind: "model", label: `${s("name") ?? "model"}${s("version") ? ` v${s("version")}` : ""}`, sub: s("model_version_id")!.slice(0, 8), href: "/registry" });
  if (s("deployment_id")) chain.push({ id: "deployment", kind: "deployment", label: "staging deployment", sub: s("deployment_id")!.slice(0, 8), href: "/deployments" });
  if (s("approval_id")) chain.push({ id: "approval", kind: "approval", label: "promotion approval", sub: s("approval_id")!.slice(0, 8), href: "/approvals" });
  if (!chain.length) return null;
  const nodes: Node<LinNode>[] = chain.map((c, i) => ({ id: c.id, type: "lin", position: { x: i * 186, y: 0 }, data: c, draggable: false }));
  const edges: Edge[] = chain.slice(1).map((c, i) => ({ id: `${chain[i].id}-${c.id}`, source: chain[i].id, target: c.id, type: "smoothstep",
    style: { stroke: "rgba(255,255,255,0.28)", strokeWidth: 1.4 }, markerEnd: { type: MarkerType.ArrowClosed, color: "rgba(255,255,255,0.4)" } }));
  return (
    <div className="flow-dark h-[120px] w-full overflow-hidden rounded-xl border border-white/5 bg-black/20">
      <ReactFlow nodes={nodes} edges={edges} nodeTypes={lineageTypes} fitView fitViewOptions={{ padding: 0.05, maxZoom: 1.3 }}
        proOptions={{ hideAttribution: true }} nodesConnectable={false} zoomOnScroll={false} panOnScroll={false} colorMode="dark">
        <Background variant={BackgroundVariant.Dots} gap={18} size={1} color="rgba(255,255,255,0.05)" />
      </ReactFlow>
    </div>
  );
}
