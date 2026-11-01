"use client";

import { Histogram } from "@/components/charts";
import { PageHeader } from "@/components/shell";
import { Button, ErrorNote, Mono, Panel, Stat, Status } from "@/components/ui";
import { api, type Check } from "@/lib/api";
import { ago, bytes, cn, num } from "@/lib/format";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CircleAlert, CircleCheck, CircleX, Info } from "lucide-react";
import Link from "next/link";
import { use, useEffect, useRef, useState } from "react";

export default function DatasetPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const qc = useQueryClient();
  const ds = useQuery({ queryKey: ["dataset", id], queryFn: () => api.dataset(id) });
  const [sel, setSel] = useState<number | null>(null);
  const latest = ds.data?.versions.at(-1)?.version ?? null;
  const v = sel ?? latest;
  useEffect(() => { if (sel === null && latest) setSel(latest); }, [latest, sel]);

  const detail = useQuery({ queryKey: ["version", id, v], queryFn: () => api.version(id, v!), enabled: !!v });
  const preview = useQuery({ queryKey: ["preview", id, v], queryFn: () => api.preview(id, v!), enabled: !!v });
  const fileRef = useRef<HTMLInputElement>(null);
  const upload = useMutation({
    mutationFn: (f: File) => { const fd = new FormData(); fd.append("file", f); return api.addVersion(id, fd); },
    onSuccess: (nv) => { qc.invalidateQueries({ queryKey: ["dataset", id] }); setSel(nv.version); },
  });

  const d = detail.data;
  const checks = d?.quality_checks ?? [];
  const order = { error: 0, warning: 1, info: 2 } as const;

  return (
    <>
      <PageHeader title={ds.data?.name ?? "Dataset"}
        description={ds.data?.description || (ds.data && `Target column “${ds.data.target_column}”`)}
        action={
          <div className="flex gap-2">
            <input ref={fileRef} type="file" accept=".csv,.json,.jsonl" className="hidden"
              onChange={(e) => { const f = e.target.files?.[0]; if (f) upload.mutate(f); e.target.value = ""; }} />
            <Button variant="outline" loading={upload.isPending} onClick={() => fileRef.current?.click()}>Upload new version</Button>
            {d?.status === "validated" && <Link href={`/experiments?version=${d.id}`}><Button>Train on v{d.version}</Button></Link>}
          </div>
        } />
      <ErrorNote error={ds.error ?? upload.error} />

      {ds.data && (
        <div className="mb-6 flex flex-wrap gap-2" role="tablist" aria-label="Versions">
          {ds.data.versions.map((x) => (
            <button key={x.id} role="tab" aria-selected={x.version === v} onClick={() => setSel(x.version)}
              className={cn("rounded-lg border px-3 py-1.5 text-left text-sm transition-colors",
                x.version === v ? "border-ember/60 bg-ember/10 text-white" : "border-white/10 text-white/60 hover:border-white/25")}>
              <span className="font-medium">v{x.version}</span>
              <span className="ml-2 text-xs text-white/45">{x.n_rows.toLocaleString()} rows · {ago(x.created_at)}</span>
              {x.status === "rejected" && <span className="ml-2 text-xs text-fail">rejected</span>}
            </button>
          ))}
        </div>
      )}

      {d && (
        <div className="grid gap-6 xl:grid-cols-3">
          <Panel className="xl:col-span-3">
            <div className="grid grid-cols-2 gap-6 md:grid-cols-6">
              <Stat label="Status" value={<Status value={d.status} />} />
              <Stat label="Rows" value={d.n_rows.toLocaleString()} sub={`${d.profile.duplicate_rows} duplicates`} />
              <Stat label="Columns" value={d.n_columns} sub={d.task_type?.replaceAll("_", " ") ?? "unsupported target"} />
              <Stat label="Holdout split" value={d.split ? `${d.split.n_train} / ${d.split.n_test}` : "—"} sub={d.split ? `stratified · seed ${d.split.seed}` : "not created"} />
              <Stat label="Size" value={bytes(d.size_bytes)} sub={`${d.file_format.toUpperCase()} · ${d.source_filename}`} />
              <Stat label="Content hash" value={<Mono className="text-sm">{d.sha256.slice(0, 12)}</Mono>} sub="SHA-256 of the upload" />
            </div>
          </Panel>

          <Panel className="xl:col-span-1" title="Quality and leakage checks"
            subtitle={`${d.checks_failed} blocking · ${d.checks_warned} warnings`}>
            <ul className="space-y-3">
              {[...checks].sort((a, b) => Number(a.passed) - Number(b.passed) || order[a.severity] - order[b.severity]).map((c) => (
                <CheckRow key={c.id} c={c} />
              ))}
            </ul>
          </Panel>

          <Panel className="xl:col-span-2" title="Column profile">
            {d.profile.target && (
              <div className="mb-5">
                <div className="mb-2 text-[13px] text-white/50">Label distribution — {d.profile.target.column}</div>
                <div className="flex h-3 overflow-hidden rounded-full">
                  {d.profile.target.classes.map((c, i) => (
                    <div key={c.label} style={{ flex: c.count, background: ["#f5b04c", "#8fb4dc", "#5fd3a2", "#c9a7f5", "#f27a85"][i % 5] }} title={`${c.label}: ${c.count}`} />
                  ))}
                </div>
                <div className="mt-2 flex flex-wrap gap-4 text-xs text-white/60">
                  {d.profile.target.classes.map((c) => <span key={c.label}>{c.label} <span className="tabular-nums text-white/40">{c.count} ({((c.count / d.n_rows) * 100).toFixed(1)}%)</span></span>)}
                </div>
              </div>
            )}
            <div className="grid gap-3 sm:grid-cols-2 2xl:grid-cols-3">
              {Object.entries(d.profile.columns).filter(([n]) => n !== d.profile.target?.column).map(([name, p]) => (
                <div key={name} className="rounded-xl border border-white/5 bg-white/[0.02] p-3">
                  <div className="flex items-baseline justify-between gap-2">
                    <span className="truncate text-sm font-medium text-white" title={name}>{name}</span>
                    <span className="text-[11px] text-white/40">{p.kind}</span>
                  </div>
                  <div className="mt-0.5 text-[11px] text-white/45 tabular-nums">
                    {p.unique} unique{p.missing ? <span className="text-warn"> · {(p.missing_ratio * 100).toFixed(1)}% missing</span> : ""}
                    {p.stats && <> · μ {num(p.stats.mean, 3)}</>}
                  </div>
                  {p.histogram ? <Histogram counts={p.histogram.counts} edges={p.histogram.edges} /> : (
                    <ul className="mt-2 space-y-1">
                      {p.top_values?.slice(0, 4).map((t) => (
                        <li key={t.value} className="flex items-center gap-2 text-[11px]">
                          <span className="w-20 truncate text-white/60">{t.value}</span>
                          <span className="h-1.5 rounded bg-steel/70" style={{ width: `${(t.count / d.n_rows) * 100}%` }} />
                          <span className="tabular-nums text-white/40">{t.count}</span>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              ))}
            </div>
          </Panel>

          <Panel className="xl:col-span-3" title="Preview" subtitle="First 25 rows of the stored canonical copy">
            <div className="max-h-96 overflow-auto">
              <table className="text-xs">
                <thead className="sticky top-0 bg-ink-900"><tr>{preview.data?.columns.map((c) => <th key={c} className="whitespace-nowrap px-2 py-1.5 text-left font-medium text-white/60">{c}</th>)}</tr></thead>
                <tbody>{preview.data?.rows.map((r, i) => (
                  <tr key={i} className="odd:bg-white/[0.02]">{r.map((cell, j) => <td key={j} className="whitespace-nowrap px-2 py-1 tabular-nums text-white/75">{cell === null ? <span className="text-white/25">null</span> : String(cell)}</td>)}</tr>
                ))}</tbody>
              </table>
            </div>
          </Panel>
        </div>
      )}
    </>
  );
}

function CheckRow({ c }: { c: Check }) {
  const Icon = c.passed ? CircleCheck : c.severity === "error" ? CircleX : c.severity === "warning" ? CircleAlert : Info;
  const color = c.passed ? "text-pass" : c.severity === "error" ? "text-fail" : c.severity === "warning" ? "text-warn" : "text-steel";
  const suspects = c.details.suspects as Record<string, string> | undefined;
  return (
    <li className="flex gap-2.5">
      <Icon className={cn("mt-0.5 size-4 shrink-0", color)} aria-label={c.passed ? "passed" : c.severity} />
      <div className="min-w-0">
        <div className="text-sm text-white">{c.title}</div>
        <div className="text-xs leading-relaxed text-white/50">{c.message}</div>
        {suspects && Object.keys(suspects).length > 0 && (
          <ul className="mt-1 text-xs text-warn">{Object.entries(suspects).map(([k, why]) => <li key={k}>{k}: {why}</li>)}</ul>
        )}
      </div>
    </li>
  );
}
