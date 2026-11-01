"use client";

import { PageHeader } from "@/components/shell";
import { Button, Empty, ErrorNote, Field, inputCls, JsonBlock, Mono, Panel, Status } from "@/components/ui";
import { api, type J } from "@/lib/api";
import { ago, cn } from "@/lib/format";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";

export default function PromptsPage() {
  const qc = useQueryClient();
  const prompts = useQuery({ queryKey: ["prompts"], queryFn: api.prompts });
  const runs = useQuery({ queryKey: ["eval-runs"], queryFn: api.evalRuns });
  const [name, setName] = useState<string | null>(null);
  const [form, setForm] = useState<J>({ name: "", system: "", template: "", output_schema: "", note: "" });
  const [diff, setDiff] = useState<string | null>(null);
  useEffect(() => { if (!name && prompts.data?.length) setName(prompts.data[0].name); }, [prompts.data, name]);
  const p = prompts.data?.find((x) => x.name === name);

  const save = useMutation({
    mutationFn: () => {
      const body = { system: form.system, template: form.template, note: form.note,
        output_schema: form.output_schema ? JSON.parse(form.output_schema) : null };
      return p && !form.name ? api.addPromptVersion(p.name, body) : api.createPrompt({ ...body, name: form.name });
    },
    onSuccess: (v: J) => { setName(v.prompt); qc.invalidateQueries(); },
  });
  const showDiff = useMutation({ mutationFn: ({ a, b }: { a: number; b: number }) => api.promptDiff(p!.name, a, b), onSuccess: (d: J) => setDiff(d.diff) });
  const release = useMutation({ mutationFn: ({ pv, run }: { pv: string; run: string }) => api.requestRelease(pv, run), onSuccess: () => qc.invalidateQueries() });

  return (
    <>
      <PageHeader title="Prompt registry"
        description="Prompt templates with typed variables and optional JSON output schemas. Versions are immutable and content-hashed; a version is released only through an approval that cites a passing, non-regressed evaluation run of that exact version." />
      <ErrorNote error={prompts.error ?? save.error ?? release.error} />
      <div className="grid gap-6 xl:grid-cols-3">
        <Panel title="Prompts">
          {!prompts.data?.length ? <Empty title="No prompts yet" /> : (
            <ul className="space-y-1">
              {prompts.data.map((x) => (
                <li key={x.id}><button onClick={() => { setName(x.name); setDiff(null); }} className={cn("w-full rounded-lg px-2 py-1.5 text-left text-sm", x.name === name ? "bg-white/[0.06] text-white" : "text-white/70 hover:bg-white/[0.03]")}>{x.name} <span className="text-white/40">· {x.versions.length} versions</span></button></li>
              ))}
            </ul>
          )}
          <div className="mt-5 space-y-3 border-t border-white/5 pt-4">
            <div className="text-sm font-medium text-white">{p ? `New version of ${p.name} (or a new prompt)` : "New prompt"}</div>
            <Field label="New prompt name" hint={p ? "leave empty to add a version to the selected prompt" : undefined}><input className={inputCls} value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} /></Field>
            <Field label="System"><textarea className={inputCls} rows={2} value={form.system} onChange={(e) => setForm({ ...form, system: e.target.value })} /></Field>
            <Field label="Template" hint="Use {{variable}} placeholders; nothing else is evaluated."><textarea className={inputCls} rows={4} value={form.template} onChange={(e) => setForm({ ...form, template: e.target.value })} /></Field>
            <Field label="Output JSON Schema (optional)"><textarea className={cn(inputCls, "font-mono text-xs")} rows={3} value={form.output_schema} onChange={(e) => setForm({ ...form, output_schema: e.target.value })} /></Field>
            <Field label="Change note"><input className={inputCls} value={form.note} onChange={(e) => setForm({ ...form, note: e.target.value })} /></Field>
            <Button loading={save.isPending} disabled={!form.template} onClick={() => save.mutate()}>Save version</Button>
          </div>
        </Panel>
        <Panel className="xl:col-span-2" title={p?.name ?? "Versions"} subtitle={p?.description}>
          {p?.versions.slice().reverse().map((v: J, i: number, arr: J[]) => {
            const evals = (runs.data ?? []).filter((r) => r.subject?.prompt_version_id === v.id && r.status === "succeeded");
            return (
              <div key={v.id} className="mb-4 rounded-xl border border-white/5 bg-white/[0.02] p-4">
                <div className="flex flex-wrap items-center gap-2 text-sm">
                  <span className="font-medium text-white">v{v.version}</span><Status value={v.status} />
                  <span className="text-white/50">{v.note}</span>
                  <span className="ml-auto text-xs text-white/40">{v.created_by} · {ago(v.created_at)} · <Mono>{v.sha256.slice(0, 10)}</Mono></span>
                </div>
                <pre className="mt-2 whitespace-pre-wrap rounded-lg bg-ink-950/60 p-2 font-mono text-[11px] text-white/70">{v.system ? `[system] ${v.system}\n` : ""}{v.template}</pre>
                <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-white/50">
                  variables: {v.variables.join(", ") || "none"}
                  {i < arr.length - 1 && <button className="text-ember" onClick={() => showDiff.mutate({ a: arr[i + 1].version, b: v.version })}>Diff vs v{arr[i + 1].version}</button>}
                </div>
                {evals.length > 0 && (
                  <div className="mt-2 space-y-1 text-xs">
                    {evals.slice(0, 3).map((r) => (
                      <div key={r.id} className="flex items-center gap-2 text-white/60">
                        <span>{r.suite} · {Math.round(r.summary.pass_rate * 100)}% · gates {r.summary.gates_passed ? "passed" : "failed"}{r.regression?.detected ? " · regressed" : ""}</span>
                        {v.status !== "released" && r.summary.gates_passed && !r.regression?.detected && (
                          <button className="text-ember" onClick={() => release.mutate({ pv: v.id, run: r.id })}>Request release with this run</button>
                        )}
                      </div>
                    ))}
                  </div>
                )}
                {v.output_schema && <details className="mt-2"><summary className="cursor-pointer text-xs text-white/40">Output schema</summary><JsonBlock value={v.output_schema} /></details>}
              </div>
            );
          })}
          {diff !== null && <pre className="whitespace-pre-wrap rounded-lg bg-ink-950/70 p-3 font-mono text-[11px]">{diff.split("\n").map((l, i) => <div key={i} className={l.startsWith("+") ? "text-pass" : l.startsWith("-") ? "text-fail" : "text-white/60"}>{l}</div>)}</pre>}
        </Panel>
      </div>
    </>
  );
}
