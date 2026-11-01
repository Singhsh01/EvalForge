"use client";

import { PageHeader } from "@/components/shell";
import { Button, Empty, ErrorNote, Field, inputCls, Panel, Status } from "@/components/ui";
import { api, type Algorithm, type Param } from "@/lib/api";
import { ago, ALGO_LABEL, cn, num } from "@/lib/format";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useMemo, useState } from "react";

export default function ExperimentsPage() {
  return <Suspense><Experiments /></Suspense>;
}

function Experiments() {
  const qc = useQueryClient();
  const router = useRouter();
  const search = useSearchParams();
  const meta = useQuery({ queryKey: ["meta"], queryFn: api.meta });
  const datasets = useQuery({ queryKey: ["datasets"], queryFn: api.datasets });
  const runs = useQuery({
    queryKey: ["runs"], queryFn: () => api.runs(),
    refetchInterval: (q) => (q.state.data?.some((r) => r.status === "queued" || r.status === "running") ? 1500 : 10000),
  });

  const versions = useMemo(() => (datasets.data ?? []).flatMap((d) =>
    d.versions.filter((v) => v.status === "validated").map((v) => ({ ...v, dataset: d.name }))), [datasets.data]);
  const [versionId, setVersionId] = useState("");
  const [algoKey, setAlgoKey] = useState("gradient_boosting");
  const [hp, setHp] = useState<Record<string, unknown>>({});
  const [seed, setSeed] = useState(42);
  const [cv, setCv] = useState(5);
  const [selected, setSelected] = useState<string[]>([]);

  useEffect(() => {
    if (versionId || !versions.length) return;
    const wanted = search.get("version");
    setVersionId(versions.find((v) => v.id === wanted)?.id ?? versions.at(-1)!.id);
  }, [versions, versionId, search]);

  const algo = meta.data?.algorithms.find((a) => a.key === algoKey);
  useEffect(() => { if (algo) setHp(Object.fromEntries(algo.params.map((p) => [p.name, p.default]))); }, [algo]);

  const launch = useMutation({
    mutationFn: () => api.launchRun({ dataset_version_id: versionId, algorithm: algoKey, hyperparameters: hp, seed, cv_folds: cv }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["runs"] }),
  });

  const toggle = (id: string) => setSelected((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s, id].slice(-8)));

  return (
    <>
      <PageHeader title="Experiment lab"
        description="Launch training runs against a validated dataset version. Each run is executed by the job worker, evaluated on that version's fixed holdout, and logged to MLflow with its model, parameters, dataset hash and source commit." />
      <div className="grid gap-6 xl:grid-cols-3">
        <Panel title="New training run" className="xl:col-span-1">
          {!versions.length && datasets.isSuccess ? (
            <Empty title="No validated dataset versions">Register one in the <Link className="text-ember underline" href="/datasets">Dataset studio</Link> first.</Empty>
          ) : (
            <form className="space-y-4" onSubmit={(e) => { e.preventDefault(); launch.mutate(); }}>
              <Field label="Dataset version">
                <select className={inputCls} value={versionId} onChange={(e) => setVersionId(e.target.value)}>
                  {versions.map((v) => <option key={v.id} value={v.id}>{v.dataset} · v{v.version} ({v.n_rows} rows)</option>)}
                </select>
              </Field>
              <Field label="Algorithm" hint={algo?.description}>
                <div className="grid grid-cols-1 gap-1.5">
                  {meta.data?.algorithms.map((a) => (
                    <button type="button" key={a.key} onClick={() => setAlgoKey(a.key)} aria-pressed={a.key === algoKey}
                      className={cn("rounded-lg border px-3 py-2 text-left text-sm transition-colors",
                        a.key === algoKey ? "border-ember/60 bg-ember/10 text-white" : "border-white/10 text-white/65 hover:border-white/25")}>
                      {a.label}
                    </button>
                  ))}
                </div>
              </Field>
              {algo && <HyperParams algo={algo} value={hp} onChange={setHp} />}
              <div className="grid grid-cols-2 gap-3">
                <Field label="Seed"><input type="number" min={0} className={inputCls} value={seed} onChange={(e) => setSeed(Number(e.target.value))} /></Field>
                <Field label="CV folds" hint="0 skips cross-validation">
                  <input type="number" min={0} max={10} className={inputCls} value={cv} onChange={(e) => setCv(Number(e.target.value))} />
                </Field>
              </div>
              <ErrorNote error={launch.error} />
              <Button type="submit" className="w-full" loading={launch.isPending} disabled={!versionId}>Launch run</Button>
            </form>
          )}
        </Panel>

        <Panel className="xl:col-span-2" title="Runs"
          subtitle="Select two or more runs to compare them"
          action={<Button variant="outline" disabled={selected.length < 2}
            onClick={() => router.push(`/experiments/compare?ids=${selected.join(",")}`)}>
            Compare {selected.length >= 2 ? `${selected.length} runs` : ""}</Button>}>
          <ErrorNote error={runs.error} />
          {runs.data?.length ? (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[720px] text-sm">
                <thead className="text-left text-xs text-white/45"><tr>
                  <th className="w-8 pb-2" /><th className="pb-2 font-normal">Run</th><th className="pb-2 font-normal">Algorithm</th>
                  <th className="pb-2 font-normal">Data</th><th className="pb-2 font-normal">Status</th>
                  <th className="pb-2 text-right font-normal">F1</th><th className="pb-2 text-right font-normal">ROC-AUC</th>
                  <th className="pb-2 text-right font-normal">CV F1</th><th className="pb-2 text-right font-normal">Started</th></tr></thead>
                <tbody className="divide-y divide-white/5">
                  {runs.data.map((r) => (
                    <tr key={r.id} className={cn(selected.includes(r.id) && "bg-ember/[0.04]")}>
                      <td className="py-2.5"><input type="checkbox" aria-label={`Select run ${r.id.slice(0, 8)}`} className="accent-[#f5b04c]"
                        checked={selected.includes(r.id)} onChange={() => toggle(r.id)} disabled={r.status !== "succeeded"} /></td>
                      <td><Link className="font-mono text-xs text-ember hover:underline" href={`/experiments/${r.id}`}>{r.id.slice(0, 8)}</Link></td>
                      <td>{ALGO_LABEL[r.algorithm] ?? r.algorithm}</td>
                      <td className="text-white/60">{r.dataset_name} v{r.dataset_version_number}</td>
                      <td><Status value={r.status} /></td>
                      <td className="text-right tabular-nums">{num(r.metrics?.f1)}</td>
                      <td className="text-right tabular-nums">{num(r.metrics?.roc_auc ?? r.metrics?.roc_auc_ovr)}</td>
                      <td className="text-right tabular-nums text-white/60">{r.metrics?.cv_f1_macro_mean !== undefined ? `${num(r.metrics.cv_f1_macro_mean, 3)} ± ${num(r.metrics.cv_f1_macro_std, 3)}` : "—"}</td>
                      <td className="text-right text-xs text-white/40">{ago(r.created_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : runs.isSuccess ? <Empty title="No runs yet">Launch your first run with the form.</Empty> : null}
        </Panel>
      </div>
    </>
  );
}

function HyperParams({ algo, value, onChange }: { algo: Algorithm; value: Record<string, unknown>; onChange: (v: Record<string, unknown>) => void }) {
  const set = (p: Param, raw: string) => {
    let v: unknown = raw;
    if (p.type === "choice") v = raw === "__none__" ? null : raw;
    else if (raw === "" && p.nullable) v = null;
    else v = p.type === "int" ? parseInt(raw, 10) : parseFloat(raw);
    onChange({ ...value, [p.name]: v });
  };
  return (
    <fieldset className="grid grid-cols-2 gap-3 rounded-xl border border-white/5 p-3">
      <legend className="px-1 text-xs text-white/45">Hyperparameters</legend>
      {algo.params.map((p) => (
        <Field key={p.name} label={p.name} hint={p.type === "choice" ? undefined : `${p.min ?? ""}–${p.max ?? ""}${p.nullable ? ", empty = none" : ""}`}>
          {p.type === "choice" ? (
            <select className={inputCls} value={value[p.name] === null ? "__none__" : String(value[p.name] ?? "")} onChange={(e) => set(p, e.target.value)}>
              {p.choices!.map((c) => <option key={String(c)} value={c === null ? "__none__" : String(c)}>{c === null ? "none" : String(c)}</option>)}
            </select>
          ) : (
            <input type="number" className={inputCls} step={p.type === "int" ? 1 : "any"} min={p.min ?? undefined} max={p.max ?? undefined}
              value={value[p.name] === null || value[p.name] === undefined ? "" : String(value[p.name])}
              placeholder={p.nullable ? "none" : undefined}
              onChange={(e) => set(p, e.target.value)} />
          )}
        </Field>
      ))}
    </fieldset>
  );
}
