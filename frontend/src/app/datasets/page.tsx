"use client";

import { PageHeader } from "@/components/shell";
import { Button, Empty, ErrorNote, Field, inputCls, Panel, Status } from "@/components/ui";
import { api } from "@/lib/api";
import { ago, bytes } from "@/lib/format";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";

export default function DatasetsPage() {
  const qc = useQueryClient();
  const router = useRouter();
  const list = useQuery({ queryKey: ["datasets"], queryFn: api.datasets });
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState("");
  const [target, setTarget] = useState("");
  const [description, setDescription] = useState("");
  const [columns, setColumns] = useState<string[]>([]);

  const create = useMutation({
    mutationFn: () => {
      const f = new FormData();
      f.append("file", file!); f.append("name", name); f.append("target_column", target);
      f.append("description", description);
      return api.createDataset(f);
    },
    onSuccess: (ds) => { qc.invalidateQueries(); router.push(`/datasets/${ds.id}`); },
  });
  const demo = useMutation({
    mutationFn: api.loadDemo,
    onSuccess: (r) => { qc.invalidateQueries(); router.push(`/datasets/${r.dataset_id}`); },
  });

  async function pick(f: File | null) {
    setFile(f); setColumns([]);
    if (!f) return;
    if (!name) setName(f.name.replace(/\.(csv|jsonl?|ndjson)$/i, "").replace(/[^A-Za-z0-9_.\- ]/g, "-").slice(0, 60));
    // Read only the header locally to offer the target column; parsing happens server-side.
    const head = await f.slice(0, 64 * 1024).text();
    if (/\.csv$/i.test(f.name)) setColumns(head.split(/\r?\n/)[0].split(",").map((c) => c.trim().replace(/^"|"$/g, "")));
    else { const m = head.match(/\{[^{}]*\}/); if (m) try { setColumns(Object.keys(JSON.parse(m[0]))); } catch { /* ignore */ } }
  }

  return (
    <>
      <PageHeader title="Dataset studio"
        description="Each upload becomes an immutable, content-hashed version with a profile, quality and leakage checks, and a fixed stratified holdout split shared by every run trained on it." />
      <div className="grid gap-6 xl:grid-cols-5">
        <Panel className="xl:col-span-3" title="Datasets">
          <ErrorNote error={list.error} />
          {list.data?.length ? (
            <ul className="divide-y divide-white/5">
              {list.data.map((d) => {
                const v = d.versions[d.versions.length - 1];
                return (
                  <li key={d.id}>
                    <Link href={`/datasets/${d.id}`} className="-mx-2 flex flex-wrap items-center gap-x-6 gap-y-1 rounded-lg px-2 py-3 hover:bg-white/[0.03]">
                      <div className="min-w-0 flex-1">
                        <div className="font-medium text-white">{d.name}</div>
                        <div className="truncate text-xs text-white/45">target “{d.target_column}” · {d.versions.length} version{d.versions.length > 1 ? "s" : ""} · {ago(d.created_at)}</div>
                      </div>
                      {v && <div className="text-right text-xs tabular-nums text-white/55">{v.n_rows.toLocaleString()} rows × {v.n_columns} cols<br />{bytes(v.size_bytes)}</div>}
                      {v && <Status value={v.status} />}
                    </Link>
                  </li>
                );
              })}
            </ul>
          ) : list.isSuccess ? (
            <Empty title="No datasets yet">
              Upload a CSV or JSON file, or start with a public dataset.
              <div className="mt-4"><Button variant="outline" onClick={() => demo.mutate()} loading={demo.isPending}>Load the breast-cancer demo dataset</Button></div>
            </Empty>
          ) : null}
        </Panel>

        <Panel className="xl:col-span-2" title="Register a dataset" subtitle="CSV, JSON array or JSON Lines. Classification targets with 2–50 classes.">
          <form className="space-y-4" onSubmit={(e) => { e.preventDefault(); create.mutate(); }}>
            <Field label="File">
              <input type="file" accept=".csv,.json,.jsonl,.ndjson" required className={inputCls + " file:mr-3 file:rounded-md file:border-0 file:bg-white/10 file:px-2.5 file:py-1 file:text-white"}
                onChange={(e) => pick(e.target.files?.[0] ?? null)} />
            </Field>
            <Field label="Name" hint="Letters, digits, dash, dot, underscore. Becomes the MLflow experiment name.">
              <input className={inputCls} value={name} onChange={(e) => setName(e.target.value)} required pattern="[A-Za-z0-9][A-Za-z0-9_.\- ]{1,63}" />
            </Field>
            <Field label="Target column">
              {columns.length ? (
                <select className={inputCls} value={target} onChange={(e) => setTarget(e.target.value)} required>
                  <option value="">Choose the label column…</option>
                  {columns.map((c) => <option key={c}>{c}</option>)}
                </select>
              ) : <input className={inputCls} value={target} onChange={(e) => setTarget(e.target.value)} required />}
            </Field>
            <Field label="Description">
              <textarea className={inputCls} rows={2} value={description} onChange={(e) => setDescription(e.target.value)} />
            </Field>
            <ErrorNote error={create.error ?? demo.error} />
            <Button type="submit" loading={create.isPending} disabled={!file}>Upload and validate</Button>
          </form>
        </Panel>
      </div>
    </>
  );
}
