"use client";

import { PageHeader } from "@/components/shell";
import { Button, Empty, ErrorNote, JsonBlock, Mono, Notice, Panel, Pill, Status } from "@/components/ui";
import { api, type J } from "@/lib/api";
import { ALGO_LABEL, ago, cn, num } from "@/lib/format";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CircleCheck, CircleX, Crown, Info } from "lucide-react";
import Link from "next/link";
import { useState } from "react";

export default function RegistryPage() {
  const qc = useQueryClient();
  const models = useQuery({ queryKey: ["models"], queryFn: api.models });
  const [sel, setSel] = useState<string | null>(null);
  const [traffic, setTraffic] = useState(100);
  const [card, setCard] = useState<J | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const all = (models.data ?? []).flatMap((m) => m.versions as J[]);
  const v = all.find((x) => x.id === sel) ?? all[0];

  const act = useMutation({
    mutationFn: async (kind: string) => {
      if (kind === "gate") return api.gate(v.id);
      if (kind === "promote") return api.requestPromotion(v.id);
      if (kind === "staging") return api.deploy(v.id, "staging", 100);
      if (kind === "production") return api.deploy(v.id, "production", traffic);
      if (kind === "card") return setCard(await api.card(v.id));
    },
    onSuccess: (r: J | void, kind) => {
      qc.invalidateQueries();
      setNote(kind === "promote" ? "Promotion requested. An approver must accept it in Approvals."
        : kind === "production" ? "Production deployment requested. It runs once an approver accepts it."
        : kind === "staging" ? "Deployed to staging after a passing health check." : null);
    },
  });

  return (
    <>
      <PageHeader title="Model registry"
        description="Versions are registered in MLflow and promoted with the champion alias. Each version carries a quality gate that is re-checked against the current champion, on the same holdout, when an approver decides." />
      <ErrorNote error={models.error ?? act.error} />
      {note && <div className="mb-4"><Notice>{note}</Notice></div>}
      {!models.data?.length && models.isSuccess ? (
        <Empty title="No registered models">Open a finished run in the <Link href="/experiments" className="text-ember underline">Experiment lab</Link> and choose Register model, or run a pipeline.</Empty>
      ) : (
        <div className="grid gap-6 xl:grid-cols-5">
          <Panel className="xl:col-span-2" title="Versions">
            {models.data?.map((m) => (
              <div key={m.name} className="mb-4">
                <div className="mb-1 text-sm font-medium text-white">{m.name} {m.champion && <span className="text-xs text-white/45">· champion v{m.champion}</span>}</div>
                <ul className="divide-y divide-white/5">
                  {(m.versions as J[]).map((x) => (
                    <li key={x.id}>
                      <button onClick={() => setSel(x.id)} className={cn("-mx-2 flex w-full items-center gap-3 rounded-lg px-2 py-2 text-left text-sm",
                        v?.id === x.id ? "bg-white/[0.06]" : "hover:bg-white/[0.03]")}>
                        <span className="w-8 font-medium text-white">v{x.version}</span>
                        <span className="flex-1 text-white/70">{ALGO_LABEL[x.algorithm] ?? x.algorithm} · data v{x.dataset_version}</span>
                        <span className="tabular-nums text-white/60">F1 {num(x.metrics?.f1, 3)}</span>
                        {x.is_champion ? <Pill tone="ember"><Crown className="mr-1 size-3" />champion</Pill>
                          : <Status value={x.status} />}
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </Panel>
          {v && (
            <Panel className="xl:col-span-3" title={`${v.name} v${v.version}`}
              subtitle={<>Run <Link className="text-ember hover:underline" href={`/experiments/${v.run_id}`}>{v.run_id.slice(0, 8)}</Link> · registered by {v.created_by} {ago(v.created_at)}</>}
              action={<Button variant="ghost" onClick={() => act.mutate("card")}>Model card</Button>}>
              <div className="mb-5 grid grid-cols-2 gap-3 sm:grid-cols-4">
                {["f1", "precision", "recall", "roc_auc"].map((k) => (
                  <div key={k} className="rounded-xl bg-white/[0.03] px-3 py-2"><div className="text-xs text-white/45">{k}</div><div className="text-lg font-semibold tabular-nums text-white">{num(v.metrics?.[k], 4)}</div></div>
                ))}
              </div>
              <h3 className="mb-2 text-sm font-medium text-white">Quality gate {v.gate && (v.gate.passed ? <Pill tone="pass">passed</Pill> : <Pill tone="fail">blocked</Pill>)}</h3>
              <ul className="mb-4 space-y-2">
                {(v.gate?.checks ?? []).map((c: J) => {
                  const Icon = c.passed ? CircleCheck : c.blocking ? CircleX : Info;
                  return (
                    <li key={c.id} className="flex gap-2 text-sm">
                      <Icon className={cn("mt-0.5 size-4 shrink-0", c.passed ? "text-pass" : c.blocking ? "text-fail" : "text-warn")} />
                      <span><span className="text-white">{c.id.replaceAll("_", " ")}</span> <span className="text-white/50">— {c.detail}</span>{!c.blocking && <span className="text-xs text-white/35"> (advisory)</span>}</span>
                    </li>
                  );
                })}
              </ul>
              {v.gate?.champion_comparison && (
                <p className="mb-4 text-xs text-white/50">Champion v{v.gate.champion_comparison.champion_version} re-scored on dataset v{v.gate.champion_comparison.evaluated_on_dataset_version}: F1 {num(v.gate.champion_comparison.champion_f1)} vs candidate {num(v.gate.champion_comparison.candidate_f1)} (Δ {num(v.gate.champion_comparison.delta_f1)}).</p>
              )}
              <div className="flex flex-wrap items-center gap-2 border-t border-white/5 pt-4">
                <Button variant="outline" loading={act.isPending && act.variables === "gate"} onClick={() => act.mutate("gate")}>Re-run gate</Button>
                <Button disabled={v.is_champion || !v.gate?.passed} loading={act.isPending && act.variables === "promote"} onClick={() => act.mutate("promote")}>Request promotion</Button>
                <Button variant="outline" loading={act.isPending && act.variables === "staging"} onClick={() => act.mutate("staging")}>Deploy to staging</Button>
                <span className="ml-auto flex items-center gap-2 text-sm text-white/60">
                  Production traffic
                  <select className="rounded-md border border-white/10 bg-ink-950 px-2 py-1.5 text-sm" value={traffic} onChange={(e) => setTraffic(Number(e.target.value))}>
                    {[100, 50, 25, 10].map((t) => <option key={t} value={t}>{t === 100 ? "100% (stable)" : `${t}% canary`}</option>)}
                  </select>
                  <Button variant="outline" disabled={v.status !== "approved"} loading={act.isPending && act.variables === "production"} onClick={() => act.mutate("production")}>Request production deploy</Button>
                </span>
              </div>
              <div className="mt-3 text-xs text-white/40">MLflow source <Mono>{v.source_uri}</Mono></div>
            </Panel>
          )}
        </div>
      )}
      {card && (
        <div role="dialog" aria-modal className="fixed inset-0 z-50 grid place-items-center bg-ink-950/70 p-4" onClick={() => setCard(null)}>
          <div className="glass-strong max-h-[85vh] w-full max-w-3xl overflow-auto rounded-2xl p-6" onClick={(e) => e.stopPropagation()}>
            <div className="mb-3 flex items-center justify-between"><h2 className="text-lg font-semibold text-white">Model card · {card.model}</h2><Button variant="ghost" onClick={() => setCard(null)}>Close</Button></div>
            <p className="mb-3 text-sm text-white/70">{card.intended_use}</p>
            <h3 className="mb-1 text-sm font-medium text-white">Limitations</h3>
            <ul className="mb-4 list-disc pl-5 text-sm text-white/60">{(card.limitations as string[]).map((l) => <li key={l}>{l}</li>)}</ul>
            <JsonBlock value={{ training_data: card.training_data, evaluation: card.evaluation, lineage: card.lineage }} />
          </div>
        </div>
      )}
    </>
  );
}
