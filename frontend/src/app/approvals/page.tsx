"use client";

import { PageHeader } from "@/components/shell";
import { Button, Empty, ErrorNote, JsonBlock, Panel, Status } from "@/components/ui";
import { api, getUser, type J } from "@/lib/api";
import { ago } from "@/lib/format";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

const KIND: Record<string, string> = {
  model_promotion: "Model promotion", deployment: "Production deployment", rollback: "Rollback",
  canary_promotion: "Canary promotion", retraining: "Retraining", evaluation: "Evaluation run",
  prompt_release: "Prompt release",
};

export default function ApprovalsPage() {
  const qc = useQueryClient();
  const list = useQuery({ queryKey: ["approvals"], queryFn: () => api.approvals(), refetchInterval: 5000 });
  const me = getUser();
  const [reason, setReason] = useState<Record<string, string>>({});
  const decide = useMutation({
    mutationFn: ({ id, ok }: { id: string; ok: boolean }) => api.decide(id, ok, reason[id] || (ok ? "Reviewed and approved" : "Rejected")),
    onSuccess: () => qc.invalidateQueries(),
  });
  const pending = (list.data ?? []).filter((a) => a.status === "pending");
  const done = (list.data ?? []).filter((a) => a.status !== "pending");
  const canDecide = me && ["approver", "admin"].includes(me.role);

  return (
    <>
      <PageHeader title="Approvals"
        description="Human checkpoints for production changes and costly work. An approver who did not file the request must accept it; accepting runs the action and records its result here." />
      <ErrorNote error={list.error ?? decide.error} />
      <Panel title="Waiting for a decision" subtitle={canDecide ? "You can approve requests filed by others." : `Signed in as ${me?.role}; only approvers can decide.`}>
        {!pending.length ? <Empty title="Nothing waiting" /> : (
          <ul className="space-y-4">
            {pending.map((a: J) => (
              <li key={a.id} className="rounded-xl border border-white/5 bg-white/[0.02] p-4">
                <div className="flex flex-wrap items-baseline justify-between gap-2">
                  <div><span className="text-sm font-medium text-white">{KIND[a.kind] ?? a.kind}</span> <span className="text-sm text-white/70">— {a.summary}</span></div>
                  <span className="text-xs text-white/45">requested by {a.requested_by} {ago(a.created_at)}</span>
                </div>
                {a.payload?.plan && <p className="mt-2 text-xs text-white/55">{(a.payload.plan.reasons as string[]).join(" ")}</p>}
                {canDecide && a.requested_by !== me?.username && (
                  <div className="mt-3 flex flex-wrap gap-2">
                    <input aria-label="Decision reason" placeholder="Reason (recorded in the audit log)" className="min-w-64 flex-1 rounded-lg border border-white/10 bg-ink-950/60 px-3 py-2 text-sm"
                      value={reason[a.id] ?? ""} onChange={(e) => setReason({ ...reason, [a.id]: e.target.value })} />
                    <Button loading={decide.isPending && decide.variables?.id === a.id && decide.variables.ok} onClick={() => decide.mutate({ id: a.id, ok: true })}>Approve</Button>
                    <Button variant="outline" onClick={() => decide.mutate({ id: a.id, ok: false })}>Reject</Button>
                  </div>
                )}
                {a.requested_by === me?.username && <p className="mt-2 text-xs text-white/40">You filed this request, so another approver must decide it.</p>}
              </li>
            ))}
          </ul>
        )}
      </Panel>
      <Panel className="mt-6" title="History">
        <ul className="divide-y divide-white/5">
          {done.map((a: J) => (
            <li key={a.id} className="py-3">
              <div className="flex flex-wrap items-center gap-3 text-sm">
                <Status value={a.status} />
                <span className="text-white">{KIND[a.kind] ?? a.kind}</span>
                <span className="text-white/60">{a.summary}</span>
                <span className="ml-auto text-xs text-white/45">{a.requested_by} → {a.decided_by} · {ago(a.decided_at ?? a.created_at)}</span>
              </div>
              {a.decision_reason && <p className="mt-1 text-xs text-white/50">“{a.decision_reason}”</p>}
              {a.result && <details className="mt-1"><summary className="cursor-pointer text-xs text-white/40">Result</summary><JsonBlock value={a.result} /></details>}
            </li>
          ))}
        </ul>
      </Panel>
    </>
  );
}
