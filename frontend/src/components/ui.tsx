"use client";

import { cn } from "@/lib/format";
import { Loader2 } from "lucide-react";
import type { ButtonHTMLAttributes, ReactNode } from "react";

export function Panel({ className, children, title, action, subtitle }: {
  className?: string; children: ReactNode; title?: ReactNode; subtitle?: ReactNode; action?: ReactNode;
}) {
  return (
    <section className={cn("glass rounded-2xl p-5", className)}>
      {(title || action) && (
        <header className="mb-4 flex items-start justify-between gap-4">
          <div>
            {title && <h2 className="text-[15px] font-semibold text-white">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-[13px] text-white/50">{subtitle}</p>}
          </div>
          {action}
        </header>
      )}
      {children}
    </section>
  );
}

export function Button({ variant = "primary", loading, className, children, ...rest }:
  ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "ghost" | "outline"; loading?: boolean }) {
  return (
    <button
      {...rest}
      disabled={rest.disabled || loading}
      className={cn(
        "inline-flex items-center justify-center gap-2 rounded-lg px-3.5 py-2 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50",
        variant === "primary" && "bg-ember text-ink-950 hover:bg-ember-soft",
        variant === "outline" && "border border-white/15 text-white hover:bg-white/5",
        variant === "ghost" && "text-white/70 hover:bg-white/5 hover:text-white",
        className,
      )}
    >
      {loading && <Loader2 className="size-4 animate-spin" aria-hidden />}
      {children}
    </button>
  );
}

const STATUS: Record<string, string> = {
  executed: "text-pass bg-pass/10 ring-pass/25",
  active: "text-pass bg-pass/10 ring-pass/25",
  approved: "text-pass bg-pass/10 ring-pass/25",
  completed: "text-pass bg-pass/10 ring-pass/25",
  resolved: "text-pass bg-pass/10 ring-pass/25",
  released: "text-pass bg-pass/10 ring-pass/25",
  pending: "text-warn bg-warn/10 ring-warn/25",
  awaiting_approval: "text-warn bg-warn/10 ring-warn/25",
  waiting_approval: "text-warn bg-warn/10 ring-warn/25",
  open: "text-fail bg-fail/10 ring-fail/25",
  investigating: "text-ember bg-ember/10 ring-ember/25",
  mitigating: "text-ember bg-ember/10 ring-ember/25",
  budget_exceeded: "text-warn bg-warn/10 ring-warn/25",
  rolled_back: "text-warn bg-warn/10 ring-warn/25",
  skipped: "text-white/40 bg-white/5 ring-white/10",
  superseded: "text-white/40 bg-white/5 ring-white/10",
  cancelled: "text-white/40 bg-white/5 ring-white/10",
  draft: "text-steel bg-steel/10 ring-steel/25",
  registered: "text-steel bg-steel/10 ring-steel/25",
  succeeded: "text-pass bg-pass/10 ring-pass/25",
  validated: "text-pass bg-pass/10 ring-pass/25",
  running: "text-ember bg-ember/10 ring-ember/25",
  queued: "text-steel bg-steel/10 ring-steel/25",
  failed: "text-fail bg-fail/10 ring-fail/25",
  rejected: "text-fail bg-fail/10 ring-fail/25",
};

export function Status({ value }: { value: string }) {
  return (
    <span className={cn("inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-xs font-medium ring-1",
      STATUS[value] ?? "text-white/60 bg-white/5 ring-white/10")}>
      {value === "running" && <span className="size-1.5 animate-pulse rounded-full bg-current" />}
      {value.replaceAll("_", " ")}
    </span>
  );
}

export function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1.5 block text-[13px] font-medium text-white/80">{label}</span>
      {children}
      {hint && <span className="mt-1 block text-xs text-white/40">{hint}</span>}
    </label>
  );
}

export const inputCls =
  "w-full rounded-lg border border-white/10 bg-ink-950/60 px-3 py-2 text-sm text-white placeholder:text-white/30 focus:border-ember/60 focus:outline-none";

export function Stat({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div>
      <div className="text-[13px] text-white/50">{label}</div>
      <div className="mt-1 text-2xl font-semibold tabular-nums text-white">{value}</div>
      {sub && <div className="mt-0.5 text-xs text-white/40">{sub}</div>}
    </div>
  );
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="rounded-xl border border-dashed border-white/10 px-6 py-10 text-center">
      <p className="text-sm font-medium text-white/80">{title}</p>
      {children && <div className="mt-2 text-sm text-white/50">{children}</div>}
    </div>
  );
}

export function ErrorNote({ error }: { error: unknown }) {
  if (!error) return null;
  return (
    <p role="alert" className="rounded-lg border border-fail/30 bg-fail/10 px-3 py-2 text-sm text-fail">
      {error instanceof Error ? error.message : String(error)}
    </p>
  );
}

export function Mono({ children, className }: { children: ReactNode; className?: string }) {
  return <code className={cn("font-mono text-[12px] text-white/60", className)}>{children}</code>;
}

export function JsonBlock({ value, className }: { value: unknown; className?: string }) {
  return (
    <pre className={cn("max-h-80 overflow-auto rounded-lg bg-ink-950/70 p-3 font-mono text-[11px] leading-relaxed text-white/70", className)}>
      {JSON.stringify(value, null, 2)}
    </pre>
  );
}

export function Pill({ children, tone = "neutral" }: { children: ReactNode; tone?: "neutral" | "pass" | "fail" | "warn" | "ember" | "steel" }) {
  const t = {
    neutral: "bg-white/5 text-white/65 ring-white/10",
    pass: "bg-pass/10 text-pass ring-pass/25",
    fail: "bg-fail/10 text-fail ring-fail/25",
    warn: "bg-warn/10 text-warn ring-warn/25",
    ember: "bg-ember/10 text-ember ring-ember/25",
    steel: "bg-steel/10 text-steel ring-steel/25",
  }[tone];
  return <span className={cn("inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ring-1", t)}>{children}</span>;
}

export function Th({ children, right }: { children?: ReactNode; right?: boolean }) {
  return <th className={cn("px-2 pb-2 text-xs font-normal text-white/45 first:pl-0", right ? "text-right" : "text-left")}>{children}</th>;
}

export function Notice({ children, tone = "pass" }: { children: ReactNode; tone?: "pass" | "warn" }) {
  return (
    <p role="status" className={cn("rounded-lg border px-3 py-2 text-sm",
      tone === "pass" ? "border-pass/30 bg-pass/10 text-pass" : "border-warn/30 bg-warn/10 text-warn")}>{children}</p>
  );
}
