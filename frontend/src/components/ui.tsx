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
      {value}
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
