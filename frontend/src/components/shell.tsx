"use client";

import { cn } from "@/lib/format";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import {
  Activity, Boxes, Database, FlaskConical, Gauge, GitBranch, LayoutDashboard, Rocket,
  ScanSearch, Wallet,
} from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";

// Modules that are not built yet are listed with the phase that delivers them,
// rather than linking to placeholder screens.
const NAV = [
  { href: "/", label: "Operations overview", icon: LayoutDashboard },
  { href: "/datasets", label: "Dataset studio", icon: Database },
  { href: "/experiments", label: "Experiment lab", icon: FlaskConical },
  { label: "Model registry", icon: Boxes, phase: 3 },
  { label: "Pipeline studio", icon: GitBranch, phase: 3 },
  { label: "Deployment center", icon: Rocket, phase: 3 },
  { label: "Evaluation workbench", icon: ScanSearch, phase: 4 },
  { label: "Agent observability", icon: Activity, phase: 5 },
  { label: "Model monitoring", icon: Gauge, phase: 6 },
  { label: "Cost & performance", icon: Wallet, phase: 4 },
];

export function Shell({ children }: { children: ReactNode }) {
  const path = usePathname();
  const meta = useQuery({ queryKey: ["meta"], queryFn: api.meta, staleTime: 60_000 });
  return (
    <div className="relative z-10 flex min-h-dvh">
      <aside className="glass-strong sticky top-0 hidden h-dvh w-64 shrink-0 flex-col border-y-0 border-l-0 px-3 py-5 lg:flex">
        <Link href="/" className="mb-7 flex items-center gap-2.5 px-2.5">
          <Mark />
          <span className="text-[17px] font-semibold tracking-tight text-white">EvalForge</span>
        </Link>
        <nav aria-label="Modules" className="flex flex-1 flex-col gap-0.5">
          {NAV.map((n) => {
            const Icon = n.icon;
            if (!n.href) {
              return (
                <div key={n.label} className="flex items-center gap-3 rounded-lg px-2.5 py-2 text-sm text-white/30"
                     title={`Arrives in phase ${n.phase}`}>
                  <Icon className="size-4" aria-hidden />
                  <span className="flex-1">{n.label}</span>
                  <span className="text-[11px] text-white/25">P{n.phase}</span>
                </div>
              );
            }
            const active = n.href === "/" ? path === "/" : path.startsWith(n.href);
            return (
              <Link key={n.href} href={n.href}
                className={cn("flex items-center gap-3 rounded-lg px-2.5 py-2 text-sm transition-colors",
                  active ? "bg-white/[0.07] text-white" : "text-white/65 hover:bg-white/[0.04] hover:text-white")}>
                <Icon className={cn("size-4", active && "text-ember")} aria-hidden />
                {n.label}
              </Link>
            );
          })}
        </nav>
        <div className="mt-4 space-y-1 border-t border-white/5 px-2.5 pt-4 text-xs text-white/40">
          {meta.data ? (
            <>
              <div>API v{meta.data.version} · {meta.data.database}</div>
              <div>Jobs: {meta.data.executor} executor</div>
              <div>MLflow: {meta.data.mlflow_tracking_uri === "embedded" ? "embedded store" : meta.data.mlflow_tracking_uri}</div>
            </>
          ) : meta.isError ? <div className="text-fail">API unreachable</div> : <div>Connecting…</div>}
        </div>
      </aside>
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="glass-strong sticky top-0 z-20 flex items-center gap-4 border-x-0 border-t-0 px-4 py-3 lg:hidden">
          <Mark />
          <nav className="flex gap-4 overflow-x-auto text-sm">
            {NAV.filter((n) => n.href).map((n) => (
              <Link key={n.href} href={n.href!} className="whitespace-nowrap text-white/75">{n.label}</Link>
            ))}
          </nav>
        </header>
        <main className="mx-auto w-full max-w-[1400px] flex-1 px-4 py-8 sm:px-8">{children}</main>
      </div>
    </div>
  );
}

function Mark() {
  // An anvil-like bracket: data enters left, a lit core in the middle.
  return (
    <svg viewBox="0 0 28 28" className="size-7" aria-hidden>
      <rect x="1" y="1" width="26" height="26" rx="8" fill="rgb(255 255 255 / 0.06)" stroke="rgb(255 255 255 / 0.12)" />
      <path d="M7 9h14M7 19h14" stroke="rgb(255 255 255 / 0.55)" strokeWidth="1.6" strokeLinecap="round" />
      <circle cx="14" cy="14" r="3.2" fill="var(--color-ember)" />
    </svg>
  );
}

export function PageHeader({ title, description, action }: { title: string; description?: ReactNode; action?: ReactNode }) {
  return (
    <div className="mb-7 flex flex-wrap items-end justify-between gap-4">
      <div>
        <h1 className="text-[28px] font-semibold tracking-tight text-white">{title}</h1>
        {description && <p className="mt-1.5 max-w-2xl text-[15px] leading-relaxed text-white/55">{description}</p>}
      </div>
      {action}
    </div>
  );
}
