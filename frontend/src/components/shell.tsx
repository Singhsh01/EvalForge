"use client";

import { cn } from "@/lib/format";
import { useQuery } from "@tanstack/react-query";
import { api, getToken, getUser, setSession } from "@/lib/api";
import {
  Activity, Bot, Boxes, Database, FileText, FlaskConical, Gauge, GitBranch, HeartPulse,
  LayoutDashboard, Rocket, ScanSearch, ShieldCheck, Siren, Wallet,
} from "lucide-react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState, type ReactNode } from "react";

const NAV: { group: string; items: { href: string; label: string; icon: typeof LayoutDashboard }[] }[] = [
  { group: "Operate", items: [
    { href: "/", label: "Operations overview", icon: LayoutDashboard },
    { href: "/copilot", label: "MLOps Copilot", icon: Bot },
    { href: "/incidents", label: "Incidents", icon: Siren },
    { href: "/approvals", label: "Approvals", icon: ShieldCheck },
  ] },
  { group: "Build", items: [
    { href: "/datasets", label: "Dataset studio", icon: Database },
    { href: "/experiments", label: "Experiment lab", icon: FlaskConical },
    { href: "/pipelines", label: "Pipeline studio", icon: GitBranch },
    { href: "/registry", label: "Model registry", icon: Boxes },
    { href: "/deployments", label: "Deployment center", icon: Rocket },
  ] },
  { group: "Evaluate", items: [
    { href: "/evaluations", label: "Evaluation workbench", icon: ScanSearch },
    { href: "/prompts", label: "Prompt registry", icon: FileText },
  ] },
  { group: "Observe", items: [
    { href: "/traces", label: "Trace explorer", icon: Activity },
    { href: "/monitoring", label: "Drift monitoring", icon: Gauge },
    { href: "/analytics", label: "Cost & performance", icon: Wallet },
    { href: "/system", label: "System health & audit", icon: HeartPulse },
  ] },
];

export function Shell({ children }: { children: ReactNode }) {
  const path = usePathname();
  const router = useRouter();
  const [user, setUser] = useState<{ username: string; role: string } | null>(null);
  const [ready, setReady] = useState(false);
  useEffect(() => {
    const u = getUser();
    if (!getToken() && path !== "/login") router.replace("/login");
    setUser(u);
    setReady(true);
  }, [path, router]);
  const meta = useQuery({ queryKey: ["meta"], queryFn: api.meta, staleTime: 60_000 });

  if (path === "/login") return <div className="relative z-10">{children}</div>;
  if (!ready || !user) return null;

  const logout = async () => {
    try { await api.logout(); } catch { /* token may already be invalid */ }
    setSession(null);
    router.replace("/login");
  };

  return (
    <div className="relative z-10 flex min-h-dvh">
      <aside className="glass-strong sticky top-0 hidden h-dvh w-64 shrink-0 flex-col overflow-y-auto border-y-0 border-l-0 px-3 py-5 lg:flex">
        <Link href="/" className="mb-6 flex items-center gap-2.5 px-2.5">
          <Mark />
          <span className="text-[17px] font-semibold tracking-tight text-white">EvalForge</span>
        </Link>
        <nav aria-label="Modules" className="flex flex-1 flex-col gap-4">
          {NAV.map((g) => (
            <div key={g.group}>
              <div className="mb-1 px-2.5 text-[11px] font-medium text-white/35">{g.group}</div>
              <div className="flex flex-col gap-0.5">
                {g.items.map((n) => {
                  const Icon = n.icon;
                  const active = n.href === "/" ? path === "/" : path.startsWith(n.href);
                  return (
                    <Link key={n.href} href={n.href}
                      className={cn("flex items-center gap-3 rounded-lg px-2.5 py-1.5 text-sm transition-colors",
                        active ? "bg-white/[0.07] text-white" : "text-white/65 hover:bg-white/[0.04] hover:text-white")}>
                      <Icon className={cn("size-4", active && "text-ember")} aria-hidden />
                      {n.label}
                    </Link>
                  );
                })}
              </div>
            </div>
          ))}
        </nav>
        <div className="mt-4 space-y-1 border-t border-white/5 px-2.5 pt-4 text-xs text-white/40">
          <div className="flex items-center justify-between text-white/70">
            <span>{user.username} · {user.role}</span>
            <button onClick={logout} className="text-white/45 hover:text-white">Sign out</button>
          </div>
          {meta.data ? (
            <>
              <div>API v{meta.data.version} · {meta.data.database}</div>
              <div>Jobs: {meta.data.executor} executor</div>
            </>
          ) : meta.isError ? <div className="text-fail">API unreachable</div> : <div>Connecting…</div>}
        </div>
      </aside>
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="glass-strong sticky top-0 z-20 flex items-center gap-4 border-x-0 border-t-0 px-4 py-3 lg:hidden">
          <Mark />
          <nav className="flex gap-4 overflow-x-auto text-sm">
            {NAV.flatMap((g) => g.items).map((n) => (
              <Link key={n.href} href={n.href} className="whitespace-nowrap text-white/75">{n.label}</Link>
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
