"use client";

import { cn } from "@/lib/format";
import { motion } from "motion/react";

// The platform lifecycle. A stage is "live" only when the backend for it exists
// AND it has real records; otherwise it is "ready" (built, no data yet) or
// "planned" (later phase). Nothing here is decorative state.
export type StageState = "live" | "ready" | "planned";
export type Stage = { key: string; label: string; detail: string; state: StageState };

export function ForgeLine({ stages }: { stages: Stage[] }) {
  return (
    <ol className="relative grid grid-cols-2 gap-x-2 gap-y-6 sm:grid-cols-5 xl:grid-cols-10" aria-label="Model lifecycle">
      {stages.map((s, i) => (
        <motion.li
          key={s.key}
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: i * 0.06, duration: 0.35, ease: "easeOut" }}
          className="relative"
        >
          <div className="flex items-center">
            <span
              className={cn(
                "relative z-10 grid size-9 shrink-0 place-items-center rounded-full text-[13px] font-semibold tabular-nums",
                s.state === "live" && "bg-ember text-ink-950 shadow-[0_0_24px_rgb(245_176_76/0.55)]",
                s.state === "ready" && "bg-ink-800 text-ember ring-1 ring-ember/50",
                s.state === "planned" && "bg-ink-800/60 text-white/30 ring-1 ring-white/10",
              )}
            >
              {i + 1}
            </span>
            {i < stages.length - 1 && (
              <span className="relative mx-1 hidden h-px flex-1 overflow-hidden bg-white/10 xl:block">
                {s.state === "live" && stages[i + 1].state !== "planned" && (
                  <motion.span
                    className="absolute inset-y-0 left-0 w-1/3 bg-gradient-to-r from-transparent via-ember to-transparent"
                    animate={{ x: ["-100%", "300%"] }}
                    transition={{ duration: 2.4, repeat: Infinity, ease: "linear", delay: i * 0.2 }}
                  />
                )}
              </span>
            )}
          </div>
          <div className={cn("mt-2.5 text-sm font-medium", s.state === "planned" ? "text-white/35" : "text-white")}>
            {s.label}
          </div>
          <div className="mt-0.5 text-xs leading-snug text-white/45">{s.detail}</div>
        </motion.li>
      ))}
    </ol>
  );
}
