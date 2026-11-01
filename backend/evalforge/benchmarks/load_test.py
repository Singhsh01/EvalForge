"""Load test for a running EvalForge API: throughput, latency percentiles, errors, memory.

    make perf                       # against http://localhost:8000, results in docs/results
    python -m evalforge.benchmarks.load_test --requests 2000 --concurrency 1 4 16

Scenarios (each run at every concurrency level):
  predict-1      single-row inference on the production model (the serving hot path)
  predict-100    100-row batch inference
  overview       GET /api/overview (dashboard aggregate)
  deployments    GET /api/deployments
  traces         GET /api/traces (trace explorer list)

The serving rate limit is per user (default 1200/min); start the API with
EVALFORGE_RATE_LIMIT_SERVE_PER_MIN set high for throughput runs, or the limiter is what
you measure. Resident memory is read from /proc for the API process when --pid is given.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np


def rss_mb(pid: int | None) -> float | None:
    if not pid:
        return None
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024, 1)
    except OSError:
        return None
    return None


def tree_rss_mb(pid: int | None) -> float | None:
    """RSS of the process and its children (uvicorn --reload / workers)."""
    if not pid:
        return None
    total, pids = 0.0, [pid]
    try:
        for p in Path("/proc").iterdir():
            if p.name.isdigit():
                try:
                    if int((p / "stat").read_text().split()[3]) == pid:
                        pids.append(int(p.name))
                except (OSError, ValueError, IndexError):
                    pass
    except OSError:
        pass
    for p in pids:
        total += rss_mb(p) or 0
    return round(total, 1)


async def scenario(
    client: httpx.AsyncClient, method: str, path: str, body: Any, n: int, conc: int
) -> dict[str, Any]:
    lat: list[float] = []
    codes: dict[int, int] = {}
    sem = asyncio.Semaphore(conc)

    async def one() -> None:
        async with sem:
            t0 = time.perf_counter()
            r = await client.request(method, path, json=body)
            lat.append((time.perf_counter() - t0) * 1000)
            codes[r.status_code] = codes.get(r.status_code, 0) + 1

    # warm-up (model load, caches) is excluded from the measurement
    for _ in range(min(10, n)):
        await client.request(method, path, json=body)
    t0 = time.perf_counter()
    await asyncio.gather(*(one() for _ in range(n)))
    wall = time.perf_counter() - t0
    ok = codes.get(200, 0)
    return {
        "requests": n,
        "concurrency": conc,
        "ok": ok,
        "status_codes": {str(k): v for k, v in sorted(codes.items())},
        "throughput_rps": round(ok / wall, 1),
        "latency_ms": {
            "p50": round(float(np.percentile(lat, 50)), 2),
            "p95": round(float(np.percentile(lat, 95)), 2),
            "p99": round(float(np.percentile(lat, 99)), 2),
            "max": round(max(lat), 2),
            "mean": round(statistics.fmean(lat), 2),
        },
    }


async def run(a: argparse.Namespace) -> dict[str, Any]:
    async with httpx.AsyncClient(base_url=a.api, timeout=60) as c:
        tok = (
            await c.post("/api/auth/login", json={"username": a.user, "password": a.password})
        ).json()["token"]
        c.headers["authorization"] = f"Bearer {tok}"
        deps = (await c.get("/api/deployments")).json()
        prod = next(
            (
                d
                for d in deps
                if d["environment"] == a.environment
                and d["status"] == "active"
                and d["role"] == "stable"
            ),
            None,
        )
        if prod is None:
            raise SystemExit(
                f"No active stable {a.environment} deployment; run make demo-enterprise"
            )
        name = prod["model_name"]
        mv = (await c.get(f"/api/registry/versions/{prod['model_version_id']}/card")).json()
        feats = (mv.get("features") or {}).get("used")
        if not feats:
            raise SystemExit("could not read the model's feature list from its card")
        row = {f: 1.0 for f in feats}
        path = f"/api/serve/{name}/predict?environment={a.environment}"
        plan = [
            ("predict-1", "POST", path, {"instances": [row]}),
            ("predict-100", "POST", path, {"instances": [row] * 100}),
            ("overview", "GET", "/api/overview", None),
            ("deployments", "GET", "/api/deployments", None),
            ("traces", "GET", "/api/traces", None),
        ]
        out: dict[str, Any] = {
            "started_at": datetime.now(UTC).isoformat(),
            "api": a.api,
            "host": {
                "platform": platform.platform(),
                "cpus": os.cpu_count(),
                "python": platform.python_version(),
            },
            "model": f"{name} ({a.environment})",
            "rss_mb_idle": tree_rss_mb(a.pid),
            "scenarios": {},
        }
        peak = out["rss_mb_idle"] or 0
        for label, method, p, body in plan:
            if label not in a.only and a.only:
                continue
            res = []
            for conc in a.concurrency:
                n = a.requests if label.startswith("predict-1") else max(50, a.requests // 4)
                r = await scenario(c, method, p, body, n, conc)
                r["rss_mb_after"] = tree_rss_mb(a.pid)
                peak = max(peak, r["rss_mb_after"] or 0)
                res.append(r)
                print(
                    f"{label:12s} c={conc:<3d} {r['throughput_rps']:8.1f} rps  "
                    f"p50 {r['latency_ms']['p50']:7.1f}  p95 {r['latency_ms']['p95']:7.1f}  "
                    f"p99 {r['latency_ms']['p99']:7.1f} ms  codes {r['status_codes']}  "
                    f"rss {r['rss_mb_after']} MB",
                    flush=True,
                )
            out["scenarios"][label] = res
        out["rss_mb_peak"] = peak or None
        return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--api", default=os.environ.get("EVALFORGE_API", "http://localhost:8000"))
    ap.add_argument("--user", default="engineer")
    ap.add_argument("--password", default=os.environ.get("EVALFORGE_PASSWORD", "evalforge-dev"))
    ap.add_argument("--environment", default="production")
    ap.add_argument("--requests", type=int, default=1000)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--pid", type=int, default=None, help="API process id for memory readings")
    ap.add_argument("--out", default="../docs/results/performance.json")
    a = ap.parse_args(argv)
    res = asyncio.run(run(a))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
