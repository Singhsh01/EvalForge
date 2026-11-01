"""Real-LLM benchmark through the EvalForge API.

Runs three evaluation suites with programmatically generated ground truth (so the answer
key is independent of any model) against a real model reference, then reports pass
rates with Wilson 95 % intervals, bootstrap intervals for mean scores, latency
percentiles, token counts and throughput. Optionally runs the Copilot with the LLM
planner on the existing ``copilot-investigations`` agent suite.

    # API started with an OpenAI-compatible endpoint, e.g. local Ollama:
    EVALFORGE_OPENAI_BASE_URL=http://localhost:11434/v1 EVALFORGE_LLM_TIMEOUT_SECONDS=180 make dev
    make llm-benchmark MODEL=openai:qwen2.5:3b

Nothing here estimates or invents a number: every figure in the output file is computed
from stored evaluation results.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import numpy as np

SEED = 20261101
CITIES = ["Berlin", "Ulm", "Hamburg", "Munich", "Leipzig", "Bremen", "Dresden", "Cologne"]
PRODUCTS = ["valve kits", "sensor boards", "pump seals", "relay modules", "cable looms"]

# --------------------------------------------------------------------------- suites


def extraction_cases(n: int = 40, seed: int = SEED) -> list[dict[str, Any]]:
    """Order notes with distractor numbers; ground truth comes from the generator."""
    rng = random.Random(seed)
    templates = [
        "Order #{oid}: {qty} {prod} for our {city} site, ship by {d}. (Ref. invoice {inv}.)",
        "Hi, please process order {oid}. We need {qty} {prod} delivered to {city} no later "
        "than {d}. Last time we ordered {qty2}.",
        "PO {oid} | qty {qty} | item: {prod} | destination {city} | ship date {d} | "
        "cost centre {inv}",
        "Following our call: {qty} {prod}, order number {oid}, going to {city}. Dispatch on "
        "{d} please; the {city2} warehouse is closed.",
    ]
    out = []
    for i in range(n):
        oid, qty, qty2, inv = (
            rng.randint(10000, 99999),
            rng.randint(1, 250),
            rng.randint(1, 250),
            rng.randint(100, 999),
        )
        city, city2 = rng.sample(CITIES, 2)
        d = date(2026, 1, 1) + timedelta(days=rng.randint(0, 364))
        text = templates[i % len(templates)].format(
            oid=oid,
            qty=qty,
            qty2=qty2,
            inv=inv,
            prod=rng.choice(PRODUCTS),
            city=city,
            city2=city2,
            d=d.isoformat(),
        )
        out.append(
            {
                "id": f"ext-{i:03d}",
                "inputs": {"text": text},
                "expected_json": {
                    "order_id": oid,
                    "quantity": qty,
                    "city": city,
                    "ship_date": d.isoformat(),
                },
            }
        )
    return out


EXTRACTION_SCHEMA = {
    "type": "object",
    "required": ["order_id", "quantity", "city", "ship_date"],
    "properties": {
        "order_id": {"type": "integer"},
        "quantity": {"type": "integer"},
        "city": {"type": "string"},
        "ship_date": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"},
    },
}


def arithmetic_cases(n: int = 30, seed: int = SEED) -> list[dict[str, Any]]:
    """Two-step word problems; the answer is computed, not written by hand."""
    rng = random.Random(seed + 1)
    out = []
    for i in range(n):
        a, b, c = rng.randint(20, 400), rng.randint(5, 90), rng.randint(2, 30)
        kind = i % 3
        if kind == 0:
            q = (
                f"A warehouse holds {a} boxes. It receives {b} more boxes and then ships {c}. "
                "How many boxes are left?"
            )
            ans = a + b - c
        elif kind == 1:
            q = (
                f"There are {c} crates with {b} bottles each. {c - 1} bottles break. "
                "How many intact bottles remain?"
            )
            ans = c * b - (c - 1)
        else:
            q = (
                f"A team ships {b} parcels per day for {c} days, then returns {c} parcels. "
                "How many parcels were shipped net?"
            )
            ans = b * c - c
        out.append(
            {
                "id": f"math-{i:03d}",
                "inputs": {"question": q},
                "expected": str(ans),
                "expected_regex": rf"(?<![\d.]){ans}(?![\d.])",
            }
        )
    return out


def rag_suite(n_questions: int = 20, seed: int = SEED) -> tuple[list[dict], list[dict]]:
    """Fictional component facts (so no model knows them) with near-duplicate distractors."""
    rng = random.Random(seed + 2)
    names = [
        "Zephyr",
        "Halden",
        "Corvin",
        "Altair",
        "Brisk",
        "Nimbus",
        "Orrin",
        "Sable",
        "Tamsin",
        "Vireo",
    ]
    kinds = [
        ("pump", "bar", "operating pressure"),
        ("motor", "rpm", "rated speed"),
        ("sensor", "ms", "response time"),
        ("valve", "mm", "nominal bore"),
    ]
    corpus, facts = [], []
    for nm in names:
        for kd, unit, attr in kinds:
            val, year = rng.randint(3, 900), rng.randint(2009, 2025)
            pid = f"{nm.lower()}-{kd}"
            corpus.append(
                {
                    "id": pid,
                    "text": f"The {nm} {kd} has a {attr} of {val} {unit}. "
                    f"It was introduced in {year} and is maintained by the {nm} group.",
                }
            )
            facts.append((pid, nm, kd, attr, val, unit))
    cases = []
    for i, (pid, nm, kd, attr, val, _unit) in enumerate(rng.sample(facts, n_questions)):
        cases.append(
            {
                "id": f"rag-{i:03d}",
                "question": f"What is the {attr} of the {nm} {kd}?",
                "relevant_ids": [pid],
                "expected_contains": [str(val)],
            }
        )
    return corpus, cases


# --------------------------------------------------------------------------- statistics


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(max(0.0, mid - half), 4), round(min(1.0, mid + half), 4))


def bootstrap_mean(xs: list[float], iters: int = 2000, seed: int = SEED) -> list[float]:
    if not xs:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    a = np.asarray(xs, dtype=float)
    means = rng.choice(a, size=(iters, len(a)), replace=True).mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def describe(run: dict[str, Any]) -> dict[str, Any]:
    rs = run.get("results", [])
    n = len(rs)
    passed = sum(1 for r in rs if r["passed"] and not r["error"])
    lat = [r["latency_ms"] for r in rs if not r["error"]]
    tout = sum(r["output_tokens"] or 0 for r in rs)
    tin = sum(r["input_tokens"] or 0 for r in rs)
    scores: dict[str, list[float]] = {}
    for r in rs:
        for k, v in (r["scores"] or {}).items():
            scores.setdefault(k, []).append(float(v["score"]))

    def pct(q: float) -> float | None:
        return round(float(np.percentile(lat, q)), 1) if lat else None

    return {
        "suite": run["suite"],
        "run_id": run["id"],
        "status": run["status"],
        "cases": n,
        "passed": passed,
        "errors": sum(1 for r in rs if r["error"]),
        "pass_rate": round(passed / n, 4) if n else None,
        "pass_rate_ci95": wilson(passed, n),
        "evaluator_means": {k: round(float(np.mean(v)), 4) for k, v in scores.items()},
        "evaluator_mean_ci95": {k: bootstrap_mean(v) for k, v in scores.items()},
        "latency_ms": {"p50": pct(50), "p95": pct(95), "p99": pct(99), "max": pct(100)},
        "tokens": {"input": tin, "output": tout},
        "output_tokens_per_s": round(tout / (sum(lat) / 1000), 2) if lat and tout else None,
        "cost_usd": round(sum(r["cost_usd"] or 0 for r in rs), 6),
        "failures": [
            {
                "case_id": r["case_id"],
                "error": r["error"],
                "output": str((r["output"] or {}).get("text", ""))[:300],
                "scores": {
                    k: v.get("detail")
                    for k, v in (r["scores"] or {}).items()
                    if not v.get("passed")
                },
            }
            for r in rs
            if not r["passed"] or r["error"]
        ][:15],
    }


# --------------------------------------------------------------------------- API driver


class Api:
    def __init__(self, base: str, user: str, password: str) -> None:
        self.c = httpx.Client(base_url=base, timeout=60)
        r = self.c.post("/api/auth/login", json={"username": user, "password": password})
        r.raise_for_status()
        self.c.headers["authorization"] = f"Bearer {r.json()['token']}"

    def get(self, path: str, **kw: Any) -> Any:
        r = self.c.get(path, **kw)
        r.raise_for_status()
        return r.json()

    def post(self, path: str, body: dict, ok: tuple[int, ...] = (200, 201, 202)) -> Any:
        r = self.c.post(path, json=body)
        if r.status_code not in ok:
            raise RuntimeError(f"POST {path} -> {r.status_code}: {r.text[:400]}")
        return r.json()


def ensure_prompt(api: Api, name: str, system: str, template: str, params: dict) -> str:
    body = {"system": system, "template": template, "params": params, "note": "llm-benchmark"}
    existing = {p["name"]: p for p in api.get("/api/prompts")}
    if name in existing:
        for v in existing[name]["versions"]:
            if v["system"] == system and v["template"] == template and v["params"] == params:
                return v["id"]
        return api.post(f"/api/prompts/{name}/versions", body)["id"]
    return api.post("/api/prompts", {"name": name, **body})["id"]


def ensure_suite(api: Api, body: dict) -> str:
    r = api.c.post("/api/evals/suites", json=body)
    if r.status_code == 201:
        return r.json()["id"]
    if r.status_code == 409:  # identical version exists: reuse it
        for s in api.get("/api/evals/suites"):
            if s["name"] == body["name"]:
                full = api.get(f"/api/evals/suites/{s['id']}")
                if full["cases"] == body["cases"] and full["evaluators"] == body["evaluators"]:
                    return s["id"]
    raise RuntimeError(f"suite {body['name']}: {r.status_code} {r.text[:300]}")


def run_and_wait(api: Api, suite_id: str, subject: dict, timeout_s: float, log: Any) -> dict:
    run = api.post("/api/evals/runs", {"suite_id": suite_id, "subject": subject})
    t0 = time.monotonic()
    while True:
        cur = api.get(f"/api/evals/runs/{run['id']}")
        if cur["status"] in ("succeeded", "failed"):
            return api.get(f"/api/evals/runs/{run['id']}", params={"results": "true"})
        if time.monotonic() - t0 > timeout_s:
            raise TimeoutError(f"eval run {run['id']} still {cur['status']} after {timeout_s}s")
        log(f"    … {cur['status']} ({time.monotonic() - t0:.0f}s)")
        time.sleep(10)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default=os.environ.get("MODEL", "openai:qwen2.5:3b"))
    ap.add_argument("--api", default=os.environ.get("EVALFORGE_API", "http://localhost:8000"))
    ap.add_argument("--user", default="engineer")
    ap.add_argument("--password", default=os.environ.get("EVALFORGE_PASSWORD", "evalforge-dev"))
    ap.add_argument("--out", default="../docs/results")
    ap.add_argument("--quick", action="store_true", help="5 cases per suite (smoke test)")
    ap.add_argument("--skip-agent", action="store_true")
    ap.add_argument("--timeout", type=float, default=3600)
    a = ap.parse_args(argv)

    def log(msg: str) -> None:
        print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)

    api = Api(a.api, a.user, a.password)
    prov = {p["provider"]: p for p in api.get("/api/llm/providers")["providers"]}
    pname = a.model.split(":", 1)[0]
    if not prov.get(pname, {}).get("configured"):
        log(
            f"provider '{pname}' is not configured on the API. For Ollama start the API with "
            "EVALFORGE_OPENAI_BASE_URL=http://localhost:11434/v1"
        )
        return 2
    # Preflight: an unreachable endpoint must abort, not produce a file that reads like a
    # 0 % result (audit BEN-03: the first local run did exactly that).
    base = prov[pname].get("base_url")
    if pname == "openai" and base:
        try:
            listed = httpx.get(f"{base.rstrip('/')}/models", timeout=10).json()
            ids = {m.get("id") for m in listed.get("data", [])}
        except Exception as exc:
            log(
                f"model endpoint {base} is not reachable ({type(exc).__name__}: {exc}). "
                "Is Ollama running? Try: ollama serve   (or open the Ollama app)"
            )
            return 3
        wanted = a.model.split(":", 1)[1]
        if ids and wanted not in ids:
            log(
                f"model '{wanted}' is not available at {base}; available: {sorted(ids)}. "
                f"Try: ollama pull {wanted}"
            )
            return 3
    n = 5 if a.quick else None
    ext, math_cases = extraction_cases(n or 40), arithmetic_cases(n or 30)
    corpus, rag_cases = rag_suite(n or 20)
    params = {"temperature": 0.0, "max_tokens": 200}
    results: dict[str, Any] = {
        "model": a.model,
        "started_at": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "quick": a.quick,
        "suites": {},
    }

    plan = [
        (
            "bench-extraction",
            "prompt",
            ext,
            [{"type": "json_schema", "schema": EXTRACTION_SCHEMA}, {"type": "json_fields"}],
            ensure_prompt(
                api,
                "bench-extract-order",
                "Extract the order fields. Reply with only a JSON object with keys "
                "order_id (integer), quantity (integer), city (string) and ship_date "
                "(YYYY-MM-DD). Ignore other numbers.",
                "{{text}}",
                params,
            ),
            None,
        ),
        (
            "bench-arithmetic",
            "prompt",
            math_cases,
            [{"type": "regex"}],
            ensure_prompt(
                api,
                "bench-arithmetic",
                "Solve the problem. Reply with only the final number, no words.",
                "{{question}}",
                params,
            ),
            None,
        ),
        (
            "bench-rag",
            "rag",
            rag_cases,
            [
                {"type": "retrieval", "k": 3},
                {"type": "contains"},
                {"type": "citations"},
                {"type": "groundedness"},
            ],
            None,
            corpus,
        ),
    ]
    for name, kind, cases, evaluators, pv_id, corp in plan:
        sid = ensure_suite(
            api,
            {
                "name": name + ("-quick" if a.quick else ""),
                "kind": kind,
                "description": "Generated ground truth (llm_benchmark.py)",
                "cases": cases,
                "evaluators": evaluators,
                "corpus": corp,
            },
        )
        subject = (
            {"type": "prompt", "prompt_version_id": pv_id, "model": a.model}
            if kind == "prompt"
            else {"type": "rag", "top_k": 3, "model": a.model}
        )
        log(f"{name}: {len(cases)} cases on {a.model}")
        run = run_and_wait(api, sid, subject, a.timeout, log)
        results["suites"][name] = describe(run)
        d = results["suites"][name]
        log(
            f"  pass {d['passed']}/{d['cases']} ({d['pass_rate']:.0%}, 95% CI "
            f"{d['pass_rate_ci95'][0]:.0%}–{d['pass_rate_ci95'][1]:.0%}), p50 "
            f"{d['latency_ms']['p50']} ms, {d['tokens']['output']} output tokens"
        )

    if not a.skip_agent:
        agent_suites = [
            s for s in api.get("/api/evals/suites") if s["name"] == "copilot-investigations"
        ]
        if not agent_suites:
            log(
                "agent: suite 'copilot-investigations' not found (run `make demo-enterprise` "
                "first); skipping the Copilot LLM-planner benchmark"
            )
        else:
            sid = max(agent_suites, key=lambda s: s["version"])["id"]
            # A separate config name, so the default "copilot" config is left untouched.
            cfg = api.post(
                "/api/copilot/configs",
                {
                    "name": "copilot-bench",
                    "config": {"planner": f"llm:{a.model}", "max_steps": 8, "timeout_seconds": 600},
                    "note": f"llm-benchmark planner {a.model}",
                },
            )
            log(f"agent: copilot-investigations with LLM planner {a.model}")
            run = run_and_wait(
                api, sid, {"type": "agent", "agent_config_id": cfg["id"]}, a.timeout, log
            )
            results["suites"]["copilot-llm-planner"] = describe(run)

    results["finished_at"] = datetime.now(UTC).isoformat()
    errored = {k: v["errors"] for k, v in results["suites"].items() if v["errors"]}
    results["valid"] = not errored
    if errored:
        results["invalid_reason"] = (
            f"cases errored (infrastructure, not model quality): {errored}; "
            "pass rates in this file are not model results"
        )
        log("WARNING: " + results["invalid_reason"])
    slug = re.sub(r"[^a-z0-9]+", "-", a.model.lower()).strip("-")
    suffix = ("-quick" if a.quick else "") + ("" if results["valid"] else "-INVALID")
    out = Path(a.out) / f"llm-benchmark-{slug}{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
