"""Reproducible end-to-end demonstration of EvalForge's enterprise workflows.

    python -m evalforge.demo.enterprise_demo --api http://localhost:8000 --out ../docs/results

Everything goes through the public HTTP API with two different accounts (an
*engineer* who builds and requests, an *approver* who decides), exactly as people
would use the UI. Numbers written to ``enterprise-demo.json`` are the values
measured in that execution.

Case 1 — ML lifecycle: public dataset → continuous-training pipeline → quality gate →
         approved promotion → approved production deploy → canary → rollback.
Case 2 — Agent regression: a Copilot configuration change regresses the agent
         evaluation suite → incident → Copilot investigation → fix → approved
         targeted re-evaluation.
Case 3 — Production incident: synthetic sensor-calibration drift → drift report with
         label-confirmed degradation → incident → Copilot investigation → justified,
         approved retraining → gate vs champion → approved release → re-measured.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from datetime import UTC, datetime
from typing import Any

import httpx
import numpy as np
import pandas as pd

from evalforge.demo.data import breast_cancer_frame

PASSWORD = os.environ.get("EVALFORGE_BOOTSTRAP_PASSWORD", "evalforge-dev")  # nosec B105 - documented dev default
SHIFTED = [
    "mean_radius",
    "mean_perimeter",
    "mean_area",
    "worst_radius",
    "worst_perimeter",
    "worst_area",
    "area_error",
]
CALIBRATION = 1.35  # simulated over-reading of size features after a scanner recalibration


class Client:
    def __init__(self, base: str, user: str) -> None:
        self.c = httpx.Client(base_url=base, timeout=300)
        tok = self.c.post(
            "/api/auth/login", json={"username": user, "password": PASSWORD}
        ).raise_for_status()
        self.c.headers["Authorization"] = f"Bearer {tok.json()['token']}"

    def get(self, path: str, **kw: Any) -> Any:
        return self.c.get(path, **kw).raise_for_status().json()

    def post(
        self, path: str, body: Any = None, ok: tuple[int, ...] = (200, 201, 202), **kw: Any
    ) -> Any:
        r = self.c.post(path, json=body, **kw)
        if r.status_code not in ok:
            raise RuntimeError(f"POST {path} -> {r.status_code}: {r.text[:400]}")
        return r.json()


def wait(cond, timeout: float = 600, every: float = 1.0) -> Any:  # type: ignore[no-untyped-def]
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = cond()
        if v:
            return v
        time.sleep(every)
    raise TimeoutError("condition not met")


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def approve(appr: Client, approval_id: str, reason: str) -> dict:
    r = appr.post(f"/api/approvals/{approval_id}/decision", {"approve": True, "reason": reason})
    if r["status"] != "executed":
        raise RuntimeError(f"approval {approval_id} {r['status']}: {r.get('result')}")
    return r


def pipeline(eng: Client, cfg: dict) -> dict:
    run = eng.post("/api/pipelines/runs", {"config": cfg})
    return wait(
        lambda: (
            (r := eng.get(f"/api/pipelines/runs/{run['id']}"))["status"]
            not in ("queued", "running")
            and r
        )
    )


def traffic(df: pd.DataFrame, n: int, seed: int, scale: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    s = df.sample(n, replace=True, random_state=seed).reset_index(drop=True)
    feats = [c for c in s.columns if c != "diagnosis"]
    s[feats] = s[feats] * (1 + rng.normal(0, 0.01, (n, len(feats))))  # measurement noise
    if scale != 1.0:
        s[SHIFTED] = s[SHIFTED] * scale
    return s


def send(eng: Client, model: str, df: pd.DataFrame, label: bool) -> list[dict]:
    out = []
    for i in range(0, len(df), 100):
        chunk = df.iloc[i : i + 100]
        inst = chunk.drop(columns=["diagnosis"]).to_dict(orient="records")
        out += eng.post(f"/api/serve/{model}/predict", {"instances": inst})["predictions"]
    if label:
        eng.post(
            "/api/serve/feedback",
            {"labels": {p["id"]: y for p, y in zip(out, df["diagnosis"], strict=True)}},
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    eng, appr = Client(args.api, "engineer"), Client(args.api, "approver")
    res: dict[str, Any] = {"generated_at": datetime.now(UTC).isoformat()}
    base = breast_cancer_frame()

    # ------------------------------------------------------------------ case 1
    log("Case 1 · ML lifecycle")
    ds_id = eng.post("/api/demo/breast-cancer")["dataset_id"]
    ds = eng.get(f"/api/datasets/{ds_id}")
    v1 = ds["versions"][0]
    p1 = pipeline(
        eng,
        {
            "dataset_version_id": v1["id"],
            "algorithm": "logistic_regression",
            "hyperparameters": {"C": 1.0, "max_iter": 2000},
            "cv_folds": 5,
        },
    )
    assert p1["status"] == "awaiting_approval", p1
    log(f"  pipeline {p1['id'][:8]} -> {p1['status']}; gate passed")
    approve(appr, p1["lineage"]["approval_id"], "Gate passed; CV F1 stable across folds")
    mv1 = p1["lineage"]["model_version_id"]
    req = eng.post("/api/deployments", {"model_version_id": mv1, "environment": "production"})
    approve(appr, req["approval"]["id"], "Champion v1 to production")
    prod1 = eng.get(f"/api/registry/versions/{mv1}")
    log("  champion v1 in production")

    p2 = pipeline(
        eng,
        {
            "dataset_version_id": v1["id"],
            "algorithm": "gradient_boosting",
            "hyperparameters": {"learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 15},
            "cv_folds": 5,
            "request_promotion": False,
        },
    )
    gate2 = next(s for s in p2["steps"] if s["name"] == "quality_gate")
    res["case1"] = {
        "dataset": {"name": ds["name"], "version": 1, "sha256": v1["sha256"], "rows": v1["n_rows"]},
        "champion": {
            "model_version_id": mv1,
            "metrics": prod1["metrics"],
            "pipeline_run_id": p1["id"],
            "lineage": p1["lineage"],
            "health": prod1["deployments"][0]["health"],
        },
        "challenger_pipeline": {
            "id": p2["id"],
            "status": p2["status"],
            "gate": gate2.get("output") or gate2.get("error"),
        },
    }
    challenger_mv = p2["lineage"].get("model_version_id")
    if p2["status"] == "succeeded" and challenger_mv:
        # Canary the challenger even without promotion? Production requires an approved
        # version, so promote it first via approval (gate already passed).
        a = eng.post(f"/api/registry/versions/{challenger_mv}/request-promotion")
        approve(appr, a["id"], "Challenger within tolerance; canary first")
        c = eng.post(
            "/api/deployments",
            {"model_version_id": challenger_mv, "environment": "production", "traffic_percent": 25},
        )
        approve(appr, c["approval"]["id"], "25% canary")
        split: dict[str, int] = {}
        for i in range(400):
            r = eng.c.post(
                f"/api/serve/{ds['name']}/predict",
                json={
                    "instances": base.drop(columns=["diagnosis"])
                    .iloc[[i % 569]]
                    .to_dict(orient="records")
                },
                headers={"x-request-id": f"canary-{i}"},
            ).json()
            split[f"v{r['version']}-{r['role']}"] = split.get(f"v{r['version']}-{r['role']}", 0) + 1
        rb = eng.post(
            "/api/deployments/rollback", {"model_name": ds["name"], "environment": "production"}
        )
        approve(appr, rb["approval"]["id"], "End canary: keep the simpler champion")
        res["case1"]["canary"] = {
            "traffic_percent": 25,
            "observed_split_over_400": split,
            "rolled_back": True,
        }
        log(f"  canary split {split}; canary rolled back")
        # restore the champion alias to v1 for the drift scenario
        a = eng.post(f"/api/registry/versions/{mv1}/request-promotion", ok=(201, 409))
        if "id" in a:
            approve(appr, a["id"], "Restore v1 as champion")
    else:
        log(f"  challenger stopped at the gate: {gate2.get('error')}")

    # ------------------------------------------------------------------ case 3
    log("Case 3 · drift incident")
    model = ds["name"]
    send(eng, model, traffic(base, 300, seed=1), label=True)
    r_ok = eng.post("/api/drift/reports", {"model_version_id": mv1, "hours": 1})
    log(f"  in-distribution report: {r_ok['summary']['verdict']}")
    time.sleep(1.2)
    t_start = datetime.now(UTC)
    drifted = traffic(base, 300, seed=2, scale=CALIBRATION)
    send(eng, model, drifted, label=True)
    # Only the post-recalibration window, so the healthy batch does not dilute the signal.
    r_bad = eng.post(
        "/api/drift/reports", {"model_version_id": mv1, "hours": 1, "since": t_start.isoformat()}
    )
    log(
        f"  after calibration change: {r_bad['summary']['verdict']} "
        f"(F1 {r_bad['performance']['baseline_f1']:.3f} -> "
        f"{r_bad['performance']['current']['f1']:.3f})"
    )
    inc = next(i for i in eng.get("/api/incidents") if i["kind"] == "performance")
    inv = eng.post(f"/api/incidents/{inc['id']}/investigate")
    log(
        f"  copilot: {len(inv['report']['findings'])} findings, "
        f"{len(inv['report']['hypotheses'])} hypotheses"
    )
    # New labelled data: original + recalibrated, labelled cases from the field
    field = traffic(base, 250, seed=3, scale=CALIBRATION)
    v2_df = pd.concat([base, field], ignore_index=True)
    buf = io.StringIO()
    v2_df.to_csv(buf, index=False, float_format="%.6g")
    v2 = (
        eng.c.post(
            f"/api/datasets/{ds_id}/versions",
            files={"file": ("v2.csv", buf.getvalue().encode(), "text/csv")},
            data={"note": "includes 250 labelled cases measured after recalibration"},
        )
        .raise_for_status()
        .json()
    )
    plan = eng.post(f"/api/incidents/{inc['id']}/retraining-plan", {"dataset_version_id": v2["id"]})
    rq = eng.post(
        f"/api/incidents/{inc['id']}/request-retraining", {"dataset_version_id": v2["id"]}
    )
    ok = approve(appr, rq["approval_id"], "Label-confirmed degradation; new data available")
    pr = wait(
        lambda: (
            (r := eng.get(f"/api/pipelines/runs/{ok['result']['pipeline_run_id']}"))["status"]
            not in ("queued", "running")
            and r
        )
    )
    gate3 = next(s for s in pr["steps"] if s["name"] == "quality_gate")
    log(f"  retraining pipeline {pr['id'][:8]} -> {pr['status']}")
    after = None
    if pr["status"] == "awaiting_approval":
        approve(appr, pr["lineage"]["approval_id"], "Candidate beats champion on v2 holdout")
        mv_new = pr["lineage"]["model_version_id"]
        d = eng.post("/api/deployments", {"model_version_id": mv_new, "environment": "production"})
        approve(appr, d["approval"]["id"], "Release retrained model")
        t_rel = datetime.now(UTC)
        send(eng, model, traffic(base, 300, seed=4, scale=CALIBRATION), label=True)
        after = eng.post(
            "/api/drift/reports",
            {"model_version_id": mv_new, "hours": 1, "since": t_rel.isoformat()},
        )
        log(
            f"  after release: F1 on recalibrated traffic "
            f"{after['performance']['current']['f1']:.3f}"
        )
        eng.post(
            f"/api/incidents/{inc['id']}/status",
            {"status": "resolved", "note": "Retrained model released; F1 re-measured"},
        )
    res["case3"] = {
        "calibration_factor": CALIBRATION,
        "shifted_features": SHIFTED,
        "window_started": t_start.isoformat(),
        "in_distribution_report": {
            "id": r_ok["id"],
            "summary": r_ok["summary"],
            "f1": (r_ok["performance"] or {}).get("current", {}).get("f1"),
        },
        "drifted_report": {
            "id": r_bad["id"],
            "summary": r_bad["summary"],
            "performance": r_bad["performance"],
            "top_features": [
                {"feature": f["feature"], "psi": f["psi"]} for f in r_bad["features"][:8]
            ],
        },
        "incident_id": inc["id"],
        "copilot_session": inv["id"],
        "copilot_hypotheses": inv["report"]["hypotheses"],
        "retraining_plan": {"justified": plan["justified"], "reasons": plan["reasons"]},
        "retraining_pipeline": {
            "id": pr["id"],
            "status": pr["status"],
            "gate": gate3.get("output") or gate3.get("error"),
        },
        "after_release_report": after
        and {"id": after["id"], "summary": after["summary"], "performance": after["performance"]},
    }

    # ------------------------------------------------------------------ case 2
    log("Case 2 · agent regression")
    cfgs = eng.get("/api/copilot/configs")
    v1cfg = cfgs[0]
    cases = [
        {
            "id": "regression",
            "question": "Why did my latest model perform worse?",
            "expected_tools": ["query_mlflow_runs", "compare_runs", "diff_dataset_versions"],
            "expected_answer_contains": ["Training data changed"],
        },
        {
            "id": "dataset",
            "question": "Which dataset changes are responsible for this regression?",
            "expected_tools": ["diff_dataset_versions"],
            "expected_answer_contains": ["Mean holdout F1 by dataset version"],
        },
        {
            "id": "recommend",
            "question": "Compare the last five experiments and recommend a candidate model.",
            "expected_tools": ["query_mlflow_runs", "recommend_candidate"],
            "expected_answer_contains": ["Pareto front"],
        },
        {
            "id": "drift",
            "question": "Investigate the drift incident in production.",
            "expected_tools": ["get_drift_reports", "list_incidents"],
            "expected_answer_contains": ["Drift report"],
        },
        {
            "id": "deploy",
            "question": "Investigate the latest deployment failure.",
            "expected_tools": ["get_deployments"],
            "expected_answer_contains": ["Deployment"],
        },
    ]
    suite = eng.post(
        "/api/evals/suites",
        {
            "name": "copilot-investigations",
            "kind": "agent",
            "cases": cases,
            "description": "Investigation tasks the Copilot must solve with the right tools.",
            "evaluators": [
                {"type": "tool_selection"},
                {"type": "task_completion"},
                {"type": "agent_efficiency", "max_steps": 8},
            ],
            "gates": {"pass_rate": 1.0, "regression_tolerance": 0.05},  # nosec B105 - not a secret
        },
    )

    def ev(cfg_id: str) -> dict:
        r = eng.post(
            "/api/evals/runs",
            {"suite_id": suite["id"], "subject": {"type": "agent", "agent_config_id": cfg_id}},
        )
        return wait(
            lambda: (
                (x := eng.get(f"/api/evals/runs/{r['id']}"))["status"] not in ("queued", "running")
                and x
            )
        )

    base_run = ev(v1cfg["id"])
    trimmed = [t for t in v1cfg["config"]["tools_enabled"] if t != "diff_dataset_versions"]
    v2cfg = eng.post(
        "/api/copilot/configs",
        {"config": {"tools_enabled": trimmed}, "note": "trim tool list to reduce latency"},
    )
    bad_run = ev(v2cfg["id"])
    log(
        f"  pass rate {base_run['summary']['pass_rate']:.0%} -> "
        f"{bad_run['summary']['pass_rate']:.0%}"
    )
    inc2 = next(i for i in eng.get("/api/incidents") if i["kind"] == "agent_regression")
    inv2 = eng.post(f"/api/incidents/{inc2['id']}/investigate")
    v3cfg = eng.post(
        "/api/copilot/configs",
        {
            "config": {"tools_enabled": v1cfg["config"]["tools_enabled"]},
            "note": "restore diff_dataset_versions",
        },
    )
    fix = eng.post(
        "/api/evals/runs",
        {"suite_id": suite["id"], "subject": {"type": "agent", "agent_config_id": v3cfg["id"]}},
    )
    fixed = wait(
        lambda: (
            (x := eng.get(f"/api/evals/runs/{fix['id']}"))["status"] not in ("queued", "running")
            and x
        )
    )
    eng.post(
        f"/api/incidents/{inc2['id']}/status",
        {
            "status": "resolved",
            "note": f"Config v{v3cfg['version']} restores the tool; "
            f"eval {fixed['id'][:8]} back to baseline",
        },
    )
    log(f"  fixed config pass rate {fixed['summary']['pass_rate']:.0%}")
    res["case2"] = {
        "suite": f"{suite['name']}@v{suite['version']}",
        "cases": len(cases),
        "baseline": {
            "run_id": base_run["id"],
            "config": f"copilot@v{v1cfg['version']}",
            "summary": base_run["summary"],
        },
        "regressed": {
            "run_id": bad_run["id"],
            "config": f"copilot@v{v2cfg['version']}",
            "summary": bad_run["summary"],
            "regression": bad_run["regression"],
        },
        "incident_id": inc2["id"],
        "copilot_session": inv2["id"],
        "copilot_findings": inv2["report"]["findings"],
        "copilot_hypotheses": inv2["report"]["hypotheses"],
        "fixed": {
            "run_id": fixed["id"],
            "config": f"copilot@v{v3cfg['version']}",
            "summary": fixed["summary"],
            "regression": fixed["regression"],
        },
    }

    if args.out:
        from pathlib import Path

        p = Path(args.out)
        p.mkdir(parents=True, exist_ok=True)
        (p / "enterprise-demo.json").write_text(json.dumps(res, indent=2, default=str))
        log(f"wrote {p / 'enterprise-demo.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
