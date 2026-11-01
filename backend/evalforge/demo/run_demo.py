"""Demonstration 1 (Phase 2 slice): register a public dataset, train three
algorithms through the public API, wait for the jobs, compare the runs.

    python -m evalforge.demo.run_demo --api http://localhost:8000 --out ../docs/results

Everything goes through HTTP exactly as the UI does; results printed here are the
measured values from this execution, written to ``phase2-demo.json`` as evidence.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

EXPERIMENTS = [
    ("logistic_regression", {"C": 1.0, "max_iter": 2000}),
    ("random_forest", {"n_estimators": 300, "min_samples_leaf": 2}),
    ("gradient_boosting", {"learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 15}),
]


def wait(c: httpx.Client, job_id: str, timeout: float = 900) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/api/jobs/{job_id}").raise_for_status().json()
        if j["status"] in ("succeeded", "failed"):
            return j
        time.sleep(0.5)
    raise TimeoutError(job_id)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--cv", type=int, default=5)
    args = ap.parse_args()

    with httpx.Client(base_url=args.api, timeout=60) as c:
        meta = c.get("/api/meta").raise_for_status().json()
        ds_id = c.post("/api/demo/breast-cancer").raise_for_status().json()["dataset_id"]
        ds = c.get(f"/api/datasets/{ds_id}").raise_for_status().json()
        version = ds["versions"][0]
        print(
            f"dataset {ds['name']} v{version['version']}  rows={version['n_rows']} "
            f"status={version['status']} sha256={version['sha256'][:12]}"
        )

        launched = []
        for algo, hp in EXPERIMENTS:
            r = c.post(
                "/api/runs",
                json={
                    "dataset_version_id": version["id"],
                    "algorithm": algo,
                    "hyperparameters": hp,
                    "cv_folds": args.cv,
                    "seed": 42,
                },
            )
            r.raise_for_status()
            launched.append(r.json())
            print(f"launched {algo:<20} job={r.json()['job']['id'][:8]}")

        for item in launched:
            j = wait(c, item["job"]["id"])
            print(f"  {item['run']['algorithm']:<20} {j['status']}  {j['logs'][-1]['message']}")

        ids = ",".join(i["run"]["id"] for i in launched)
        cmp = c.get("/api/runs/compare", params={"ids": ids}).raise_for_status().json()
        first = c.get(f"/api/runs/{launched[0]['run']['id']}").raise_for_status().json()
        pos_label = (first["report"] or {}).get("positive_label")
        n_test = (first["report"] or {}).get("n_samples")
        print(f"positive label: {pos_label}  holdout rows: {n_test}")

    keys = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "log_loss",
        "cv_f1_macro_mean",
        "cv_f1_macro_std",
        "training_seconds",
        "latency_single_p50_ms",
    ]
    print("\n" + f"{'algorithm':<22}" + "".join(f"{k[:14]:>16}" for k in keys))
    for run in cmp["runs"]:
        m = run["metrics"] or {}
        print(f"{run['algorithm']:<22}" + "".join(f"{m.get(k, float('nan')):>16.4f}" for k in keys))
    algo = {r["id"]: r["algorithm"] for r in cmp["runs"]}
    print("\nbest:", {k: [algo[i] for i in v] for k, v in cmp["best"].items() if k in keys})

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        evidence = {
            "generated_at": datetime.now(UTC).isoformat(),
            "host": {
                "python": platform.python_version(),
                "machine": platform.machine(),
                "system": platform.system(),
            },
            "platform": {k: meta[k] for k in ("version", "executor", "database")},
            "dataset": {
                "name": ds["name"],
                "version": version["version"],
                "sha256": version["sha256"],
                "rows": version["n_rows"],
            },
            "positive_label": pos_label,
            "holdout_rows": n_test,
            "runs": [
                {
                    "algorithm": r["algorithm"],
                    "hyperparameters": r["hyperparameters"],
                    "mlflow_run_id": r["mlflow_run_id"],
                    "metrics": r["metrics"],
                }
                for r in cmp["runs"]
            ],
            "best": {k: [algo[i] for i in v] for k, v in cmp["best"].items()},
        }
        (args.out / "phase2-demo.json").write_text(json.dumps(evidence, indent=2))
        print(f"\nevidence written to {args.out / 'phase2-demo.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
