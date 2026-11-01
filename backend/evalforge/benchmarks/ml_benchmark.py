"""Classic-ML benchmark across public datasets, through the EvalForge API.

Uses datasets bundled with scikit-learn (no download; labels come from the original
sources, independent of EvalForge): Breast Cancer Wisconsin (binary, 569 rows), Wine
(3 classes, 178 rows) and Optical Digits (10 classes, 1797 rows). Each built-in algorithm
is trained through the normal upload -> validate -> train path, and the report records
holdout F1 with its bootstrap 95% interval, cross-validated F1 (mean ± std), single-row
latency and training time.

    make ml-benchmark            # API running; writes docs/results/ml-benchmark.json
"""

from __future__ import annotations

import argparse
import io
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from sklearn import datasets

DATASETS = {
    "bench-breast-cancer": (datasets.load_breast_cancer, "binary"),
    "bench-wine": (datasets.load_wine, "3 classes"),
    "bench-digits": (datasets.load_digits, "10 classes"),
}
ALGORITHMS = ["logistic_regression", "random_forest", "gradient_boosting"]


def frame(loader: Any) -> pd.DataFrame:
    d = loader(as_frame=True)
    df = d.frame.copy()
    names = list(d.target_names) if hasattr(d, "target_names") else None
    df["label"] = [f"c_{names[i]}" if names is not None else f"c_{i}" for i in d.target]
    return df.drop(columns=["target"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--api", default=os.environ.get("EVALFORGE_API", "http://localhost:8000"))
    ap.add_argument("--password", default=os.environ.get("EVALFORGE_PASSWORD", "evalforge-dev"))
    ap.add_argument("--out", default="../docs/results/ml-benchmark.json")
    a = ap.parse_args(argv)
    c = httpx.Client(base_url=a.api, timeout=600)
    tok = c.post("/api/auth/login", json={"username": "engineer", "password": a.password})
    c.headers["authorization"] = f"Bearer {tok.raise_for_status().json()['token']}"
    existing = {d["name"]: d for d in c.get("/api/datasets").json()}
    results: dict[str, Any] = {"started_at": datetime.now(UTC).isoformat(), "datasets": {}}
    for name, (loader, kind) in DATASETS.items():
        df = frame(loader)
        if name in existing:
            vid = existing[name]["versions"][-1]["id"]
        else:
            buf = io.BytesIO()
            df.to_csv(buf, index=False)
            r = c.post(
                "/api/datasets",
                data={"name": name, "target_column": "label"},
                files={"file": (f"{name}.csv", buf.getvalue(), "text/csv")},
            )
            r.raise_for_status()
            vid = r.json()["versions"][0]["id"]
        rows = []
        for algo in ALGORITHMS:
            run = (
                c.post(
                    "/api/runs",
                    json={
                        "dataset_version_id": vid,
                        "algorithm": algo,
                        "hyperparameters": {},
                        "cv_folds": 5,
                    },
                )
                .raise_for_status()
                .json()["run"]
            )
            t0 = time.monotonic()
            while (cur := c.get(f"/api/runs/{run['id']}").json())["status"] not in (
                "succeeded",
                "failed",
            ):
                if time.monotonic() - t0 > 900:
                    raise TimeoutError(run["id"])
                time.sleep(2)
            m = cur.get("metrics") or {}
            rows.append(
                {
                    "algorithm": algo,
                    "status": cur["status"],
                    "error": cur.get("error"),
                    "run_id": cur["id"],
                    "holdout_f1": m.get("f1"),
                    "holdout_f1_ci95": [m.get("f1_ci95_low"), m.get("f1_ci95_high")],
                    "cv_f1_macro": [m.get("cv_f1_macro_mean"), m.get("cv_f1_macro_std")],
                    "accuracy": m.get("accuracy"),
                    "latency_single_p95_ms": m.get("latency_single_p95_ms"),
                    "training_seconds": cur.get("training_seconds"),
                }
            )
            print(
                f"{name:20s} {algo:20s} {cur['status']:9s} F1 {m.get('f1')} "
                f"CI [{m.get('f1_ci95_low')}, {m.get('f1_ci95_high')}] "
                f"CV {m.get('cv_f1_macro_mean')}±{m.get('cv_f1_macro_std')}",
                flush=True,
            )
        results["datasets"][name] = {
            "task": kind,
            "rows": len(df),
            "features": df.shape[1] - 1,
            "runs": rows,
        }
    results["finished_at"] = datetime.now(UTC).isoformat()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=2))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
