from __future__ import annotations

import numpy as np
import pandas as pd

from evalforge.services import drift

from .conftest import as_user, mixed_frame, to_csv_bytes
from .test_ops import approve, train, upload


def _rows(df: pd.DataFrame) -> list[dict]:
    X = df.drop(columns=["label"])
    return X.astype(object).where(X.notna(), None).to_dict(orient="records")


def deploy_champion(client, vid):
    eng = as_user(client, "engineer")
    run = train(client, vid, "logistic_regression")
    mv = client.post("/api/registry/versions", json={"run_id": run}, headers=eng).json()
    a = client.post(f"/api/registry/versions/{mv['id']}/request-promotion", headers=eng).json()
    approve(client, a["id"])
    p = client.post(
        "/api/deployments",
        json={"model_version_id": mv["id"], "environment": "production"},
        headers=eng,
    ).json()
    approve(client, p["approval"]["id"])
    return mv


def test_statistics_units():
    rng = np.random.default_rng(0)
    same = drift.numeric_drift(
        pd.Series(rng.normal(0, 1, 2000)), pd.Series(rng.normal(0, 1, 500)), drift.DEFAULTS
    )
    moved = drift.numeric_drift(
        pd.Series(rng.normal(0, 1, 2000)), pd.Series(rng.normal(1.0, 1, 500)), drift.DEFAULTS
    )
    assert not same["drifted"] and moved["drifted"] and moved["psi"] > 0.2
    cat = drift.categorical_drift(
        pd.Series(["a"] * 80 + ["b"] * 20), pd.Series(["a"] * 20 + ["b"] * 80), drift.DEFAULTS
    )
    assert cat["drifted"]


def test_drift_lifecycle_incident_and_controlled_retraining(client, churn_csv):
    ds = upload(client, churn_csv)
    vid = ds["versions"][0]["id"]
    mv = deploy_champion(client, vid)
    eng = as_user(client, "engineer")

    # in-distribution traffic -> no drift
    normal = mixed_frame(160, seed=21)
    client.post("/api/serve/churn/predict", json={"instances": _rows(normal)})
    r = client.post("/api/drift/reports", json={"model_version_id": mv["id"]}, headers=eng).json()
    assert r["summary"]["verdict"] == "no_drift", r["summary"]
    assert client.get("/api/incidents").json() == []

    # shifted traffic without labels -> statistical drift only
    shifted = mixed_frame(160, seed=22)
    shifted["age"] = shifted["age"] + 25
    shifted["income"] = shifted["income"] * 1.8
    client.post(
        "/api/serve/churn/predict?environment=production", json={"instances": _rows(shifted)}
    )
    r2 = client.post(
        "/api/drift/reports", json={"model_version_id": mv["id"], "hours": 1}, headers=eng
    ).json()
    assert r2["summary"]["data_drift"] and "age" in r2["summary"]["drifted_features"]
    assert r2["summary"]["verdict"] == "statistical_drift_without_labels"
    assert r2["summary"]["performance_degradation"] is None
    age = next(f for f in r2["features"] if f["feature"] == "age")
    assert len(age["histogram"]["reference"]) == 20
    inc = client.get("/api/incidents").json()[0]
    assert inc["kind"] == "data_drift"

    plan = client.post(f"/api/incidents/{inc['id']}/retraining-plan", json={}, headers=eng).json()
    assert not plan["justified"]
    r = client.post(f"/api/incidents/{inc['id']}/request-retraining", json={}, headers=eng)
    assert r.status_code == 409 and "not justified" in r.json()["detail"]

    # labels arrive and show the model is wrong on the shifted population
    preds = client.post("/api/serve/churn/predict", json={"instances": _rows(shifted)}).json()
    flipped = {
        p["id"]: ("stay" if p["label"] == "churn" else "churn") for p in preds["predictions"]
    }
    assert (
        client.post("/api/serve/feedback", json={"labels": flipped}, headers=eng).json()["updated"]
        == 160
    )
    r3 = client.post(
        "/api/drift/reports", json={"model_version_id": mv["id"], "hours": 1}, headers=eng
    ).json()
    assert r3["summary"]["verdict"] == "performance_degradation"
    assert r3["performance"]["current"]["f1"] < r3["performance"]["baseline_f1"]
    perf_inc = next(i for i in client.get("/api/incidents").json() if i["kind"] == "performance")
    assert perf_inc["severity"] == "high"

    # a new labelled data version justifies retraining; it runs only after approval
    df2 = pd.concat(
        [mixed_frame(240), shifted.assign(label=np.where(shifted["age"] > 60, "churn", "stay"))]
    ).reset_index(drop=True)
    v2 = client.post(
        f"/api/datasets/{ds['id']}/versions",
        files={"file": ("v2.csv", to_csv_bytes(df2), "text/csv")},
    ).json()
    plan = client.post(
        f"/api/incidents/{perf_inc['id']}/retraining-plan",
        json={"dataset_version_id": v2["id"]},
        headers=eng,
    ).json()
    assert plan["justified"] and plan["config"]["dataset_version_id"] == v2["id"]
    req = client.post(
        f"/api/incidents/{perf_inc['id']}/request-retraining",
        json={"dataset_version_id": v2["id"]},
        headers=eng,
    ).json()
    pipelines_before = client.get("/api/pipelines/runs").json()
    assert pipelines_before == []  # nothing runs before approval
    done = approve(client, req["approval_id"]).json()
    assert done["status"] == "executed", done
    run = client.get(f"/api/pipelines/runs/{done['result']['pipeline_run_id']}").json()
    assert run["trigger"] == "incident"
    assert run["status"] in ("awaiting_approval", "failed")  # gate decides on measured metrics
    gate = next(s for s in run["steps"] if s["name"] == "quality_gate")
    if run["status"] == "awaiting_approval":
        assert gate["output"]["champion_comparison"]["evaluated_on_dataset_version"] == 2
    events = client.get(f"/api/incidents/{perf_inc['id']}").json()["events"]
    assert [e["kind"] for e in events][:3] == ["detected", "action", "action"]


def test_drift_against_dataset_version(client, churn_csv):
    ds = upload(client, churn_csv)
    mv = deploy_champion(client, ds["versions"][0]["id"])
    df = mixed_frame(200, seed=5)
    df["plan"] = "pro"
    v2 = client.post(
        f"/api/datasets/{ds['id']}/versions",
        files={"file": ("v2.csv", to_csv_bytes(df), "text/csv")},
    ).json()
    r = client.post(
        "/api/drift/reports",
        json={
            "model_version_id": mv["id"],
            "source": "dataset_version",
            "dataset_version_id": v2["id"],
        },
    ).json()
    plan = next(f for f in r["features"] if f["feature"] == "plan")
    assert plan["kind"] == "categorical" and plan["drifted"]
    assert r["performance"] is not None  # labelled version -> label-aware check
    few = client.post("/api/drift/reports", json={"model_version_id": mv["id"]})
    assert few.status_code == 422 and "at least" in few.json()["detail"]
