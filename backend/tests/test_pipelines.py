from __future__ import annotations

import numpy as np

from .conftest import as_user, mixed_frame, to_csv_bytes
from .test_ops import approve, upload


def trigger(client, cfg, headers=None):
    r = client.post("/api/pipelines/runs", json={"config": cfg}, headers=headers or {})
    assert r.status_code == 202, r.text
    return client.get(f"/api/pipelines/runs/{r.json()['id']}").json()


def steps(run):
    return {s["name"]: s for s in run["steps"]}


def test_definition_is_verified():
    from evalforge.services import pipelines

    pipelines.verify_definition("continuous_training")
    try:
        pipelines.verify_definition("nope")
    except ValueError:
        pass
    else:
        raise AssertionError


def test_continuous_training_pipeline_to_approval(client, churn_csv):
    ds = upload(client, churn_csv)
    vid = ds["versions"][0]["id"]
    eng = as_user(client, "engineer")
    run = trigger(
        client, {"dataset_version_id": vid, "algorithm": "logistic_regression", "cv_folds": 3}, eng
    )
    s = steps(run)
    assert run["status"] == "awaiting_approval", run
    assert [x["status"] for x in run["steps"]] == ["succeeded"] * 6 + ["waiting_approval"]
    assert s["deploy_staging"]["output"]["health"]["healthy"]
    lin = run["lineage"]
    assert lin["dataset_sha256"] == ds["versions"][0]["sha256"]
    assert lin["run_id"] and lin["mlflow_run_id"] and lin["model_version_id"]
    assert run["config_hash"] and len(run["config_hash"]) == 64

    res = approve(client, lin["approval_id"]).json()
    assert res["status"] == "executed"
    done = client.get(f"/api/pipelines/runs/{run['id']}").json()
    assert done["status"] == "succeeded"
    assert steps(done)["request_promotion"]["status"] == "succeeded"
    models = client.get("/api/registry/models").json()
    assert models[0]["champion"] == 1


def test_gate_failure_stops_pipeline_without_retry(client):
    df = mixed_frame(240)
    df["label"] = np.random.default_rng(3).choice(["churn", "stay"], len(df))
    ds = upload(client, to_csv_bytes(df), name="noise")
    run = trigger(client, {"dataset_version_id": ds["versions"][0]["id"], "max_retries": 2})
    s = steps(run)
    assert run["status"] == "failed" and "min_f1" in run["error"]
    assert s["quality_gate"]["status"] == "failed" and s["quality_gate"]["attempts"] == 1
    assert s["deploy_staging"]["status"] == "skipped"


def test_transient_failure_is_retried_and_failed_run_resumes(client, churn_csv, monkeypatch):
    from evalforge.services import pipelines

    ds = upload(client, churn_csv)
    vid = ds["versions"][0]["id"]
    original = list(pipelines.PIPELINES["continuous_training"])
    calls = {"n": 0}

    def flaky_register(db, ctx, actor):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("registry temporarily unavailable")
        return original[3][1](db, ctx, actor)

    patched = [(n, flaky_register if n == "register" else f, d) for n, f, d in original]
    monkeypatch.setitem(pipelines.PIPELINES, "continuous_training", patched)
    monkeypatch.setattr(pipelines.time, "sleep", lambda s: None)
    run = trigger(client, {"dataset_version_id": vid, "request_promotion": False})
    assert run["status"] == "succeeded", run
    assert steps(run)["register"]["attempts"] == 2

    # permanent failure with no retries, then resume after the fault is fixed
    def broken(db, ctx, actor):
        raise ConnectionError("still down")

    patched = [(n, broken if n == "register" else f, d) for n, f, d in original]
    monkeypatch.setitem(pipelines.PIPELINES, "continuous_training", patched)
    failed = trigger(
        client,
        {
            "dataset_version_id": vid,
            "max_retries": 0,
            "request_promotion": False,
            "deploy_to_staging": False,
        },
    )
    assert failed["status"] == "failed"
    train_run = steps(failed)["train"]["output"]["run_id"]
    monkeypatch.setitem(pipelines.PIPELINES, "continuous_training", original)
    r = client.post(f"/api/pipelines/runs/{failed['id']}/resume")
    assert r.status_code == 202
    resumed = client.get(f"/api/pipelines/runs/{failed['id']}").json()
    assert resumed["status"] == "succeeded"
    assert steps(resumed)["train"]["output"]["run_id"] == train_run  # not retrained
    assert steps(resumed)["register"]["attempts"] == 2


def test_schedules_trigger_when_due(client, churn_csv):
    ds = upload(client, churn_csv)
    body = {
        "config": {
            "dataset_version_id": ds["versions"][0]["id"],
            "request_promotion": False,
            "deploy_to_staging": False,
        },
        "interval_minutes": 60,
    }
    assert client.post("/api/pipelines/schedules", json=body).status_code == 201
    first = client.post("/api/pipelines/schedules/tick").json()["triggered"]
    assert len(first) == 1
    assert client.post("/api/pipelines/schedules/tick").json()["triggered"] == []
    run = client.get(f"/api/pipelines/runs/{first[0]}").json()
    assert run["trigger"] == "schedule" and run["status"] == "succeeded"
    bad = client.post("/api/pipelines/schedules", json={**body, "interval_minutes": 1})
    assert bad.status_code == 422
