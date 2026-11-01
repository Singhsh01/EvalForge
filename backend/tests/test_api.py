from __future__ import annotations

import pandas as pd

from .conftest import mixed_frame, to_csv_bytes


def upload(client, data: bytes, name="churn", target="label", filename="churn.csv"):
    return client.post(
        "/api/datasets",
        data={"name": name, "target_column": target},
        files={"file": (filename, data, "text/csv")},
    )


def test_health_and_meta(client):
    assert client.get("/api/health").json()["status"] == "ok"
    assert client.get("/api/health/ready").json()["checks"] == {"database": "ok", "mlflow": "ok"}
    algos = {a["key"] for a in client.get("/api/meta").json()["algorithms"]}
    assert algos == {"logistic_regression", "random_forest", "gradient_boosting"}


def test_overview_is_empty_without_data(client):
    o = client.get("/api/overview").json()
    assert o["counts"]["runs"] == 0 and o["best_by_dataset"] == [] and o["recent_runs"] == []


def test_upload_profile_and_versioning(client, churn_csv):
    r = upload(client, churn_csv)
    assert r.status_code == 201, r.text
    ds = r.json()
    v1 = ds["versions"][0]
    assert v1["version"] == 1 and v1["status"] == "validated" and v1["n_rows"] == 240
    detail = client.get(f"/api/datasets/{ds['id']}/versions/1").json()
    assert detail["split"]["n_test"] == 48
    assert "schema" in detail and detail["profile"]["columns"]["plan"]["unique"] == 3

    # identical content is deduplicated
    dup = client.post(
        f"/api/datasets/{ds['id']}/versions", files={"file": ("again.csv", churn_csv, "text/csv")}
    )
    assert dup.status_code == 409

    # new content -> v2; JSON accepted
    df2 = mixed_frame(seed=7)
    js = df2.to_json(orient="records").encode()
    v2 = client.post(
        f"/api/datasets/{ds['id']}/versions",
        files={"file": ("v2.json", js, "application/json")},
        data={"note": "fresh"},
    )
    assert v2.status_code == 201, v2.text
    assert v2.json()["version"] == 2

    # schema break rejected
    broken = to_csv_bytes(df2.drop(columns=["income"]))
    r = client.post(
        f"/api/datasets/{ds['id']}/versions", files={"file": ("bad.csv", broken, "text/csv")}
    )
    assert r.status_code == 422 and "income" in r.json()["detail"]

    prev = client.get(f"/api/datasets/{ds['id']}/versions/1/preview?limit=5").json()
    assert len(prev["rows"]) == 5


def test_rejects_bad_uploads(client):
    assert upload(client, b"", filename="x.csv").status_code == 422
    assert upload(client, b"MZ\x90\x00", filename="x.exe").status_code == 422
    assert upload(client, b'{"not": "rows"}', filename="x.json").status_code == 422
    assert (
        client.post(
            "/api/datasets",
            data={"name": "../etc", "target_column": "y"},
            files={"file": ("a.csv", b"a,y\n1,2", "text/csv")},
        ).status_code
        == 422
    )


def test_rejected_version_cannot_be_trained(client):
    tiny = to_csv_bytes(pd.DataFrame({"x": range(10), "y": ["a", "b"] * 5}))
    ds = upload(client, tiny, name="tiny", target="y").json()
    v = ds["versions"][0]
    assert v["status"] == "rejected" and v["checks_failed"] >= 1
    r = client.post(
        "/api/runs", json={"dataset_version_id": v["id"], "algorithm": "logistic_regression"}
    )
    assert r.status_code == 422


def test_end_to_end_train_track_evaluate_compare(client, churn_csv):
    import mlflow

    ds = upload(client, churn_csv).json()
    vid = ds["versions"][0]["id"]

    bad = client.post(
        "/api/runs",
        json={
            "dataset_version_id": vid,
            "algorithm": "random_forest",
            "hyperparameters": {"n_estimators": 0},
        },
    )
    assert bad.status_code == 422

    run_ids = []
    for algo, hp in [
        ("logistic_regression", {"C": 0.5}),
        ("random_forest", {"n_estimators": 50, "max_depth": 6}),
        ("gradient_boosting", {"max_iter": 60}),
    ]:
        r = client.post(
            "/api/runs",
            json={
                "dataset_version_id": vid,
                "algorithm": algo,
                "hyperparameters": hp,
                "cv_folds": 3,
            },
        )
        assert r.status_code == 202, r.text
        run_ids.append(r.json()["run"]["id"])

    for rid in run_ids:
        run = client.get(f"/api/runs/{rid}").json()
        assert run["status"] == "succeeded", run["error"] or run["job"]["logs"]
        m = run["metrics"]
        for k in (
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "cv_f1_macro_mean",
            "latency_single_p50_ms",
            "training_seconds",
        ):
            assert k in m
        assert m["f1"] > 0.6
        assert run["feature_columns"] and "customer_id" not in run["feature_columns"]
        assert {"column": "customer_id", "reason": "identifier-like"} in run["dropped_columns"]
        assert run["report"]["feature_importance"]
        assert run["job"]["progress"] == 1.0

        # MLflow is the system of record: verify the run really exists there
        mr = mlflow.get_run(run["mlflow_run_id"])
        assert mr.data.tags["evalforge.run_id"] == rid
        assert mr.data.tags["evalforge.dataset_sha256"] == ds["versions"][0]["sha256"]
        assert abs(mr.data.metrics["f1"] - m["f1"]) < 1e-9
        assert mr.inputs.dataset_inputs[0].dataset.name == "churn@v1"

    cmp = client.get(f"/api/runs/compare?ids={','.join(run_ids)}").json()
    assert cmp["same_holdout"] and not cmp["warnings"]
    assert "algorithm" in cmp["differing_params"]
    assert set(cmp["best"]["f1"]) <= set(run_ids)
    assert set(cmp["best"]["log_loss"]) <= set(run_ids)

    # evaluate run 1 against a new data version via the model loaded back from MLflow
    v2 = client.post(
        f"/api/datasets/{ds['id']}/versions",
        files={"file": ("v2.csv", to_csv_bytes(mixed_frame(seed=3)), "text/csv")},
    ).json()
    ev = client.post(f"/api/runs/{run_ids[0]}/evaluations", json={"dataset_version_id": v2["id"]})
    assert ev.status_code == 202, ev.text
    evs = client.get(f"/api/runs/{run_ids[0]}/evaluations").json()
    assert [e["split"] for e in evs] == ["holdout", "full"]
    assert evs[1]["status"] == "succeeded" and evs[1]["metrics"]["f1"] > 0.5

    o = client.get("/api/overview").json()
    assert o["counts"]["runs"] == 3 and o["best_by_dataset"][0]["n_runs"] == 3
    assert any(a["action"] == "run.completed" for a in o["recent_activity"])


def test_failed_job_is_recorded_not_raised(client, churn_csv, monkeypatch):
    from evalforge.services import training

    ds = upload(client, churn_csv).json()

    def boom(job_id):
        raise RuntimeError("simulated worker crash")

    monkeypatch.setattr(training, "run_training", boom)
    r = client.post(
        "/api/runs",
        json={"dataset_version_id": ds["versions"][0]["id"], "algorithm": "logistic_regression"},
    )
    run = client.get(f"/api/runs/{r.json()['run']['id']}").json()
    assert run["status"] == "failed" and "simulated worker crash" in run["error"]
    assert run["job"]["status"] == "failed" and run["job"]["logs"][-1]["traceback"]


def test_demo_dataset(client):
    r = client.post("/api/demo/breast-cancer")
    assert r.status_code == 201 and r.json()["created"]
    ds = client.get(f"/api/datasets/{r.json()['dataset_id']}").json()
    v = ds["versions"][0]
    assert v["n_rows"] == 569 and v["n_columns"] == 31 and v["status"] == "validated"
    assert client.post("/api/demo/breast-cancer").json()["created"] is False


def _utc(ts: str) -> bool:
    return ts.endswith(("Z", "+00:00"))


def test_timestamps_are_timezone_aware(client, churn_csv):
    ds = upload(client, churn_csv).json()
    assert _utc(ds["created_at"]) and _utc(ds["versions"][0]["created_at"])
    o = client.get("/api/overview").json()
    assert o["recent_activity"] and all(_utc(a["created_at"]) for a in o["recent_activity"])
