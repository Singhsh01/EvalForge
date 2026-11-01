from __future__ import annotations

from .conftest import as_user, mixed_frame, to_csv_bytes


def upload(client, data: bytes, name="churn"):
    r = client.post(
        "/api/datasets",
        data={"name": name, "target_column": "label"},
        files={"file": ("d.csv", data, "text/csv")},
    )
    assert r.status_code == 201, r.text
    return r.json()


def train(client, version_id: str, algo: str, hp: dict | None = None) -> str:
    r = client.post(
        "/api/runs",
        json={"dataset_version_id": version_id, "algorithm": algo, "hyperparameters": hp or {}},
    )
    assert r.status_code == 202, r.text
    run = client.get(f"/api/runs/{r.json()['run']['id']}").json()
    assert run["status"] == "succeeded", run["error"]
    return run["id"]


def approve(client, approval_id: str, user="approver", ok=True):
    return client.post(
        f"/api/approvals/{approval_id}/decision",
        json={"approve": ok, "reason": "reviewed metrics and gate"},
        headers=as_user(client, user),
    )


# --------------------------------------------------------------------------- auth


def test_auth_required_and_roles(client, churn_csv):
    anon = {"Authorization": ""}
    assert client.get("/api/health").status_code == 200  # public
    assert client.get("/api/datasets", headers=anon).status_code == 401
    assert client.get("/api/datasets", headers={"Authorization": "Bearer nope"}).status_code == 401
    viewer = as_user(client, "viewer")
    assert client.get("/api/datasets", headers=viewer).status_code == 200
    r = client.post(
        "/api/datasets",
        data={"name": "x1", "target_column": "label"},
        files={"file": ("d.csv", churn_csv, "text/csv")},
        headers=viewer,
    )
    assert r.status_code == 403
    assert (
        client.post("/api/auth/login", json={"username": "admin", "password": "wrong"}).status_code
        == 401
    )
    assert client.get("/api/auth/me", headers=viewer).json()["role"] == "viewer"


def test_logout_revokes_token(client):
    h = as_user(client, "engineer")
    assert client.post("/api/auth/logout", headers=h).status_code == 200
    assert client.get("/api/auth/me", headers=h).status_code == 401


def test_login_rate_limited(client, monkeypatch):
    from evalforge.config import get_settings
    from evalforge.services.auth import limiter

    monkeypatch.setattr(get_settings(), "rate_limit_login_per_min", 3)
    limiter.reset()  # the fixture's own admin login counts against the same per-IP budget
    codes = [
        client.post("/api/auth/login", json={"username": "x", "password": "y"}).status_code
        for _ in range(5)
    ]
    assert codes[:3] == [401, 401, 401] and codes[3:] == [429, 429]


def test_admin_user_management(client):
    r = client.post(
        "/api/auth/users",
        json={"username": "alice", "password": "s3cure-passw0rd", "role": "engineer"},
    )
    assert r.status_code == 201
    assert (
        client.post(
            "/api/auth/users",
            json={"username": "bob", "password": "s3cure-passw0rd", "role": "engineer"},
            headers=as_user(client, "engineer"),
        ).status_code
        == 403
    )


# --------------------------------------------------------------------------- lifecycle


def test_full_lifecycle_register_gate_approve_deploy_canary_rollback(client, churn_csv):
    import mlflow

    ds = upload(client, churn_csv)
    vid = ds["versions"][0]["id"]
    run1 = train(client, vid, "logistic_regression")
    run2 = train(client, vid, "logistic_regression", {"C": 0.8})
    eng = as_user(client, "engineer")

    # register + gate
    mv1 = client.post("/api/registry/versions", json={"run_id": run1}, headers=eng).json()
    assert mv1["version"] == 1 and mv1["gate"]["passed"], mv1["gate"]
    assert mv1["gate"]["champion_comparison"] is None
    assert (
        client.post("/api/registry/versions", json={"run_id": run1}, headers=eng).status_code == 409
    )  # already registered

    # promotion requires an approver who is not the requester
    appr = client.post(f"/api/registry/versions/{mv1['id']}/request-promotion", headers=eng).json()
    assert appr["status"] == "pending"
    assert approve(client, appr["id"], user="engineer").status_code == 403  # wrong role
    done = approve(client, appr["id"]).json()
    assert done["status"] == "executed", done
    assert str(mlflow.MlflowClient().get_model_version_by_alias("churn", "champion").version) == "1"
    card = client.get(f"/api/registry/versions/{mv1['id']}/card").json()
    assert card["status"] == "champion" and card["training_data"]["sha256"]

    # staging deploy is immediate; production deploy needs approval
    st = client.post(
        "/api/deployments",
        json={"model_version_id": mv1["id"], "environment": "staging"},
        headers=eng,
    )
    assert st.status_code == 201 and st.json()["deployment"]["health"]["healthy"]
    pr = client.post(
        "/api/deployments",
        json={"model_version_id": mv1["id"], "environment": "production"},
        headers=eng,
    ).json()
    assert "approval" in pr
    assert client.post("/api/serve/churn/predict", json={"instances": [{}]}).status_code == 404
    assert approve(client, pr["approval"]["id"]).json()["status"] == "executed"

    frame = mixed_frame(30, seed=9).drop(columns=["label"])
    rows = frame.astype(object).where(frame.notna(), None).to_dict(orient="records")
    out = client.post("/api/serve/churn/predict", json={"instances": rows}).json()
    assert out["version"] == 1 and len(out["predictions"]) == 30
    assert {p["label"] for p in out["predictions"]} <= {"churn", "stay"}
    bad = client.post("/api/serve/churn/predict", json={"instances": [{"age": 3}]})
    assert bad.status_code == 422 and "Missing required features" in bad.json()["detail"]
    fb = client.post(
        "/api/serve/feedback", json={"labels": {out["predictions"][0]["id"]: "stay"}}, headers=eng
    ).json()
    assert fb == {"updated": 1, "unknown": 0}

    # second version: gate compares against champion on the same holdout
    mv2 = client.post("/api/registry/versions", json={"run_id": run2}, headers=eng).json()
    cmp = mv2["gate"]["champion_comparison"]
    assert cmp["champion_version"] == 1 and "delta_f1" in cmp
    assert mv2["gate"]["passed"], mv2["gate"]
    a2 = client.post(f"/api/registry/versions/{mv2['id']}/request-promotion", headers=eng).json()
    assert approve(client, a2["id"]).json()["status"] == "executed"

    # canary at 30 %
    c = client.post(
        "/api/deployments",
        json={"model_version_id": mv2["id"], "environment": "production", "traffic_percent": 30},
        headers=eng,
    ).json()
    assert approve(client, c["approval"]["id"]).json()["status"] == "executed"
    versions = set()
    for i in range(40):
        r = client.post(
            "/api/serve/churn/predict",
            json={"instances": rows[:1]},
            headers={"x-request-id": f"req-{i}"},
        ).json()
        versions.add((r["version"], r["role"]))
    assert versions == {(1, "stable"), (2, "canary")}
    pinned = client.post("/api/serve/churn/predict?version=2", json={"instances": rows[:1]}).json()
    assert pinned["version"] == 2

    # promote canary, then roll back production to v1
    pc = client.post(
        "/api/deployments/promote-canary",
        json={"model_name": "churn", "environment": "production"},
        headers=eng,
    ).json()
    assert approve(client, pc["approval"]["id"]).json()["status"] == "executed"
    assert (
        client.post("/api/serve/churn/predict", json={"instances": rows[:1]}).json()["version"] == 2
    )
    rb = client.post(
        "/api/deployments/rollback",
        json={"model_name": "churn", "environment": "production"},
        headers=eng,
    ).json()
    res = approve(client, rb["approval"]["id"]).json()
    assert res["status"] == "executed" and res["result"]["version"] == 1
    assert (
        client.post("/api/serve/churn/predict", json={"instances": rows[:1]}).json()["version"] == 1
    )
    hist = client.get("/api/deployments?model_name=churn").json()
    assert {"rolled_back", "superseded", "active"} <= {d["status"] for d in hist}

    audit = client.get("/api/overview").json()["recent_activity"]
    assert any(a["action"].startswith("approval.") for a in audit)


def test_self_approval_blocked_and_rejection(client, churn_csv):
    ds = upload(client, churn_csv)
    run = train(client, ds["versions"][0]["id"], "logistic_regression")
    ap = as_user(client, "approver")
    mv = client.post("/api/registry/versions", json={"run_id": run}, headers=ap).json()
    a = client.post(f"/api/registry/versions/{mv['id']}/request-promotion", headers=ap).json()
    r = approve(client, a["id"], user="approver")
    assert r.status_code == 403 and "Separation of duties" in r.json()["detail"]
    rej = approve(client, a["id"], user="admin", ok=False).json()
    assert rej["status"] == "rejected"
    assert approve(client, a["id"], user="admin").status_code == 409  # already decided


def test_production_requires_approved_version_and_failed_health_is_recorded(
    client, churn_csv, monkeypatch
):
    from evalforge.services import serving

    ds = upload(client, churn_csv)
    run = train(client, ds["versions"][0]["id"], "logistic_regression")
    eng = as_user(client, "engineer")
    mv = client.post("/api/registry/versions", json={"run_id": run}, headers=eng).json()
    r = client.post(
        "/api/deployments",
        json={"model_version_id": mv["id"], "environment": "production"},
        headers=eng,
    )
    assert r.status_code == 409 and "approved" in r.json()["detail"]

    monkeypatch.setattr(
        serving,
        "health_check",
        lambda mv, n=25: {"healthy": False, "checks": {"model_loads": True, "latency_ok": False}},
    )
    r = client.post(
        "/api/deployments",
        json={"model_version_id": mv["id"], "environment": "staging"},
        headers=eng,
    )
    assert r.status_code == 409 and "health check failed" in r.json()["detail"]
    hist = client.get("/api/deployments").json()
    assert hist[0]["status"] == "failed"


def test_gate_blocks_weak_model(client):
    import numpy as np

    df = mixed_frame(240)
    df["label"] = np.random.default_rng(3).choice(["churn", "stay"], len(df))  # pure noise
    ds = upload(client, to_csv_bytes(df), name="noise")
    run = train(client, ds["versions"][0]["id"], "logistic_regression")
    mv = client.post("/api/registry/versions", json={"run_id": run}).json()
    assert not mv["gate"]["passed"]
    r = client.post(f"/api/registry/versions/{mv['id']}/request-promotion")
    assert r.status_code == 409 and "min_f1" in r.json()["detail"]


def test_gate_blocks_regression_against_champion(client, churn_csv):
    ds = upload(client, churn_csv)
    vid = ds["versions"][0]["id"]
    champ = client.post(
        "/api/registry/versions", json={"run_id": train(client, vid, "logistic_regression")}
    ).json()
    a = client.post(
        f"/api/registry/versions/{champ['id']}/request-promotion",
        headers=as_user(client, "engineer"),
    ).json()
    assert approve(client, a["id"]).json()["status"] == "executed"
    weak = client.post(
        "/api/registry/versions",
        json={"run_id": train(client, vid, "gradient_boosting", {"max_iter": 80})},
    ).json()
    check = next(c for c in weak["gate"]["checks"] if c["id"] == "no_regression_vs_champion")
    cmp = weak["gate"]["champion_comparison"]
    # Measured on this fixture: boosting trails the linear champion on the same holdout.
    assert cmp["delta_f1"] < -0.01 and not check["passed"] and not weak["gate"]["passed"]
