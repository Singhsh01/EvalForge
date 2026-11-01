"""Security regression tests from the audit (docs/audit/AUDIT-REPORT.md, SEC-*).

Each test reproduces an attack that worked against the pre-audit code.
"""

from __future__ import annotations

import re
import statistics
import time

import pytest

from .conftest import as_user
from .test_ops import approve, train, upload

PUBLIC = {("GET", "/api/health"), ("GET", "/api/health/ready"), ("GET", "/api/meta")}


def test_every_operation_requires_authentication(client):
    """Black-box sweep of the OpenAPI schema: only documented public routes answer anonymously."""
    spec = client.get("/openapi.json").json()
    leaks = []
    for path, ops in spec["paths"].items():
        url = re.sub(r"\{[^}]+\}", "x0", path)
        for m in ops:
            if (m.upper(), path) in PUBLIC or path == "/api/auth/login":
                continue
            r = client.request(m.upper(), url, json={}, headers={"Authorization": ""})
            if r.status_code != 401:
                leaks.append((m.upper(), path, r.status_code))
    assert not leaks, leaks
    assert sum(len(v) for v in spec["paths"].values()) >= 80


# POST operations a viewer may call by design: none of them changes models, data or approvals.
VIEWER_POSTS = {
    "/api/auth/tokens/revoke",  # revoke one of your own tokens
    "/api/serve/{model_name}/predict",  # inference is read-only (and rate limited per user)
    "/api/copilot/sessions",  # read-only investigations; action tools need engineer (SEC-06)
    "/api/copilot/sessions/{sid}/cancel",  # only cancels a session
}


def test_viewer_cannot_call_any_write_operation(client):
    spec = client.get("/openapi.json").json()
    viewer = as_user(client, "viewer")
    allowed = []
    for path, ops in spec["paths"].items():
        url = re.sub(r"\{[^}]+\}", "x0", path)
        for m in ops:
            if (
                m.upper() == "GET"
                or path in ("/api/auth/login", "/api/auth/logout")
                or path in VIEWER_POSTS
            ):
                continue
            r = client.request(m.upper(), url, json={}, headers=viewer)
            if r.status_code not in (401, 403):
                allowed.append((m.upper(), path, r.status_code))
    assert not allowed, allowed


def test_login_rate_limit_cannot_be_bypassed_by_rotating_authorization_header(client, monkeypatch):
    """SEC-01: the limiter keyed on the raw Authorization header, so a fresh fake header per
    request created a fresh bucket and allowed unlimited password guessing."""
    from evalforge.config import get_settings

    monkeypatch.setattr(get_settings(), "rate_limit_login_per_min", 3)
    codes = [
        client.post(
            "/api/auth/login",
            json={"username": "admin", "password": f"guess-{i}"},
            headers={"Authorization": f"Bearer forged-{i}"},
        ).status_code
        for i in range(6)
    ]
    assert codes.count(429) >= 3, codes


def test_write_rate_limit_cannot_be_bypassed_with_invalid_tokens(client, monkeypatch):
    from evalforge.config import get_settings

    monkeypatch.setattr(get_settings(), "rate_limit_login_per_min", 1)  # auth failures: 6/min/IP
    codes = [
        client.post("/api/auth/logout", headers={"Authorization": f"Bearer junk-{i}"}).status_code
        for i in range(9)
    ]
    assert 429 in codes, codes


def test_login_timing_does_not_reveal_which_usernames_exist(client):
    """SEC-02: unknown usernames returned without running PBKDF2 (~100x faster)."""

    def t(user: str) -> float:
        samples = []
        for _ in range(5):
            t0 = time.perf_counter()
            client.post("/api/auth/login", json={"username": user, "password": "wrong-password"})
            samples.append(time.perf_counter() - t0)
        return statistics.median(samples)

    known, unknown = t("admin"), t("no-such-user-xyz")
    assert unknown > 0.5 * known, (known, unknown)


def test_meta_does_not_expose_credentials_in_mlflow_uri(client, monkeypatch):
    """SEC-03: /api/meta (public) echoed the configured MLflow URI verbatim."""
    from evalforge.config import get_settings

    monkeypatch.setattr(get_settings(), "mlflow_tracking_uri", "http://svc:s3cretpw@mlflow:5000")
    body = client.get("/api/meta", headers={"Authorization": ""}).text
    assert "s3cretpw" not in body


def _registered_version(client, churn_csv):
    ds = upload(client, churn_csv)
    run = train(client, ds["versions"][0]["id"], "logistic_regression")
    eng = as_user(client, "engineer")
    mv = client.post("/api/registry/versions", json={"run_id": run}, headers=eng).json()
    return mv, eng


def test_concurrent_approvals_execute_once(client, churn_csv):
    """SEC-04: two approvers deciding the same pending approval both ran its executor."""
    from evalforge.db import session_scope
    from evalforge.models import Approval, AuditEvent
    from evalforge.services.auth import Principal
    from evalforge.services.governance import GovernanceError, decide

    mv, eng = _registered_version(client, churn_csv)
    appr = client.post(f"/api/registry/versions/{mv['id']}/request-promotion", headers=eng).json()

    from sqlalchemy.orm import Session

    from evalforge.db import get_engine

    s1, s2 = Session(get_engine()), Session(get_engine())
    a1, a2 = s1.get(Approval, appr["id"]), s2.get(Approval, appr["id"])  # both read "pending"
    p1 = Principal("u1", "approver", "approver")
    p2 = Principal("u2", "admin", "admin")
    decide(s1, a1, approve=True, actor=p1, reason="first")
    s1.commit()
    with pytest.raises(GovernanceError):
        decide(s2, a2, approve=True, actor=p2, reason="second")
    s2.rollback()
    s1.close(), s2.close()
    with session_scope() as db:
        executed = db.query(AuditEvent).filter_by(action="approval.executed", entity_id=appr["id"])
        assert executed.count() == 1


def _artifact_dir(mv) -> str:
    import mlflow

    return mlflow.artifacts.download_artifacts(mv["source_uri"])


def test_tampered_model_artifact_is_refused(client, churn_csv, tmp_path):
    """SEC-05: swapping the registered artifact for a pickle with a malicious __reduce__
    executed attacker code when the platform loaded the model."""
    import os
    import pickle
    from pathlib import Path

    import yaml

    from evalforge.services import model_cache

    mv, eng = _registered_version(client, churn_csv)
    src = Path(_artifact_dir(mv))
    # The local file store holds the canonical copy; tamper with it in place.
    store = next(Path(os.environ["EVALFORGE_DATA_DIR"]).rglob(f"*/{src.parent.name}/artifacts"))
    marker = tmp_path / "pwned"

    class Evil:
        def __reduce__(self):
            return (os.system, (f"touch {marker}",))

    (store / "model.pkl").write_bytes(pickle.dumps(Evil()))
    meta = yaml.safe_load((store / "MLmodel").read_text())
    meta["flavors"]["sklearn"].update(
        pickled_model="model.pkl", serialization_format="cloudpickle", skops_trusted_types=None
    )
    meta["flavors"]["python_function"]["model_path"] = "model.pkl"
    (store / "MLmodel").write_text(yaml.safe_dump(meta))
    model_cache.clear()

    r = client.post(
        "/api/deployments",
        json={"model_version_id": mv["id"], "environment": "staging"},
        headers=eng,
    )
    assert not marker.exists(), "attacker code executed during model load"
    assert r.status_code != 201 or not r.json()["deployment"]["health"]["healthy"]


def test_model_artifact_with_extra_trusted_types_is_refused(client, churn_csv):
    """A skops artifact whose MLmodel trusts types outside the platform allowlist."""
    import os
    from pathlib import Path

    import yaml

    from evalforge.services import model_cache

    mv, eng = _registered_version(client, churn_csv)
    src = Path(_artifact_dir(mv))
    store = next(Path(os.environ["EVALFORGE_DATA_DIR"]).rglob(f"*/{src.parent.name}/artifacts"))
    meta = yaml.safe_load((store / "MLmodel").read_text())
    meta["flavors"]["sklearn"]["skops_trusted_types"] = ["builtins.eval", "os.system"]
    (store / "MLmodel").write_text(yaml.safe_dump(meta))
    (store / "model.skops").write_bytes((store / "model.skops").read_bytes() + b"\0")  # modified
    model_cache.clear()
    r = client.post(
        "/api/deployments",
        json={"model_version_id": mv["id"], "environment": "staging"},
        headers=eng,
    )
    assert r.status_code != 201 or not r.json()["deployment"]["health"]["healthy"]


def test_separation_of_duties_holds_for_admin(client, churn_csv):
    mv, _ = _registered_version(client, churn_csv)
    admin = as_user(client, "admin")
    appr = client.post(f"/api/registry/versions/{mv['id']}/request-promotion", headers=admin).json()
    assert approve(client, appr["id"], user="admin").status_code == 403


def _action_runner(role: str):
    from evalforge.copilot.agent import DEFAULT_CONFIG, ToolRunner
    from evalforge.copilot.tools import TOOLS

    cfg = {**DEFAULT_CONFIG, "allow_actions": True, "tools_enabled": list(TOOLS)}
    from evalforge.db import session_scope

    return session_scope, lambda db: ToolRunner(db, cfg, None, actor=f"a-{role}", actor_role=role)


def test_copilot_action_tools_need_engineer_role(client):
    """SEC-06: a viewer could trigger approval requests filed as an engineer."""
    scope, make = _action_runner("viewer")
    with scope() as db:
        r = make(db)
        assert r.call("request_evaluation", suite_id="x", subject="{}") is None
        assert "needs the engineer role" in r.calls[-1]["error"]


def test_copilot_cannot_spoof_requester(client, monkeypatch):
    """SEC-06: requested_by was a model-controlled argument, so an injected tool call could
    file a request under another user's name and defeat separation of duties."""
    import dataclasses

    from evalforge.copilot import tools as T

    seen = {}

    def fake(db, suite_id, subject, requested_by="copilot"):
        seen["requested_by"] = requested_by
        return {"evidence": []}

    monkeypatch.setitem(
        T.TOOLS, "request_evaluation", dataclasses.replace(T.TOOLS["request_evaluation"], fn=fake)
    )
    scope, make = _action_runner("engineer")
    with scope() as db:
        make(db).call("request_evaluation", suite_id="x", subject="{}", requested_by="approver")
    assert seen["requested_by"] == "a-engineer"
