"""Resilience and fault-injection tests from the audit (RES-*)."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.orm import Session

from .conftest import as_user, mixed_frame
from .test_ops import approve, train, upload


def _two_versions(client, churn_csv):
    ds = upload(client, churn_csv)
    vid = ds["versions"][0]["id"]
    eng = as_user(client, "engineer")
    mvs = []
    for hp in ({}, {"C": 0.8}):
        run = train(client, vid, "logistic_regression", hp)
        mvs.append(client.post("/api/registry/versions", json={"run_id": run}, headers=eng).json())
    return mvs, eng


def _rows(n=20, seed=3):
    f = mixed_frame(n, seed=seed).drop(columns=["label"])
    return f.astype(object).where(f.notna(), None).to_dict(orient="records")


def _active(name, env, role="stable"):
    from evalforge.db import session_scope
    from evalforge.models import Deployment

    with session_scope() as db:
        return (
            db.query(Deployment)
            .filter_by(model_name=name, environment=env, role=role, status="active")
            .count()
        )


def _tamper(mv):
    import os
    from pathlib import Path

    import mlflow

    from evalforge.services import model_cache

    src = Path(mlflow.artifacts.download_artifacts(mv["source_uri"]))
    store = next(Path(os.environ["EVALFORGE_DATA_DIR"]).rglob(f"*/{src.parent.name}/artifacts"))
    (store / "model.skops").write_bytes((store / "model.skops").read_bytes()[:-50])
    model_cache.clear()


def test_stale_running_job_is_reaped_for_any_executor(client, churn_csv):
    """RES-01: a job whose worker died stayed 'running' forever (RQ / crashed process)."""
    from evalforge.config import get_settings
    from evalforge.db import session_scope
    from evalforge.models import Job, JobKind, JobStatus, TrainingRun
    from evalforge.services import jobs

    ds = upload(client, churn_csv)
    run_id = train(client, ds["versions"][0]["id"], "logistic_regression")
    long_ago = datetime.now(UTC) - timedelta(seconds=get_settings().job_timeout_seconds + 3600)
    with session_scope() as db:
        tr = db.get(TrainingRun, run_id)
        job = Job(kind=JobKind.train, payload={}, status=JobStatus.running, started_at=long_ago)
        db.add(job)
        db.flush()
        tr.job_id, tr.status, jid = job.id, JobStatus.running, job.id
        fresh = Job(
            kind=JobKind.train, payload={}, status=JobStatus.running, started_at=datetime.now(UTC)
        )
        db.add(fresh)
        db.flush()
        fid = fresh.id
    assert jobs.reap_stale() == 1
    with session_scope() as db:
        assert db.get(Job, jid).status == JobStatus.failed
        assert "worker lost" in db.get(Job, jid).error
        assert db.get(TrainingRun, run_id).status == JobStatus.failed
        assert db.get(Job, fid).status == JobStatus.running  # still within its timeout


def test_restart_marks_interrupted_thread_jobs_failed(client, monkeypatch):
    from evalforge.config import get_settings
    from evalforge.db import session_scope
    from evalforge.models import Job, JobKind, JobStatus
    from evalforge.services import jobs

    with session_scope() as db:
        j = Job(kind=JobKind.train, payload={}, status=JobStatus.running)
        db.add(j)
        db.flush()
        jid = j.id
    monkeypatch.setattr(get_settings(), "executor", "thread")
    assert jobs.recover_orphans() >= 1
    with session_scope() as db:
        assert db.get(Job, jid).status == JobStatus.failed


def test_unhealthy_deploy_is_recorded_and_current_stable_keeps_serving(client, churn_csv):
    (mv1, mv2), eng = _two_versions(client, churn_csv)
    ok = client.post(
        "/api/deployments",
        json={"model_version_id": mv1["id"], "environment": "staging"},
        headers=eng,
    )
    assert ok.status_code == 201
    _tamper(mv2)
    bad = client.post(
        "/api/deployments",
        json={"model_version_id": mv2["id"], "environment": "staging"},
        headers=eng,
    )
    assert bad.status_code in (409, 422)
    deps = client.get("/api/deployments").json()
    rows = deps if isinstance(deps, list) else deps.get("deployments", deps.get("history", []))
    assert any(d.get("status") == "failed" for d in rows), rows
    assert _active("churn", "staging") == 1
    r = client.post("/api/serve/churn/predict?environment=staging", json={"instances": _rows()})
    assert r.status_code == 200 and r.json()["version"] == 1


def test_rollback_refuses_broken_target_and_keeps_current(client, churn_csv):
    (mv1, mv2), eng = _two_versions(client, churn_csv)
    for mv in (mv1, mv2):
        assert (
            client.post(
                "/api/deployments",
                json={"model_version_id": mv["id"], "environment": "staging"},
                headers=eng,
            ).status_code
            == 201
        )
    _tamper(mv1)
    r = client.post(
        "/api/deployments/rollback",
        json={"model_name": "churn", "environment": "staging"},
        headers=eng,
    )
    assert r.status_code in (409, 422), r.text
    assert _active("churn", "staging") == 1
    out = client.post(
        "/api/serve/churn/predict?environment=staging", json={"instances": _rows()}
    ).json()
    assert out["version"] == 2


def test_canary_promotion_rechecks_health(client, churn_csv):
    """RES-02: promotion reused the health result recorded when the canary was deployed."""
    from evalforge.db import session_scope
    from evalforge.models import ModelVersion
    from evalforge.services import serving
    from evalforge.services.auth import Principal
    from evalforge.services.governance import GovernanceError

    (mv1, mv2), _ = _two_versions(client, churn_csv)
    actor = Principal("u", "engineer", "engineer")
    with session_scope() as db:
        serving.deploy(db, db.get(ModelVersion, mv1["id"]), "staging", 100, actor)
    with session_scope() as db:
        serving.deploy(db, db.get(ModelVersion, mv2["id"]), "staging", 20, actor)
    _tamper(mv2)
    with session_scope() as db, pytest.raises(GovernanceError):
        serving.promote_canary(db, "churn", "staging", actor)
    assert _active("churn", "staging", "stable") == 1


def test_concurrent_deploys_leave_exactly_one_active_stable(client, churn_csv):
    """RES-03: two deploys racing for the same model/environment could both commit an
    active stable deployment (no database constraint prevented it)."""
    from evalforge.db import get_engine
    from evalforge.models import ModelVersion
    from evalforge.services import serving
    from evalforge.services.auth import Principal

    (mv1, mv2), _ = _two_versions(client, churn_csv)
    actor = Principal("u", "engineer", "engineer")
    s0 = Session(get_engine())
    serving.deploy(s0, s0.get(ModelVersion, mv1["id"]), "staging", 100, actor)
    s0.commit()
    s0.close()

    s1, s2 = Session(get_engine()), Session(get_engine())
    errors: list[Exception] = []
    # Both read the current stable before either commits.
    serving.stable_and_canary(s1, "churn", "staging")
    serving.stable_and_canary(s2, "churn", "staging")
    serving.deploy(s1, s1.get(ModelVersion, mv2["id"]), "staging", 100, actor, note="A")

    def second():
        try:
            serving.deploy(s2, s2.get(ModelVersion, mv2["id"]), "staging", 100, actor, note="B")
            s2.commit()
        except Exception as exc:  # expected: unique violation / lock conflict
            errors.append(exc)
            print("losing deploy:", type(exc).__name__, str(exc)[:200])
            s2.rollback()

    t = threading.Thread(target=second)
    t.start()
    t.join(timeout=2)
    s1.commit()
    t.join(timeout=60)
    s1.close(), s2.close()
    assert _active("churn", "staging") == 1
    assert errors, "the losing deploy should fail, not silently create a second stable"


def test_concurrent_predictions_are_all_served_and_logged(client, churn_csv):
    from evalforge.db import session_scope
    from evalforge.models import PredictionLog

    (mv1, _), eng = _two_versions(client, churn_csv)
    client.post(
        "/api/deployments",
        json={"model_version_id": mv1["id"], "environment": "staging"},
        headers=eng,
    )
    rows = _rows(20)[:1]
    with session_scope() as db:
        before = db.query(PredictionLog).count()

    def call(i):
        return client.post(
            "/api/serve/churn/predict?environment=staging",
            json={"instances": rows},
            headers={"x-request-id": f"conc-{i}"},
        ).status_code

    with ThreadPoolExecutor(8) as pool:
        codes = list(pool.map(call, range(40)))
    assert codes.count(200) == 40, codes
    with session_scope() as db:
        assert db.query(PredictionLog).count() - before == 40


def test_approval_executor_failure_leaves_no_partial_state(client, churn_csv):
    """A production deploy approved while the artifact is broken: the approval is marked
    failed and nothing is activated."""
    (mv1, _), eng = _two_versions(client, churn_csv)
    promo = client.post(f"/api/registry/versions/{mv1['id']}/request-promotion", headers=eng)
    assert approve(client, promo.json()["id"]).json()["status"] == "executed"
    pr = client.post(
        "/api/deployments",
        json={"model_version_id": mv1["id"], "environment": "production"},
        headers=eng,
    ).json()
    _tamper(mv1)
    done = approve(client, pr["approval"]["id"]).json()
    assert done["status"] == "failed" and "health" in done["result"]["error"].lower()
    assert _active("churn", "production") == 0


def test_single_pass_serving_matches_pipeline_predict(client, churn_csv):
    """PERF-01 changed serving to derive labels from predict_proba (one pipeline pass);
    the labels must equal Pipeline.predict exactly."""
    import pandas as pd

    from evalforge.db import session_scope
    from evalforge.models import ModelVersion
    from evalforge.services.model_cache import load_cached

    (mv1, _), eng = _two_versions(client, churn_csv)
    client.post(
        "/api/deployments",
        json={"model_version_id": mv1["id"], "environment": "staging"},
        headers=eng,
    )
    rows = _rows(60, seed=21)
    out = client.post("/api/serve/churn/predict?environment=staging", json={"instances": rows})
    served = [p["label"] for p in out.json()["predictions"]]
    with session_scope() as db:
        mv = db.get(ModelVersion, mv1["id"])
        feats = mv.run.feature_columns
        model = load_cached(mv.source_uri, mv.artifact_sha256)
    X = pd.DataFrame([{c: r.get(c) for c in feats} for r in rows], columns=feats)
    for c in X.columns:
        if X[c].map(lambda v: isinstance(v, int | float) or v is None).all():
            X[c] = pd.to_numeric(X[c], errors="coerce")
    assert served == [str(p) for p in model.predict(X)]


def test_slow_model_call_does_not_hold_the_database_write_lock(client):
    """RES-05: an agent evaluation kept a write transaction open across slow model calls
    (local LLM: seconds per call), so every other write - including the token bookkeeping
    of a plain GET - timed out with 'database is locked' and returned 500. Found by the
    first real Ollama benchmark run."""
    import json
    import threading
    from datetime import UTC, datetime

    from sqlalchemy import create_engine, text

    from evalforge.config import get_settings
    from evalforge.db import get_engine, session_scope
    from evalforge.llm.providers import FixtureProvider
    from evalforge.models import EvalSuite
    from evalforge.services import evals, jobs

    entered, release = threading.Event(), threading.Event()

    def slow(system, user):
        entered.set()
        release.wait(20)
        return json.dumps({"action": "final", "findings": []})

    FixtureProvider.responders["slow"] = slow
    try:
        cfg = client.post(
            "/api/copilot/configs", json={"config": {"planner": "llm:fixture:slow"}}
        ).json()
        suite = client.post(
            "/api/evals/suites",
            json={
                "name": "slow-agent",
                "kind": "agent",
                "cases": [{"id": "c1", "question": "overview"}],
                "evaluators": [{"type": "task_completion"}],
            },
        ).json()
        with session_scope() as db:
            run = evals.create_run(
                db,
                db.get(EvalSuite, suite["id"]),
                {"type": "agent", "agent_config_id": cfg["id"]},
                "engineer",
            )
            job_id = run.job_id
        worker = threading.Thread(target=jobs.execute, args=(job_id,))
        worker.start()
        assert entered.wait(20), "planner was never called"
        # While the model is "thinking", an unrelated write must go through promptly.
        url = get_settings().resolved_database_url
        probe = (
            create_engine(url, connect_args={"timeout": 2})
            if url.startswith("sqlite")
            else get_engine()
        )
        with probe.begin() as c:
            c.execute(text("UPDATE api_tokens SET last_used_at = :t"), {"t": datetime.now(UTC)})
        release.set()
        worker.join(30)
    finally:
        release.set()
        FixtureProvider.responders.clear()
