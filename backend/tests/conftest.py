from __future__ import annotations

import io
import os
from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"


@pytest.fixture()
def settings_env(tmp_path, monkeypatch) -> Iterator[None]:
    monkeypatch.setenv("EVALFORGE_DATA_DIR", str(tmp_path / "var"))
    monkeypatch.setenv("EVALFORGE_EXECUTOR", "sync")
    monkeypatch.setenv("EVALFORGE_ENV", "test")
    if url := os.environ.get("EVALFORGE_TEST_DATABASE_URL"):
        monkeypatch.setenv("EVALFORGE_DATABASE_URL", url)
    _clear()
    yield
    _clear()


def _clear() -> None:
    from evalforge import db
    from evalforge.config import get_settings
    from evalforge.services import jobs, model_cache, tracking
    from evalforge.services.auth import limiter

    get_settings.cache_clear()
    limiter.reset()
    model_cache.clear()
    db.reset_caches()
    tracking.configure.cache_clear()
    jobs._pool.cache_clear()


@pytest.fixture()
def client(settings_env) -> Iterator[TestClient]:
    from evalforge.main import create_app

    if os.environ.get("EVALFORGE_TEST_DATABASE_URL"):
        from evalforge.db import Base, get_engine

        Base.metadata.drop_all(get_engine())
        with get_engine().begin() as c:
            c.exec_driver_sql("DROP TABLE IF EXISTS alembic_version")
            for t in ("jobkind", "jobstatus", "tasktype", "versionstatus"):
                c.exec_driver_sql(f"DROP TYPE IF EXISTS {t}")
    with TestClient(create_app()) as c:
        c.headers["Authorization"] = f"Bearer {login(c, 'admin')}"
        yield c


DEV_PASSWORD = "evalforge-dev"


def login(c: TestClient, username: str) -> str:
    r = c.post("/api/auth/login", json={"username": username, "password": DEV_PASSWORD})
    assert r.status_code == 200, r.text
    return r.json()["token"]


def as_user(c: TestClient, username: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {login(c, username)}"}


def mixed_frame(n: int = 240, seed: int = 0) -> pd.DataFrame:
    """Small synthetic table with numeric, categorical and missing values and a learnable label."""
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 80, n).astype(float)
    income = rng.normal(50_000, 15_000, n)
    plan = rng.choice(["basic", "plus", "pro"], n)
    region = rng.choice(["north", "south", "east", "west"], n)
    score = (age - 50) / 15 + (income - 50_000) / 15_000 + (plan == "pro") * 1.2
    y = np.where(score + rng.normal(0, 0.6, n) > 0.3, "churn", "stay")
    age[rng.choice(n, 12, replace=False)] = np.nan
    return pd.DataFrame(
        {
            "customer_id": np.arange(n),
            "age": age,
            "income": income.round(2),
            "plan": plan,
            "region": region,
            "label": y,
        }
    )


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()


@pytest.fixture()
def churn_csv() -> bytes:
    return to_csv_bytes(mixed_frame())
