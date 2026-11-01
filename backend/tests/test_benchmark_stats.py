"""Uncertainty reporting for model comparisons (audit BEN-01/BEN-02)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

from evalforge.ml import metrics as M

from .conftest import as_user
from .test_ops import train, upload


def _codes(y, p, labels):
    idx = {lbl: i for i, lbl in enumerate(labels)}
    return np.array([idx[v] for v in y]), np.array([idx[v] for v in p])


def test_vectorised_f1_matches_sklearn():
    rng = np.random.default_rng(0)
    for labels in (["no", "yes"], ["a", "b", "c", "d"]):
        y = rng.choice(labels, 300)
        p = np.where(rng.random(300) < 0.7, y, rng.choice(labels, 300))
        yc, pc = _codes(y, p, labels)
        if len(labels) == 2:
            ref = f1_score(y, p, average="binary", pos_label=labels[1], zero_division=0)
            got = M._f1_codes(yc, pc, 2, 1)
        else:
            ref = f1_score(y, p, average="macro", labels=labels, zero_division=0)
            got = M._f1_codes(yc, pc, len(labels), None)
        assert abs(ref - got) < 1e-12


def test_identical_models_are_not_distinguishable():
    rng = np.random.default_rng(1)
    y = rng.choice(["no", "yes"], 120)
    p = np.where(rng.random(120) < 0.85, y, "no")
    s = M.bootstrap_f1(y, [p, p.copy()], ["no", "yes"])
    lo, hi = M.ci95(s[:, 0] - s[:, 1])
    assert lo == hi == 0.0


def test_clearly_better_model_is_distinguishable_and_ci_brackets_estimate():
    rng = np.random.default_rng(2)
    y = rng.choice(["no", "yes"], 400)
    good = np.where(rng.random(400) < 0.95, y, rng.choice(["no", "yes"], 400))
    bad = np.where(rng.random(400) < 0.55, y, rng.choice(["no", "yes"], 400))
    s = M.bootstrap_f1(y, [good, bad], ["no", "yes"])
    lo, _ = M.ci95(s[:, 0] - s[:, 1])
    assert lo > 0
    point = f1_score(y, good, pos_label="yes")
    glo, ghi = M.ci95(s[:, 0])
    assert glo <= point <= ghi and ghi - glo < 0.15


def test_small_holdout_interval_is_wide():
    """114 rows (the breast-cancer holdout size) cannot resolve sub-percent differences."""
    rng = np.random.default_rng(3)
    y = rng.choice(["B", "M"], 114, p=[0.63, 0.37])
    a = np.where(rng.random(114) < 0.95, y, np.where(y == "B", "M", "B"))
    b = a.copy()
    flip = rng.choice(114, 1, replace=False)  # one row differs: Δ ≈ ±0.01
    b[flip] = np.where(b[flip] == "B", "M", "B")
    s = M.bootstrap_f1(y, [a, b], ["B", "M"])
    lo, hi = M.ci95(s[:, 0] - s[:, 1])
    assert lo <= 0 <= hi


def test_run_metrics_and_gate_report_uncertainty(client, churn_csv):
    ds = upload(client, churn_csv)
    vid = ds["versions"][0]["id"]
    eng = as_user(client, "engineer")
    r1 = train(client, vid, "logistic_regression")
    m = client.get(f"/api/runs/{r1}").json()["metrics"]
    assert m["f1_ci95_low"] <= m["f1"] <= m["f1_ci95_high"]
    mv1 = client.post("/api/registry/versions", json={"run_id": r1}, headers=eng).json()
    appr = client.post(f"/api/registry/versions/{mv1['id']}/request-promotion", headers=eng).json()
    from .test_ops import approve

    assert approve(client, appr["id"]).json()["status"] == "executed"
    r2 = train(client, vid, "logistic_regression", {"C": 0.8})
    mv2 = client.post("/api/registry/versions", json={"run_id": r2}, headers=eng).json()
    comp = mv2["gate"]["champion_comparison"]
    lo, hi = comp["delta_f1_ci95"]
    assert lo <= comp["delta_f1"] <= hi and comp["n_rows"] > 0
    check = next(c for c in mv2["gate"]["checks"] if c["id"] == "difference_is_distinguishable")
    assert check["blocking"] is False and check["passed"] == comp["distinguishable"]
