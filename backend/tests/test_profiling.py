from __future__ import annotations

import numpy as np
import pandas as pd

from evalforge.models import TaskType
from evalforge.services import profiling

from .conftest import mixed_frame


def _by_id(checks):
    return {c.id: c for c in checks}


def test_quality_checks_pass_on_clean_data():
    checks, task = profiling.run_quality_checks(mixed_frame(), "label")
    c = _by_id(checks)
    assert task is TaskType.binary_classification
    assert all(x.passed for x in checks if x.severity == "error")
    assert c["identifier_columns"].details["columns"] == ["customer_id"]


def test_missing_target_is_blocking():
    checks, task = profiling.run_quality_checks(mixed_frame(), "nope")
    assert task is None
    assert not _by_id(checks)["target_present"].passed


def test_too_few_rows_and_regression_target_blocked():
    df = pd.DataFrame({"x": range(10), "y": np.linspace(0, 1, 10)})
    checks, task = profiling.run_quality_checks(df, "y")
    c = _by_id(checks)
    assert not c["min_rows"].passed
    assert task is None and not c["task_type"].passed


def test_target_leakage_detected():
    df = mixed_frame()
    df["label_copy_code"] = df["label"].map({"churn": "C", "stay": "S"})
    df["leaky_numeric"] = (df["label"] == "stay").astype(float) * 10 + np.random.default_rng(
        1
    ).normal(0, 0.01, len(df))
    checks, _ = profiling.run_quality_checks(df, "label")
    suspects = _by_id(checks)["target_leakage"].details["suspects"]
    assert "label_copy_code" in suspects
    assert "leaky_numeric" in suspects
    assert "age" not in suspects


def test_profile_has_histograms_and_top_values():
    p = profiling.profile_frame(mixed_frame(), "label")
    assert p["columns"]["age"]["missing"] == 12
    assert sum(p["columns"]["income"]["histogram"]["counts"]) == 240
    assert {v["value"] for v in p["columns"]["plan"]["top_values"]} == {"basic", "plus", "pro"}
    assert sum(c["count"] for c in p["target"]["classes"]) == 240


def test_split_is_deterministic_stratified_and_detects_overlap(tmp_path):
    df = mixed_frame()
    s1 = profiling.make_split(df, "label", tmp_path)
    tr1, te1 = profiling.load_split(s1["path"])
    s2 = profiling.make_split(df, "label", tmp_path)
    tr2, te2 = profiling.load_split(s2["path"])
    assert (te1 == te2).all() and (tr1 == tr2).all()
    assert len(set(tr1) & set(te1)) == 0
    ratio_all = (df["label"] == "churn").mean()
    ratio_test = (df.loc[te1, "label"] == "churn").mean()
    assert abs(ratio_all - ratio_test) < 0.03
    assert s1["train_test_overlap_rows"] == 0

    dup = pd.concat([df, df]).reset_index(drop=True)
    s3 = profiling.make_split(dup, "label", tmp_path)
    assert s3["train_test_overlap_rows"] > 0
