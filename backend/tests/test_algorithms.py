from __future__ import annotations

import pytest

from evalforge.ml import metrics as M
from evalforge.ml.algorithms import ALGORITHMS, HyperparameterError, build_pipeline, get_algorithm

from .conftest import mixed_frame


def test_hyperparameter_validation():
    lr = get_algorithm("logistic_regression")
    assert lr.validate({})["C"] == 1.0
    with pytest.raises(HyperparameterError):
        lr.validate({"C": -1})
    with pytest.raises(HyperparameterError):
        lr.validate({"evil": "__import__('os')"})
    with pytest.raises(HyperparameterError):
        get_algorithm("random_forest").validate({"n_estimators": 2.5})
    with pytest.raises(HyperparameterError):
        get_algorithm("nope")


@pytest.mark.parametrize("key", sorted(ALGORITHMS))
def test_every_algorithm_learns_mixed_types_with_missing_values(key):
    df = mixed_frame(400)
    X, y = df.drop(columns=["label", "customer_id"]), df["label"]
    algo = ALGORITHMS[key]
    pipe = build_pipeline(algo, algo.validate({}), X, seed=0)
    pipe.fit(X.iloc[:300], y.iloc[:300])
    m, report = M.classification_metrics(pipe, X.iloc[300:], y.iloc[300:], ["churn", "stay"])
    assert m["accuracy"] > 0.7  # signal is learnable; a constant model scores ~0.5-0.6
    assert 0 <= m["roc_auc"] <= 1
    assert len(report["confusion_matrix"]) == 2
    lat = M.inference_latency(pipe, X.iloc[300:], n_single=5)
    assert lat["latency_single_p95_ms"] >= lat["latency_single_p50_ms"] > 0


def test_multiclass_metrics_use_macro_average():
    df = mixed_frame(300)
    df["label"] = df["plan"]
    X, y = df.drop(columns=["label", "plan", "customer_id"]), df["label"]
    algo = ALGORITHMS["logistic_regression"]
    pipe = build_pipeline(algo, algo.validate({}), X, seed=0).fit(X, y)
    m, report = M.classification_metrics(pipe, X, y, ["basic", "plus", "pro"])
    assert report["averaging"] == "macro" and "roc_auc_ovr" in m and "roc_auc" not in m
