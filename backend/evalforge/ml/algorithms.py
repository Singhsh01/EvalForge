"""Algorithm catalogue with validated hyperparameter schemas.

The frontend renders its training form from :func:`catalogue`, and the API
validates user hyperparameters against the same specs, so no arbitrary keyword
arguments ever reach scikit-learn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from evalforge.services.profiling import column_kind


class HyperparameterError(ValueError):
    pass


@dataclass(frozen=True)
class Param:
    name: str
    type: str  # "int" | "float" | "choice"
    default: Any
    min: float | None = None
    max: float | None = None
    choices: tuple[Any, ...] | None = None
    nullable: bool = False
    log_scale: bool = False
    help: str = ""

    def validate(self, value: Any) -> Any:
        if value is None:
            if self.nullable:
                return None
            raise HyperparameterError(f"{self.name} may not be null")
        if self.type == "choice":
            if value not in (self.choices or ()):
                raise HyperparameterError(f"{self.name} must be one of {self.choices}")
            return value
        try:
            v = int(value) if self.type == "int" else float(value)
        except (TypeError, ValueError) as exc:
            raise HyperparameterError(f"{self.name} must be {self.type}") from exc
        if self.type == "int" and float(value) != v:
            raise HyperparameterError(f"{self.name} must be an integer")
        if self.min is not None and v < self.min:
            raise HyperparameterError(f"{self.name} must be >= {self.min}")
        if self.max is not None and v > self.max:
            raise HyperparameterError(f"{self.name} must be <= {self.max}")
        return v


@dataclass(frozen=True)
class Algorithm:
    key: str
    label: str
    description: str
    params: tuple[Param, ...]
    scale_numeric: bool
    native_missing: bool = False
    tags: tuple[str, ...] = field(default_factory=tuple)

    def validate(self, hp: dict[str, Any] | None) -> dict[str, Any]:
        hp = dict(hp or {})
        known = {p.name for p in self.params}
        unknown = set(hp) - known
        if unknown:
            raise HyperparameterError(f"Unknown hyperparameters for {self.key}: {sorted(unknown)}")
        return {p.name: p.validate(hp.get(p.name, p.default)) for p in self.params}


ALGORITHMS: dict[str, Algorithm] = {
    a.key: a
    for a in [
        Algorithm(
            key="logistic_regression",
            label="Logistic Regression",
            description="Regularised linear baseline. Fast, calibrated-ish probabilities, "
            "interpretable coefficients. Numeric features are standardised.",
            scale_numeric=True,
            tags=("linear", "baseline"),
            params=(
                Param(
                    "C",
                    "float",
                    1.0,
                    1e-4,
                    1e4,
                    log_scale=True,
                    help="Inverse regularisation strength",
                ),
                Param("max_iter", "int", 1000, 50, 10000),
                Param("class_weight", "choice", None, choices=(None, "balanced"), nullable=True),
            ),
        ),
        Algorithm(
            key="random_forest",
            label="Random Forest",
            description="Bagged decision trees. Robust non-linear model with little tuning; "
            "parallelism capped for laptop profiles.",
            scale_numeric=False,
            tags=("ensemble", "trees"),
            params=(
                Param("n_estimators", "int", 200, 10, 1000),
                Param("max_depth", "int", None, 1, 64, nullable=True),
                Param("min_samples_leaf", "int", 1, 1, 100),
                Param(
                    "max_features", "choice", "sqrt", choices=("sqrt", "log2", None), nullable=True
                ),
                Param("class_weight", "choice", None, choices=(None, "balanced"), nullable=True),
            ),
        ),
        Algorithm(
            key="gradient_boosting",
            label="Histogram Gradient Boosting",
            description="scikit-learn's LightGBM-style boosted trees. Strong tabular "
            "performance, native missing-value handling, CPU-efficient.",
            scale_numeric=False,
            native_missing=True,
            tags=("ensemble", "boosting"),
            params=(
                Param("learning_rate", "float", 0.1, 0.001, 1.0, log_scale=True),
                Param("max_iter", "int", 200, 10, 2000, help="Boosting rounds"),
                Param("max_depth", "int", None, 1, 32, nullable=True),
                Param("max_leaf_nodes", "int", 31, 2, 256),
                Param("l2_regularization", "float", 0.0, 0.0, 100.0),
            ),
        ),
    ]
}


def catalogue() -> list[dict[str, Any]]:
    return [
        {
            "key": a.key,
            "label": a.label,
            "description": a.description,
            "tags": list(a.tags),
            "params": [
                {
                    "name": p.name,
                    "type": p.type,
                    "default": p.default,
                    "min": p.min,
                    "max": p.max,
                    "choices": list(p.choices) if p.choices is not None else None,
                    "nullable": p.nullable,
                    "log_scale": p.log_scale,
                    "help": p.help,
                }
                for p in a.params
            ],
        }
        for a in ALGORITHMS.values()
    ]


def get_algorithm(key: str) -> Algorithm:
    try:
        return ALGORITHMS[key]
    except KeyError as exc:
        raise HyperparameterError(
            f"Unknown algorithm '{key}'. Available: {sorted(ALGORITHMS)}"
        ) from exc


def build_pipeline(
    algo: Algorithm, hp: dict[str, Any], X: pd.DataFrame, seed: int, n_jobs: int = 2
) -> Pipeline:
    """Reproducible preprocessing + estimator in a single, serialisable Pipeline."""
    numeric = [c for c in X.columns if column_kind(X[c]) in {"numeric", "boolean"}]
    categorical = [c for c in X.columns if c not in numeric]

    num_steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median"))]
    if algo.scale_numeric:
        num_steps.append(("scale", StandardScaler()))
    cat_steps = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        (
            "encode",
            OneHotEncoder(
                handle_unknown="infrequent_if_exist", max_categories=30, sparse_output=False
            ),
        ),
    ]
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline(num_steps), numeric))
    if categorical:
        transformers.append(("cat", Pipeline(cat_steps), categorical))
    pre = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)

    if algo.key == "logistic_regression":
        est = LogisticRegression(random_state=seed, **hp)
    elif algo.key == "random_forest":
        est = RandomForestClassifier(random_state=seed, n_jobs=n_jobs, **hp)
    elif algo.key == "gradient_boosting":
        est = HistGradientBoostingClassifier(random_state=seed, **hp)
    else:  # pragma: no cover - guarded by get_algorithm
        raise HyperparameterError(algo.key)
    return Pipeline([("preprocess", pre), ("model", est)])
