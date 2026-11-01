"""MLflow integration.

* Experiments are named ``<prefix>/<dataset name>`` so every dataset gets its own
  comparable run table in the MLflow UI.
* Each run logs params, metrics, reproducibility tags, the dataset version as an
  MLflow dataset input (with our content hash as digest), diagnostics JSON and the
  fitted pipeline as an MLflow model with signature and input example.
* Models are serialised with **skops** rather than pickle where possible, so
  loading a model never executes arbitrary code (MLflow ≥3 supports this).
* Registry/promotion uses aliases (``champion``/``challenger``) in Phase 3; MLflow
  stage transitions are deprecated and are not used.
"""

from __future__ import annotations

import logging
import platform
import subprocess
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline

from evalforge.config import get_settings

log = logging.getLogger(__name__)
_lock = threading.Lock()


@lru_cache
def configure() -> str:
    uri = get_settings().resolved_mlflow_uri
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    return uri


def experiment_for(dataset_name: str) -> str:
    configure()
    s = get_settings()
    name = f"{s.mlflow_experiment_prefix}/{dataset_name}"
    with _lock:
        exp = mlflow.get_experiment_by_name(name)
        if exp is not None:
            return exp.experiment_id
        artifact_location = None
        if not s.mlflow_tracking_uri:  # embedded store -> keep artifacts under data_dir
            artifact_location = (s.mlflow_artifacts_dir / dataset_name).resolve().as_uri()
        return mlflow.create_experiment(name, artifact_location=artifact_location)


def git_info() -> dict[str, str | bool]:
    s = get_settings()
    if s.git_commit:
        return {"commit": s.git_commit, "dirty": False, "source": "env"}
    try:
        root = Path(__file__).resolve().parents[2]
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, timeout=5
            )
            .decode()
            .strip()
        )
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, stderr=subprocess.DEVNULL, timeout=5
            ).strip()
        )
        return {"commit": commit, "dirty": dirty, "source": "git"}
    except Exception:
        return {"commit": "unknown", "dirty": False, "source": "none"}


def environment_info() -> dict[str, str]:
    import numpy

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "sklearn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": numpy.__version__,
        "mlflow": mlflow.__version__,
    }


def log_training_run(
    *,
    experiment_id: str,
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    tags: dict[str, str],
    model: Pipeline,
    X_example: pd.DataFrame,
    train_df: pd.DataFrame,
    dataset_name: str,
    dataset_digest: str,
    dataset_source: str,
    target: str,
    artifacts: dict[str, Any],
) -> tuple[str, str]:
    """Log one completed training run. Returns ``(mlflow_run_id, model_uri)``."""
    configure()
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        mlflow.set_tags(tags)
        mlflow.log_params({k: ("None" if v is None else v) for k, v in params.items()})
        mlflow.log_metrics(metrics)
        ds = mlflow.data.from_pandas(
            train_df,
            source=dataset_source,
            targets=target,
            name=dataset_name,
            digest=dataset_digest[:36],
        )
        mlflow.log_input(ds, context="training")
        for fname, payload in artifacts.items():
            mlflow.log_dict(payload, f"diagnostics/{fname}")

        example = X_example.head(5)
        signature = infer_signature(example, model.predict(example))
        info = _log_model(model, signature, example)
        return run.info.run_id, info.model_uri


def _log_model(model: Pipeline, signature: Any, example: pd.DataFrame) -> Any:
    try:
        return mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=example,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_SKOPS,
            skops_trusted_types=_trusted_types(model),
        )
    except Exception as exc:  # pragma: no cover - only when skops can't represent a type
        log.warning("skops serialisation failed (%s); falling back to cloudpickle", exc)
        return mlflow.sklearn.log_model(
            model, name="model", signature=signature, input_example=example
        )


def _trusted_types(model: Pipeline) -> list[str]:
    """Types skops flags as untrusted by default that we produced ourselves."""
    try:
        import skops.io as sio

        buf = sio.dumps(model)
        return list(sio.get_untrusted_types(data=buf))
    except Exception:
        return []


def load_model(model_uri: str) -> Pipeline:
    configure()
    return mlflow.sklearn.load_model(model_uri)


def ping() -> bool:
    configure()
    mlflow.search_experiments(max_results=1)
    return True
