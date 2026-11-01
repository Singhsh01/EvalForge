"""Lazy, bounded in-process cache of models loaded from MLflow (laptop-friendly)."""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

from evalforge.config import get_settings
from evalforge.services import tracking

_cache: OrderedDict[str, Any] = OrderedDict()
_lock = threading.Lock()


def load_cached(uri: str) -> Any:
    with _lock:
        if uri in _cache:
            _cache.move_to_end(uri)
            return _cache[uri]
    model = tracking.load_model(uri)  # skops-safe loading via MLflow
    with _lock:
        _cache[uri] = model
        while len(_cache) > get_settings().model_cache_size:
            _cache.popitem(last=False)
    return model


def clear() -> None:
    with _lock:
        _cache.clear()
