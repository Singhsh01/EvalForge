"""Lazy, bounded in-process cache of models loaded from MLflow (laptop-friendly)."""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

from evalforge.config import get_settings
from evalforge.services import tracking

_cache: OrderedDict[str, Any] = OrderedDict()
_lock = threading.Lock()


def load_cached(uri: str, sha256: str | None = None) -> Any:
    key = f"{uri}#{sha256 or ''}"
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    model = tracking.load_model(uri, sha256)  # skops-only, digest-verified, allowlisted types
    with _lock:
        _cache[key] = model
        while len(_cache) > get_settings().model_cache_size:
            _cache.popitem(last=False)
    return model


def clear() -> None:
    with _lock:
        _cache.clear()
