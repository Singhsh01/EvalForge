"""Provider abstraction for LLM calls.

Model references are ``<provider>:<model>``:

* ``anthropic:<model>``  — Anthropic Messages API (``EVALFORGE_ANTHROPIC_API_KEY``)
* ``openai:<model>``     — any OpenAI-compatible ``/chat/completions`` endpoint
  (``EVALFORGE_OPENAI_BASE_URL`` / ``EVALFORGE_OPENAI_API_KEY``; works with vLLM,
  Ollama, LM Studio, OpenAI)
* ``fixture:<name>``     — deterministic test double that replays registered
  responses. It is not an LLM; it is only available when ``EVALFORGE_ENV=test``.

Cost is computed only from prices supplied in ``EVALFORGE_LLM_PRICING`` (JSON:
``{"model": [input_usd_per_mtok, output_usd_per_mtok]}``). Without a configured
price the cost is reported as unknown (``None``), never guessed.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from evalforge.config import get_settings


class ProviderError(RuntimeError):
    pass


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)


def _pricing() -> dict[str, list[float]]:
    raw = os.environ.get("EVALFORGE_LLM_PRICING")
    try:
        return json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {}


def cost_for(model: str, tin: int, tout: int) -> float | None:
    p = _pricing().get(model)
    if not p:
        return None
    return round(tin / 1e6 * p[0] + tout / 1e6 * p[1], 8)


class Provider:
    name = "base"

    def complete(
        self,
        model: str,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        raise NotImplementedError


class AnthropicProvider(Provider):
    name = "anthropic"

    def complete(
        self,
        model: str,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        s = get_settings()
        if not s.anthropic_api_key:
            raise ProviderError("EVALFORGE_ANTHROPIC_API_KEY is not configured")
        t0 = time.perf_counter()
        r = httpx.post(
            f"{s.anthropic_base_url.rstrip('/')}/v1/messages",
            headers={
                "x-api-key": s.anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
            timeout=s.llm_timeout_seconds,
        )
        ms = (time.perf_counter() - t0) * 1000
        if r.status_code >= 400:
            raise ProviderError(f"Anthropic API {r.status_code}: {r.text[:300]}")
        data = r.json()
        text = "".join(
            b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"
        )
        u = data.get("usage", {})
        tin, tout = int(u.get("input_tokens", 0)), int(u.get("output_tokens", 0))
        return LLMResponse(text, self.name, model, tin, tout, ms, cost_for(model, tin, tout))


class OpenAICompatibleProvider(Provider):
    name = "openai"

    def complete(
        self,
        model: str,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        s = get_settings()
        base = s.openai_base_url or ("https://api.openai.com/v1" if s.openai_api_key else None)
        if not base:
            raise ProviderError("EVALFORGE_OPENAI_BASE_URL (or OPENAI API key) is not configured")
        headers = {"content-type": "application/json"}
        if s.openai_api_key:
            headers["authorization"] = f"Bearer {s.openai_api_key}"
        t0 = time.perf_counter()
        r = httpx.post(
            f"{base.rstrip('/')}/chat/completions",
            headers=headers,
            json={
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=s.llm_timeout_seconds,
        )
        ms = (time.perf_counter() - t0) * 1000
        if r.status_code >= 400:
            raise ProviderError(f"OpenAI-compatible API {r.status_code}: {r.text[:300]}")
        data = r.json()
        text = data["choices"][0]["message"].get("content") or ""
        u = data.get("usage", {})
        tin, tout = int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0))
        return LLMResponse(text, self.name, model, tin, tout, ms, cost_for(model, tin, tout))


class FixtureProvider(Provider):
    """Deterministic replay for automated tests. Refuses to run outside EVALFORGE_ENV=test."""

    name = "fixture"
    responders: dict[str, Callable[[str, str], str]] = {}

    def complete(
        self,
        model: str,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        if get_settings().env != "test":
            raise ProviderError("The fixture provider is a test double and only runs in tests")
        fn = self.responders.get(model)
        if fn is None:
            raise ProviderError(f"No fixture responder registered for '{model}'")
        text = fn(system, user)
        return LLMResponse(text, self.name, model, len(user.split()), len(text.split()), 0.1, None)


PROVIDERS: dict[str, Provider] = {
    p.name: p for p in (AnthropicProvider(), OpenAICompatibleProvider(), FixtureProvider())
}


def parse_ref(ref: str) -> tuple[Provider, str]:
    if ":" not in ref:
        raise ProviderError("Model reference must look like '<provider>:<model>'")
    pname, model = ref.split(":", 1)
    if pname not in PROVIDERS:
        raise ProviderError(f"Unknown provider '{pname}'. Available: {sorted(PROVIDERS)}")
    return PROVIDERS[pname], model


def available() -> list[dict[str, Any]]:
    s = get_settings()
    return [
        {"provider": "anthropic", "configured": bool(s.anthropic_api_key)},
        {
            "provider": "openai",
            "configured": bool(s.openai_base_url or s.openai_api_key),
            "base_url": s.openai_base_url,
        },
    ]


def complete(ref: str, system: str, user: str, **kw: Any) -> LLMResponse:
    provider, model = parse_ref(ref)
    return provider.complete(model, system, user, **kw)
