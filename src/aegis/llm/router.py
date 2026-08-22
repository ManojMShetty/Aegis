"""Role -> provider routing, driven by ``config/models.yaml``.

WHY A ROUTER, AND WHY IT READS CONFIG
-------------------------------------
The security architecture assigns each Aegis principal (privileged agent,
quarantine extractor, judge, attacker) to its own model. *Which* model, and even
which vendor, is an operational choice that should be auditable as data - the
same principle as the trust policy in ``config/trust_tiers.yaml``. So the mapping
lives in YAML and this module is the small, strict bridge that turns it into live
providers.

Two design points matter:

* One :class:`Budget` is shared across every provider the router builds, so the
  free-tier caps span all roles on the key - the quarantine model cannot quietly
  spend the agent's allowance.
* Providers are built lazily and cached, so importing the router (or routing one
  role) never forces construction of a provider whose key or network you do not
  need - which is what lets the ``fake`` kind route entirely offline.

Unknown roles and unknown provider kinds fail closed with a clear error rather
than defaulting to something that silently makes real calls.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from aegis.llm.base import LLMError, LLMProvider, Role
from aegis.llm.budget import Budget
from aegis.llm.providers.fake import FakeProvider
from aegis.llm.providers.gemini import GeminiProvider

__all__ = ["DEFAULT_MODELS_PATH", "LLMRouter", "RouterError"]

DEFAULT_MODELS_PATH = Path(__file__).resolve().parents[3] / "config" / "models.yaml"


class RouterError(LLMError):
    """The model routing config is malformed, or names an unknown role/provider.

    Subclasses :class:`LLMError` so callers can catch the whole LLM layer's
    failures uniformly, while still distinguishing a config fault by type.
    """


class LLMRouter:
    """Resolves an Aegis :class:`Role` to a configured :class:`LLMProvider`."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        budget: Budget | None = None,
        api_key: str | None = None,
        http_client: object | None = None,
    ) -> None:
        roles = config.get("roles")
        if not isinstance(roles, Mapping):
            raise RouterError("models config must contain a 'roles' mapping")
        self._roles: Mapping[str, Any] = roles
        self._api_key = api_key
        self._http_client = http_client
        self._budget = budget if budget is not None else _budget_from_config(config.get("budget"))
        self._cache: dict[str, LLMProvider] = {}

    @classmethod
    def load(
        cls,
        path: Path | str | None = None,
        *,
        budget: Budget | None = None,
        api_key: str | None = None,
        http_client: object | None = None,
    ) -> LLMRouter:
        """Load and validate a models config file."""
        p = Path(path) if path is not None else DEFAULT_MODELS_PATH
        if not p.is_file():
            raise RouterError(f"models config not found at {p}")
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise RouterError(f"{p} is not valid YAML: {exc}") from exc
        if not isinstance(raw, Mapping):
            raise RouterError(f"{p} must contain a mapping at the top level")
        return cls(raw, budget=budget, api_key=api_key, http_client=http_client)

    @property
    def budget(self) -> Budget:
        """The single budget shared by every provider this router builds."""
        return self._budget

    def close(self) -> None:
        """Close every provider this router built that owns a client.

        Providers are lazy, so this only touches ones actually used. Safe to call
        on a router whose providers have no ``close`` (e.g. the fake provider).
        """
        for provider in self._cache.values():
            close = getattr(provider, "close", None)
            if callable(close):
                close()

    def provider_for(self, role: Role) -> LLMProvider:
        """Return the provider for ``role``, building and caching it on first use."""
        if role.value in self._cache:
            return self._cache[role.value]

        spec = self._roles.get(role.value)
        if not isinstance(spec, Mapping):
            available = ", ".join(sorted(self._roles)) or "<none>"
            raise RouterError(
                f"no config for role '{role.value}'; models config defines: {available}"
            )

        provider = self._build(role, spec)
        self._cache[role.value] = provider
        return provider

    def _build(self, role: Role, spec: Mapping[str, Any]) -> LLMProvider:
        kind = spec.get("provider")
        model = spec.get("model")
        if not isinstance(kind, str) or not kind:
            raise RouterError(f"role '{role.value}' is missing a 'provider' name")
        if not isinstance(model, str) or not model:
            raise RouterError(f"role '{role.value}' is missing a 'model' name")

        if kind == "gemini":
            return GeminiProvider(
                model=model,
                api_key=self._api_key,
                budget=self._budget,
                http_client=_as_client(self._http_client),
            )
        if kind == "fake":
            return FakeProvider(
                model=model,
                echo_json=bool(spec.get("echo_json", False)),
                budget=self._budget,
            )

        raise RouterError(
            f"role '{role.value}' names unknown provider kind '{kind}'; "
            "supported kinds are: gemini, fake"
        )


def _budget_from_config(raw: object) -> Budget:
    """Build the shared budget from the optional ``budget`` config section."""
    section: Mapping[str, object] = raw if isinstance(raw, Mapping) else {}
    return Budget(
        max_requests_per_day=_cap(section, "max_requests_per_day"),
        max_requests_per_run=_cap(section, "max_requests_per_run"),
    )


def _cap(section: Mapping[str, object], key: str) -> int | None:
    """A positive-int cap from config: absent -> None, present-but-invalid -> error.

    Fail-closed. A quoted number, a bool, a zero, or a negative must not silently
    resolve to *unlimited* in the one component whose entire purpose is to cap
    spend - so an invalid value raises rather than disabling the guard.
    """
    if key not in section:
        return None
    value = section[key]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RouterError(
            f"budget.{key} must be a positive integer, got {value!r}; "
            "an invalid cap must fail rather than silently disable the budget"
        )
    return value


def _as_client(value: object) -> Any:
    """Pass an injected httpx.Client straight through (typed loosely so the router
    itself need not import httpx)."""
    return value
