"""The provider-agnostic LLM layer.

Aegis runs several LLMs as distinct trust principals (privileged agent,
quarantine extractor, judge, attacker). This package gives them one narrow,
vendor-neutral surface plus the plumbing that keeps a free-tier run honest:

    base     -> the contract: Role, Message, LLMResponse, LLMProvider, errors
    budget   -> the fail-closed frugality guard (per-day / per-run caps)
    router   -> role -> provider routing from config/models.yaml
    providers-> GeminiProvider (live, httpx) and FakeProvider (offline)

The design rationale lives in ``base`` and ``budget``; start there.
"""

from aegis.llm.base import (
    LLMError,
    LLMProvider,
    LLMRefusal,
    LLMResponse,
    Message,
    RateLimitError,
    Role,
)
from aegis.llm.budget import Budget, BudgetExceeded
from aegis.llm.providers.fake import FakeProvider, RecordedCall
from aegis.llm.router import LLMRouter, RouterError

__all__ = [
    "Budget",
    "BudgetExceeded",
    "FakeProvider",
    "GeminiProvider",
    "LLMError",
    "LLMProvider",
    "LLMRefusal",
    "LLMResponse",
    "LLMRouter",
    "Message",
    "RateLimitError",
    "RecordedCall",
    "Role",
    "RouterError",
]


def __getattr__(name: str) -> object:
    """Resolve ``GeminiProvider`` on first use.

    See :mod:`aegis.llm.providers` for why: importing it eagerly made httpx a hard
    requirement of ``import aegis.middleware``, for a layer that is off by default.
    """
    if name == "GeminiProvider":
        from aegis.llm.providers import GeminiProvider as _GeminiProvider

        return _GeminiProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
