"""Concrete LLM providers.

Each implements the :class:`~aegis.llm.base.LLMProvider` protocol structurally,
so the router and tests treat them interchangeably:

    gemini -> aegis.llm.providers.gemini.GeminiProvider   (live, needs httpx)
    fake   -> aegis.llm.providers.fake.FakeProvider        (offline, scriptable)

WHY ``GeminiProvider`` IS IMPORTED LAZILY
-----------------------------------------
Importing it eagerly made ``httpx`` a hard requirement of anything that touched
``aegis.llm`` - and, through the L4 quarantine extractor, of
:mod:`aegis.middleware` itself. So ``pip install aegis-rag`` without the ``llm``
extra produced a ``ModuleNotFoundError`` on ``import aegis.middleware``, for a
transport belonging to a layer that is OFF by default.

That went unnoticed for the usual reason: every test runs from a checkout where
httpx is already installed, so the import chain always resolved. It is only
visible from a clean install, which is exactly the audience the middleware exists
for.

``__getattr__`` (PEP 562) keeps ``from aegis.llm.providers import GeminiProvider``
working unchanged for anyone who has the extra, while the name costs nothing to
anyone who does not. The error, when it comes, now names the extra to install
rather than a transitive package the caller never asked for.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aegis.llm.providers.fake import FakeProvider, RecordedCall

if TYPE_CHECKING:
    from aegis.llm.providers.gemini import GeminiProvider

__all__ = ["FakeProvider", "GeminiProvider", "RecordedCall"]


def __getattr__(name: str) -> Any:
    """Resolve ``GeminiProvider`` on first use, not on package import."""
    if name == "GeminiProvider":
        try:
            from aegis.llm.providers.gemini import GeminiProvider as _GeminiProvider
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on the install
            raise ModuleNotFoundError(
                "GeminiProvider needs the 'llm' extra (httpx). Install it with "
                "`pip install aegis-rag[llm]`. Nothing else in aegis requires it: "
                "the trust lattice, the capability gate and the middleware all run "
                "without a network stack."
            ) from exc
        return _GeminiProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
