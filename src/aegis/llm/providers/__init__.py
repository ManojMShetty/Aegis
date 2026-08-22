"""Concrete LLM providers.

Each implements the :class:`~aegis.llm.base.LLMProvider` protocol structurally,
so the router and tests treat them interchangeably:

    gemini -> aegis.llm.providers.gemini.GeminiProvider   (live, httpx)
    fake   -> aegis.llm.providers.fake.FakeProvider        (offline, scriptable)
"""

from aegis.llm.providers.fake import FakeProvider, RecordedCall
from aegis.llm.providers.gemini import GeminiProvider

__all__ = ["FakeProvider", "GeminiProvider", "RecordedCall"]
