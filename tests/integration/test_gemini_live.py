"""One real Gemini call - the only test that spends quota.

This is the smoke test that proves the wire contract the unit tests fake is the
one Gemini actually speaks. It is guarded three ways so it never runs by
accident: the ``costly`` marker (excluded by ``-m "not costly"``), and a skip
unless BOTH ``GEMINI_API_KEY`` and ``AEGIS_RUN_COSTLY=1`` are set. It caps output
at a handful of tokens, so a run costs essentially nothing.

    AEGIS_RUN_COSTLY=1 uv run pytest -m costly
"""

from __future__ import annotations

import os

import pytest

from aegis.llm.base import LLMResponse, Message
from aegis.llm.providers.gemini import GeminiProvider

pytestmark = pytest.mark.costly

_RUN = os.environ.get("AEGIS_RUN_COSTLY") == "1" and bool(os.environ.get("GEMINI_API_KEY"))


@pytest.mark.skipif(
    not _RUN,
    reason="set GEMINI_API_KEY and AEGIS_RUN_COSTLY=1 to run the live Gemini call",
)
def test_live_generate_content_smoke() -> None:
    provider = GeminiProvider(model="gemini-3.5-flash-lite")
    try:
        resp = provider.complete(
            [Message(role="user", content="Reply with the single word: pong")],
            max_output_tokens=16,
            temperature=0.0,
        )
    finally:
        provider.close()

    assert isinstance(resp, LLMResponse)
    assert resp.text.strip() != ""
    assert resp.model == "gemini-3.5-flash-lite"
    assert resp.input_tokens > 0
    assert resp.finish_reason != ""
