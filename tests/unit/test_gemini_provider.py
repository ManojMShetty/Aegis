"""GeminiProvider against a faked transport - no network, no key, no spend.

httpx ships MockTransport, so the entire wire contract (request shape, retries,
refusals, key redaction) is exercised in-process. Backoff sleeps are patched to
nothing, so even the retry tests are instant.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from aegis.llm import providers
from aegis.llm.base import LLMError, LLMRefusal, Message, RateLimitError
from aegis.llm.budget import Budget, BudgetExceeded
from aegis.llm.providers.gemini import GeminiProvider

API_KEY = "SECRET_TEST_KEY_do_not_leak_123"

OK_PAYLOAD: dict[str, Any] = {
    "candidates": [
        {
            "content": {"role": "model", "parts": [{"text": "Hello "}, {"text": "there"}]},
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 12,
        "candidatesTokenCount": 3,
        "totalTokenCount": 15,
    },
}


def client_for(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def provider_for(
    handler: Callable[[httpx.Request], httpx.Response],
    **kwargs: Any,
) -> GeminiProvider:
    return GeminiProvider(
        model="gemini-2.5-flash",
        api_key=API_KEY,
        http_client=client_for(handler),
        **kwargs,
    )


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Replace the backoff sleep with a recorder so retries never actually wait."""
    slept: list[float] = []
    monkeypatch.setattr(
        "aegis.llm.providers.gemini.time.sleep",
        lambda seconds: slept.append(seconds),
    )
    return slept


# --------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------


def test_plain_completion_parses_text_and_tokens() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=OK_PAYLOAD)

    provider = provider_for(handler)
    resp = provider.complete([Message(role="user", content="hi")])

    assert resp.text == "Hello there"  # parts concatenated
    assert resp.model == "gemini-2.5-flash"
    assert resp.input_tokens == 12
    assert resp.output_tokens == 3
    assert resp.finish_reason == "STOP"
    assert resp.raw["candidates"]  # full payload retained for debugging


def test_request_sends_key_in_header_not_url() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["header_key"] = request.headers.get("x-goog-api-key")
        seen["url_key"] = request.url.params.get("key")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=OK_PAYLOAD)

    provider = provider_for(handler)
    provider.complete([Message(role="user", content="hi")])

    assert seen["url"].startswith(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    )
    # The key rides the x-goog-api-key header, never the URL - httpx logs the
    # full URL at INFO, so a key in the query string would leak to any log.
    assert seen["header_key"] == API_KEY
    assert seen["url_key"] is None
    assert API_KEY not in seen["url"]
    assert seen["body"]["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]
    assert seen["body"]["generationConfig"]["maxOutputTokens"] == 1024


def test_model_role_maps_through_and_temperature_is_passed() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=OK_PAYLOAD)

    provider = provider_for(handler)
    provider.complete(
        [Message(role="user", content="q"), Message(role="model", content="a")],
        temperature=0.2,
    )

    assert seen["body"]["contents"] == [
        {"role": "user", "parts": [{"text": "q"}]},
        {"role": "model", "parts": [{"text": "a"}]},
    ]
    assert seen["body"]["generationConfig"]["temperature"] == 0.2


def test_assistant_role_maps_to_model() -> None:
    """'assistant' is the near-universal alias; accept it rather than mislabel it."""
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=OK_PAYLOAD)

    provider = provider_for(handler)
    provider.complete([Message(role="assistant", content="a")])
    assert seen["body"]["contents"] == [{"role": "model", "parts": [{"text": "a"}]}]


def test_unknown_role_raises_rather_than_silently_relabelling() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - not reached
        return httpx.Response(200, json=OK_PAYLOAD)

    provider = provider_for(handler)
    with pytest.raises(LLMError, match="unsupported message role"):
        provider.complete([Message(role="tool", content="x")])


# --------------------------------------------------------------------------
# Structured output
# --------------------------------------------------------------------------


def test_json_schema_sets_mime_and_schema_and_returns_json() -> None:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        payload = {
            "candidates": [
                {
                    "content": {"parts": [{"text": '{"answer": "42"}'}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 4},
        }
        return httpx.Response(200, json=payload)

    provider = provider_for(handler)
    resp = provider.complete([Message(role="user", content="q")], json_schema=schema)

    gen = seen["body"]["generationConfig"]
    assert gen["responseMimeType"] == "application/json"
    assert gen["responseSchema"] == schema
    assert json.loads(resp.text) == {"answer": "42"}


def test_no_schema_omits_response_mime_and_schema() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=OK_PAYLOAD)

    provider = provider_for(handler)
    provider.complete([Message(role="user", content="q")])

    gen = seen["body"]["generationConfig"]
    assert "responseMimeType" not in gen
    assert "responseSchema" not in gen


# --------------------------------------------------------------------------
# System instruction folding
# --------------------------------------------------------------------------


def test_system_argument_and_system_message_fold_into_system_instruction() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=OK_PAYLOAD)

    provider = provider_for(handler)
    provider.complete(
        [
            Message(role="system", content="also obey this"),
            Message(role="user", content="hi"),
        ],
        system="be terse",
    )

    body = seen["body"]
    system_text = body["systemInstruction"]["parts"][0]["text"]
    assert "be terse" in system_text
    assert "also obey this" in system_text
    # "system" is never a contents role for Gemini.
    assert all(part["role"] in ("user", "model") for part in body["contents"])
    assert body["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]


def test_no_system_omits_system_instruction() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=OK_PAYLOAD)

    provider = provider_for(handler)
    provider.complete([Message(role="user", content="hi")])
    assert "systemInstruction" not in seen["body"]


# --------------------------------------------------------------------------
# Rate limiting and retries
# --------------------------------------------------------------------------


def _rate_limit_response() -> httpx.Response:
    return httpx.Response(
        429,
        json={
            "error": {
                "code": 429,
                "message": "Resource has been exhausted (e.g. check quota).",
                "status": "RESOURCE_EXHAUSTED",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.RetryInfo",
                        "retryDelay": "7s",
                    }
                ],
            }
        },
    )


def test_429_raises_rate_limit_error_with_retry_after_and_backs_off(
    _no_sleep: list[float],
) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return _rate_limit_response()

    provider = provider_for(handler, max_retries=2)
    with pytest.raises(RateLimitError) as exc_info:
        provider.complete([Message(role="user", content="hi")])

    assert exc_info.value.retry_after == 7.0
    assert calls["n"] == 3  # initial + 2 retries
    assert _no_sleep == [7.0, 7.0]  # honored the server-advised delay each time


def test_429_no_retries_fails_immediately_without_sleeping(_no_sleep: list[float]) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return _rate_limit_response()

    provider = provider_for(handler, max_retries=0)
    with pytest.raises(RateLimitError):
        provider.complete([Message(role="user", content="hi")])

    assert calls["n"] == 1
    assert _no_sleep == []  # never slept


def test_429_then_success_recovers(_no_sleep: list[float]) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return _rate_limit_response()
        return httpx.Response(200, json=OK_PAYLOAD)

    provider = provider_for(handler, max_retries=3)
    resp = provider.complete([Message(role="user", content="hi")])
    assert resp.text == "Hello there"
    assert calls["n"] == 2


def test_5xx_is_retried_then_surfaced(_no_sleep: list[float]) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, json={"error": {"message": "backend unavailable"}})

    provider = provider_for(handler, max_retries=1)
    with pytest.raises(LLMError) as exc_info:
        provider.complete([Message(role="user", content="hi")])

    assert calls["n"] == 2  # initial + 1 retry
    assert not isinstance(exc_info.value, RateLimitError)
    assert "503" in str(exc_info.value)


def test_other_4xx_is_not_retried(_no_sleep: list[float]) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, json={"error": {"message": "bad request"}})

    provider = provider_for(handler, max_retries=3)
    with pytest.raises(LLMError):
        provider.complete([Message(role="user", content="hi")])
    assert calls["n"] == 1  # 4xx is a caller error, not retried


# --------------------------------------------------------------------------
# Safety refusals
# --------------------------------------------------------------------------


def test_safety_finish_reason_raises_refusal() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": []}, "finishReason": "SAFETY"}]},
        )

    provider = provider_for(handler)
    with pytest.raises(LLMRefusal, match="SAFETY"):
        provider.complete([Message(role="user", content="hi")])


def test_prohibited_content_finish_reason_raises_refusal() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": []}, "finishReason": "PROHIBITED_CONTENT"}]},
        )

    provider = provider_for(handler)
    with pytest.raises(LLMRefusal):
        provider.complete([Message(role="user", content="hi")])


def test_blocked_prompt_with_no_candidates_raises_refusal() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}})

    provider = provider_for(handler)
    with pytest.raises(LLMRefusal, match="SAFETY"):
        provider.complete([Message(role="user", content="hi")])


def test_empty_response_with_no_block_reason_is_an_error_not_a_refusal() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    provider = provider_for(handler)
    with pytest.raises(LLMError) as exc_info:
        provider.complete([Message(role="user", content="hi")])
    assert not isinstance(exc_info.value, LLMRefusal)


# --------------------------------------------------------------------------
# The key must never leak
# --------------------------------------------------------------------------


def test_api_key_is_redacted_from_error_messages() -> None:
    # Defense in depth: the key is not in our URLs at all now, but if an upstream
    # error text ever carried a key=... token, redaction must still scrub it.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"error": {"message": f"Invalid request; upstream logged key={API_KEY}"}},
        )

    provider = provider_for(handler)
    with pytest.raises(LLMError) as exc_info:
        provider.complete([Message(role="user", content="hi")])

    message = str(exc_info.value)
    assert API_KEY not in message
    assert "key=REDACTED" in message


def test_missing_api_key_raises_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        GeminiProvider(model="gemini-2.5-flash")


def test_api_key_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("x-goog-api-key") == "from-env"
        assert "from-env" not in str(request.url)
        return httpx.Response(200, json=OK_PAYLOAD)

    provider = GeminiProvider(model="gemini-2.5-flash", http_client=client_for(handler))
    provider.complete([Message(role="user", content="hi")])


# --------------------------------------------------------------------------
# Budget integration
# --------------------------------------------------------------------------


def test_budget_is_checked_before_and_recorded_after(tmp_path: Any) -> None:
    budget = Budget(path=tmp_path / "b.json", max_requests_per_day=5)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=OK_PAYLOAD)

    provider = provider_for(handler, budget=budget)
    provider.complete([Message(role="user", content="hi")])

    assert budget.request_count == 1
    assert budget.input_tokens == 12
    assert budget.output_tokens == 3


def test_budget_cap_blocks_the_call_before_it_is_made(tmp_path: Any) -> None:
    budget = Budget(path=tmp_path / "b.json", max_requests_per_day=1)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=OK_PAYLOAD)

    provider = provider_for(handler, budget=budget)
    provider.complete([Message(role="user", content="hi")])
    with pytest.raises(BudgetExceeded):
        provider.complete([Message(role="user", content="hi")])

    assert calls["n"] == 1  # the second call never reached the transport


def test_budget_counts_a_safety_refusal(tmp_path: Any) -> None:
    """A SAFETY 200 spent real quota; it must count even though it refuses.

    This is the exact workload the attacker harness produces, so an undercount
    here would defeat the guard where it matters most.
    """
    budget = Budget(path=tmp_path / "b.json", max_requests_per_day=5)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": []}, "finishReason": "SAFETY"}],
                "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 0},
            },
        )

    provider = provider_for(handler, budget=budget)
    with pytest.raises(LLMRefusal):
        provider.complete([Message(role="user", content="hi")])

    assert budget.request_count == 1  # counted despite the refusal
    assert budget.input_tokens == 9


def test_budget_counts_every_retry_attempt(tmp_path: Any, _no_sleep: list[float]) -> None:
    """Each 429 retry is its own request against the daily quota."""
    budget = Budget(path=tmp_path / "b.json", max_requests_per_day=10)

    def handler(request: httpx.Request) -> httpx.Response:
        return _rate_limit_response()

    provider = provider_for(handler, budget=budget, max_retries=2)
    with pytest.raises(RateLimitError):
        provider.complete([Message(role="user", content="hi")])

    assert budget.request_count == 3  # initial + 2 retries, all counted


def test_server_retry_delay_is_capped(_no_sleep: list[float]) -> None:
    """A hostile 'retryDelay: 86400s' must not park the client for a day."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={
                "error": {
                    "code": 429,
                    "status": "RESOURCE_EXHAUSTED",
                    "details": [
                        {
                            "@type": "type.googleapis.com/google.rpc.RetryInfo",
                            "retryDelay": "86400s",
                        }
                    ],
                }
            },
        )

    provider = provider_for(handler, max_retries=1)
    with pytest.raises(RateLimitError):
        provider.complete([Message(role="user", content="hi")])

    assert _no_sleep == [60.0]  # capped, not 86400


def test_thinking_tokens_count_toward_output() -> None:
    """gemini-2.5-* bills candidates + thoughts; the budget must see both."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
                "usageMetadata": {
                    "promptTokenCount": 5,
                    "candidatesTokenCount": 3,
                    "thoughtsTokenCount": 8,
                },
            },
        )

    provider = provider_for(handler)
    resp = provider.complete([Message(role="user", content="hi")])
    assert resp.input_tokens == 5
    assert resp.output_tokens == 11  # 3 answer + 8 thinking


def test_provider_satisfies_the_protocol() -> None:
    # A structural sanity check that GeminiProvider is a usable LLMProvider.
    assert hasattr(providers.GeminiProvider, "complete")
