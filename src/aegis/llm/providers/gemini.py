"""Gemini provider - a thin, auditable httpx client for generateContent.

WHY httpx AND NOT THE SDK
-------------------------
The Gemini REST contract for a completion is a single POST with a small JSON
body. Owning that call outright (the same choice as the hand-written BM25 index)
buys three things an SDK would not: a dependency surface of one well-understood
library, a wire shape we can read in review, and - the point that actually
matters for a security project - a hard guarantee about the API key. The key
travels in the ``x-goog-api-key`` header, never in the URL, so it cannot leak
through httpx's own request logging (which prints the full URL at INFO) or a
proxy access log; as defense in depth it is also scrubbed from every error this
module can raise.

BUDGET ACCOUNTING IS PER ATTEMPT, NOT PER SUCCESS
-------------------------------------------------
Every attempt that reaches the server draws on the daily quota - including a
retried 429 and a SAFETY refusal (a real HTTP 200 that still spends quota). So
the request is counted in the transport loop for each server response, not once
per :meth:`complete`, or the guard would undercount exactly the refusal- and
retry-heavy attacker workload it exists to bound.

WHY THE RETRY / REFUSAL HANDLING IS EXPLICIT
--------------------------------------------
On a free tier, 429s are normal, not exceptional. The client honours the
server's advised ``retryDelay`` when present and backs off exponentially
otherwise, retrying transient 429/5xx up to a cap and then surfacing a typed
:class:`RateLimitError`. Safety blocks are surfaced as :class:`LLMRefusal`
because they are a policy outcome a caller may expect (the attacker-simulation
harness routinely trips them), not a bug.

The exact request/response and 429 ``RetryInfo`` shapes were checked against the
current Google AI docs rather than recalled from memory.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Mapping, Sequence

import httpx

from aegis.llm.base import (
    LLMError,
    LLMRefusal,
    LLMResponse,
    Message,
    RateLimitError,
)
from aegis.llm.budget import Budget

__all__ = ["GeminiProvider"]

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

# finishReason values that mean the model refused rather than answered.
_REFUSAL_FINISH_REASONS = frozenset({"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST"})

# Anything that looks like an API key in a URL, so it never reaches a log/trace.
_KEY_IN_URL = re.compile(r"(key=)[^&\s]+", re.IGNORECASE)

# A protobuf Duration string, e.g. "34s" or "1.5s".
_DURATION = re.compile(r"\s*([0-9]+(?:\.[0-9]+)?)s\s*")


def _redact(text: str) -> str:
    """Strip any ``key=...`` value from a string bound for an error or log."""
    return _KEY_IN_URL.sub(r"\1REDACTED", text)


class GeminiProvider:
    """Calls Google's Gemini ``generateContent`` endpoint.

    Satisfies :class:`~aegis.llm.base.LLMProvider` structurally.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        *,
        budget: Budget | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        base_url: str = DEFAULT_BASE_URL,
        http_client: httpx.Client | None = None,
    ) -> None:
        key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY")
        if not key:
            raise LLMError(
                "no Gemini API key: pass api_key= or set the GEMINI_API_KEY environment "
                "variable (it lives in .env, which is gitignored)"
            )
        self._model = model
        self._api_key = key
        self._budget = budget
        self._timeout = timeout
        self._max_retries = max_retries
        self._base_url = base_url.rstrip("/")
        # Inject a client for tests (httpx.MockTransport); otherwise own one.
        self._client = http_client if http_client is not None else httpx.Client(timeout=timeout)
        self._owns_client = http_client is None

    def close(self) -> None:
        """Close the underlying client if this provider created it."""
        if self._owns_client:
            self._client.close()

    # -- LLMProvider -----------------------------------------------------

    def complete(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        max_output_tokens: int = 1024,
        temperature: float | None = None,
        json_schema: Mapping[str, object] | None = None,
    ) -> LLMResponse:
        url = f"{self._base_url}/models/{self._model}:generateContent"
        body = _build_body(messages, system, max_output_tokens, temperature, json_schema)
        response = self._post_with_retries(url, body)
        # Budget accounting happens inside the loop, per server response; by here
        # the request has already been counted (even if it turns out to refuse).
        return _parse_response(response, self._model)

    # -- transport -------------------------------------------------------

    def _post_with_retries(self, url: str, body: dict[str, object]) -> httpx.Response:
        last_error: LLMError | None = None
        for attempt in range(self._max_retries + 1):
            # Fail-closed before every real request, not just the first: a retry
            # is itself a quota-spending call.
            if self._budget is not None:
                self._budget.check()
            try:
                response = self._client.post(
                    url,
                    headers={"x-goog-api-key": self._api_key},
                    json=body,
                    timeout=self._timeout,
                )
            except httpx.TimeoutException as exc:
                last_error = LLMError(f"Gemini request timed out: {_redact(str(exc))}")
                if attempt < self._max_retries:
                    self._backoff(attempt, None)
                    continue
                raise last_error from exc
            except httpx.HTTPError as exc:
                # Transport-level failure (connection reset, etc.): retry.
                last_error = LLMError(f"Gemini request failed: {_redact(str(exc))}")
                if attempt < self._max_retries:
                    self._backoff(attempt, None)
                    continue
                raise last_error from exc

            # This attempt reached the server, so it spent quota - count it now,
            # before any status branch or refusal parse can bypass the accounting.
            # Tokens are only present on a 200 (incl. blocked 200s, which report
            # usageMetadata); other statuses count as a request with zero tokens.
            if self._budget is not None:
                in_tok, out_tok = _usage_tokens(response) if response.status_code == 200 else (0, 0)
                self._budget.record_request(input_tokens=in_tok, output_tokens=out_tok)

            status = response.status_code
            if status == 200:
                return response

            if status == 429:
                retry_after = _parse_retry_delay(response)
                last_error = RateLimitError(
                    f"Gemini rate limited (429): {_redact(_error_message(response))}",
                    retry_after=retry_after,
                )
                if attempt < self._max_retries:
                    self._backoff(attempt, retry_after)
                    continue
                raise last_error

            if 500 <= status < 600:
                last_error = LLMError(
                    f"Gemini server error ({status}): {_redact(_error_message(response))}"
                )
                if attempt < self._max_retries:
                    self._backoff(attempt, None)
                    continue
                raise last_error

            # Any other 4xx is a caller/config problem - do not retry.
            raise LLMError(f"Gemini request failed ({status}): {_redact(_error_message(response))}")

        # Loop only exits via return/raise above; this satisfies the type checker.
        raise last_error or LLMError("Gemini request failed with no response")

    def _backoff(self, attempt: int, retry_after: float | None) -> None:
        """Sleep before the next attempt, preferring the server's advice.

        The server-advised delay is capped too: a hostile or buggy 429 carrying
        ``retryDelay: "86400s"`` must not park the eval for a day.
        """
        # Cap both paths: a server-advised delay at 60s, our own backoff at 30s.
        delay = min(retry_after, 60.0) if retry_after is not None else min(2.0**attempt, 30.0)
        time.sleep(delay)


# ---------------------------------------------------------------------------
# request construction
# ---------------------------------------------------------------------------


def _build_body(
    messages: Sequence[Message],
    system: str | None,
    max_output_tokens: int,
    temperature: float | None,
    json_schema: Mapping[str, object] | None,
) -> dict[str, object]:
    """Assemble the generateContent request body.

    Gemini's ``contents`` accepts only ``"user"`` / ``"model"`` roles; there is
    no ``"system"`` content role. A ``system=`` argument and any ``Message`` with
    ``role == "system"`` are therefore folded into ``systemInstruction``.
    """
    system_texts: list[str] = []
    if system:
        system_texts.append(system)

    contents: list[dict[str, object]] = []
    for msg in messages:
        if msg.role == "system":
            system_texts.append(msg.content)
            continue
        contents.append({"role": _content_role(msg.role), "parts": [{"text": msg.content}]})

    body: dict[str, object] = {"contents": contents}
    if system_texts:
        body["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_texts)}]}

    generation_config: dict[str, object] = {"maxOutputTokens": max_output_tokens}
    if temperature is not None:
        generation_config["temperature"] = temperature
    if json_schema is not None:
        generation_config["responseMimeType"] = "application/json"
        generation_config["responseSchema"] = dict(json_schema)
    body["generationConfig"] = generation_config
    return body


def _content_role(role: str) -> str:
    """Map a Message role to a Gemini ``contents`` role.

    Gemini accepts only ``user`` and ``model``. We accept ``assistant`` as an
    alias for ``model`` (the near-universal convention), and refuse anything
    else rather than silently relabelling it - misattributing who said what
    would quietly corrupt the transcript a trust-boundary system reasons over.
    """
    if role in ("model", "assistant"):
        return "model"
    if role == "user":
        return "user"
    raise LLMError(
        f"unsupported message role {role!r}; expected one of: user, model, assistant, system"
    )


# ---------------------------------------------------------------------------
# response parsing
# ---------------------------------------------------------------------------


def _parse_response(response: httpx.Response, model: str) -> LLMResponse:
    data = _as_mapping(_safe_json(response))

    # A prompt blocked before generation reports why and returns no candidates.
    feedback = _as_mapping(data.get("promptFeedback"))
    block_reason = _as_str(feedback.get("blockReason"))

    candidates = _as_sequence(data.get("candidates"))
    if not candidates:
        if block_reason:
            raise LLMRefusal(f"Gemini blocked the prompt: blockReason={block_reason}")
        raise LLMError("Gemini returned no candidates and no block reason")

    candidate = _as_mapping(candidates[0])
    finish_reason = _as_str(candidate.get("finishReason"))
    if finish_reason in _REFUSAL_FINISH_REASONS:
        raise LLMRefusal(f"Gemini blocked the response: finishReason={finish_reason}")

    content = _as_mapping(candidate.get("content"))
    parts = _as_sequence(content.get("parts"))
    text = "".join(_as_str(_as_mapping(part).get("text")) for part in parts)

    in_tok, out_tok = _tokens_from(data)
    return LLMResponse(
        text=text,
        model=model,
        input_tokens=in_tok,
        output_tokens=out_tok,
        finish_reason=finish_reason,
        raw=dict(data),
    )


def _tokens_from(data: Mapping[str, object]) -> tuple[int, int]:
    """(prompt_tokens, output_tokens) from usageMetadata, safe on any shape.

    Output includes ``thoughtsTokenCount``: on gemini-2.5-* models the billed
    and quota-charged output is candidate tokens plus thinking tokens, so the
    budget must count both or it undercounts spend on reasoning-heavy calls.
    """
    usage = _as_mapping(data.get("usageMetadata"))
    prompt = _as_int(usage.get("promptTokenCount"))
    out = _as_int(usage.get("candidatesTokenCount")) + _as_int(usage.get("thoughtsTokenCount"))
    return prompt, out


def _usage_tokens(response: httpx.Response) -> tuple[int, int]:
    """(prompt_tokens, output_tokens) straight from an HTTP response body."""
    return _tokens_from(_as_mapping(_safe_json(response)))


def _parse_retry_delay(response: httpx.Response) -> float | None:
    """Pull ``retryDelay`` out of a 429's ``RetryInfo`` detail, if present."""
    error = _as_mapping(_as_mapping(_safe_json(response)).get("error"))
    for detail in _as_sequence(error.get("details")):
        detail_map = _as_mapping(detail)
        if _as_str(detail_map.get("@type")).endswith("RetryInfo"):
            match = _DURATION.fullmatch(_as_str(detail_map.get("retryDelay")))
            if match:
                return float(match.group(1))
    return None


def _error_message(response: httpx.Response) -> str:
    """A human-readable message from an error body, falling back to the text."""
    error = _as_mapping(_as_mapping(_safe_json(response)).get("error"))
    message = _as_str(error.get("message"))
    return message or response.text


def _safe_json(response: httpx.Response) -> object:
    try:
        return response.json()
    except ValueError:
        return {}


# -- defensive accessors: provider JSON is untrusted-shape, so narrow at the edge


def _as_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _as_sequence(value: object) -> Sequence[object]:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return value
    return ()


def _as_str(value: object) -> str:
    return value if isinstance(value, str) else ""


def _as_int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0
