"""Provider-agnostic LLM surface - the vocabulary the rest of Aegis speaks.

WHY A HAND-ROLLED, PROVIDER-NEUTRAL LAYER
-----------------------------------------
Aegis runs several LLMs in *distinct trust roles*, and the whole security
argument rests on those roles being separate principals:

    PRIVILEGED_AGENT  the model that may call tools, reads only trusted intent
    QUARANTINE        an isolated, tool-less extractor that reads untrusted text
    JUDGE             scores attack success / answer quality in the eval harness
    ATTACKER          the red-team model that authors injection attempts

That separation (the dual-LLM / CaMeL pattern that ``aegis.domain.trust``
formalises) is a property of *how we wire the models*, not of any vendor. So the
code models "which role" (:class:`Role`) independently of "which provider/model",
and every caller talks to a narrow :class:`LLMProvider` protocol in our own
types. Swapping Gemini for Claude is then a config edit, and nothing
security-relevant depends on a provider SDK.

WHY NO SDK
----------
Same ethos as the hand-written BM25 index: the wire contract is small, so we own
it. A thin httpx client is auditable end to end, keeps the dependency surface
tiny, and - crucially for the security story - lets us guarantee things an SDK
will not, like *never* letting the API key reach an exception message or a log.

This module is deliberately free of any transport import: it is the shared
contract, and the concrete providers live under ``aegis.llm.providers``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

__all__ = [
    "LLMError",
    "LLMProvider",
    "LLMRefusal",
    "LLMResponse",
    "Message",
    "RateLimitError",
    "Role",
]


class Role(StrEnum):
    """Which principal in the Aegis architecture an LLM call is acting as.

    Values are the stable identifiers used as keys in ``config/models.yaml``, so
    a role can be named in config and parsed straight back to a member.
    """

    PRIVILEGED_AGENT = "privileged_agent"
    QUARANTINE = "quarantine"
    JUDGE = "judge"
    ATTACKER = "attacker"


@dataclass(frozen=True, slots=True)
class Message:
    """One turn of a conversation, in provider-neutral form.

    ``role`` is the *chat* role - ``"user"``, ``"model"``, or ``"system"`` - and
    is intentionally a plain string, not :class:`Role`: it describes who spoke in
    the transcript, which is a different axis from which Aegis principal is
    driving the call. Providers map these onto their own wire vocabulary (Gemini,
    for instance, has no ``"system"`` content role - see ``GeminiProvider``).
    """

    role: str
    content: str


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """The normalised result of a single completion.

    ``raw`` keeps the full parsed provider payload so a failing eval run can be
    debugged without re-issuing (and re-paying for) the call.
    """

    text: str
    model: str
    input_tokens: int
    output_tokens: int
    finish_reason: str
    raw: dict[str, object]


class LLMError(Exception):
    """Any failure originating from the LLM layer.

    Providers must ensure the message is safe to log: in particular it must never
    contain the API key (see the redaction in ``GeminiProvider``).
    """


class RateLimitError(LLMError):
    """The provider throttled us (HTTP 429 / RESOURCE_EXHAUSTED).

    ``retry_after`` carries the server-advised wait in seconds when the provider
    reports one (Gemini returns it as a ``RetryInfo`` detail), else ``None``.
    """

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class LLMRefusal(LLMError):
    """The provider blocked the request or response on safety grounds.

    Raised for a blocked prompt (``promptFeedback.blockReason``) or a candidate
    that finished on ``SAFETY`` / ``PROHIBITED_CONTENT`` / ``BLOCKLIST``. It is a
    distinct type because a refusal is a *policy* outcome, not a transport fault:
    a caller (e.g. the attacker-simulation harness) may legitimately expect and
    tolerate it, whereas a bare :class:`LLMError` signals something went wrong.
    """


class LLMProvider(Protocol):
    """The one method every provider implements.

    Keeping this a ``Protocol`` (structural, not inherited) is what lets the
    offline :class:`~aegis.llm.providers.fake.FakeProvider` and the live
    ``GeminiProvider`` be used interchangeably by the router and the tests
    without a shared base class.
    """

    def complete(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        max_output_tokens: int = 1024,
        temperature: float | None = None,
        json_schema: Mapping[str, object] | None = None,
    ) -> LLMResponse:
        """Produce a completion for ``messages``.

        When ``json_schema`` is given the provider must return ``text`` that is
        valid JSON conforming to that schema (providers request constrained
        decoding where the API supports it).
        """
        ...
