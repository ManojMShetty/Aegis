"""The primary Week-0 agent-under-test: AgentDojo's own OpenAILLM, over any
OpenAI-compatible endpoint (Groq's by default), with spend counters and two
endpoint compatibility guards bolted on.

WHY THIS PATH IS THE DEFAULT, AND WHY IT REUSES THE STOCK ELEMENT
-----------------------------------------------------------------
The Week-0 number is a *baseline*, and a baseline is only trustworthy if the agent
plumbing under it is battle-tested rather than bespoke. AgentDojo ships
``OpenAILLM``, and every endpoint we point it at (Groq's
``https://api.groq.com/openai/v1``, NVIDIA's ``https://integrate.api.nvidia.com/v1``)
speaks the OpenAI wire protocol, so the whole multi-turn tool loop - message
conversion, tool-call round-tripping, retries - is the stock, widely-exercised
code. Unlike Gemini 3.x (which needs the ``thought_signature`` fix in
:mod:`evals.agentdojo.gemini_llm`), an open model behind an OpenAI-compatible API
needs no protocol surgery at all. So this path adds nothing to the model behaviour;
it only wraps the endpoint, counts the traffic, and works around two documented
endpoint quirks.

THE TWO COUNTERS, AND WHY THERE ARE EXACTLY TWO
------------------------------------------------
``OpenAILLM`` has no counters at all, but the runner's frugality report needs
numbers that mean the same thing on every provider. Two units exist and they are
NOT interchangeable, so both are exposed under distinct names rather than one field
that quietly means something different per provider:

* :attr:`CountingOpenAILLM.call_count` - model TURNS. Bumped once per
  :meth:`CountingOpenAILLM.query`, i.e. once per distinct agent decision. This is
  the unit that answers "how many times did the agent think".
* :attr:`CountingOpenAILLM.request_count` - API REQUEST ATTEMPTS. Bumped inside the
  patched ``chat.completions.create``, so it also counts the retries AgentDojo's
  ``chat_completion_request`` makes: that function wraps the create in a tenacity
  ``stop_after_attempt(3)``, so a single turn can issue up to three attempts. Every
  attempt reaches the endpoint and spends quota, so this is the number that answers
  "how much did the run actually cost".

:mod:`evals.agentdojo.gemini_llm` exposes the same two attributes with the same two
meanings, which is what makes ``total_model_calls`` (attempts) and
``total_model_turns`` (turns) comparable across providers in the result JSON.

Neither counter sees the SDK's own transport-level retry (the client is built with
``max_retries=5``, which retries *inside* a single ``create`` call, honouring
``Retry-After``). Those are bounded and invisible at this seam, and the gemini path
cannot see its SDK's internal retries either, so both providers under-report the
same way and stay comparable to each other.

THE TWO ENDPOINT GUARDS, AND WHY THEY ARE NOT SYMMETRIC
-------------------------------------------------------
Free tiers are rate limited, so :func:`_install_endpoint_guards` spaces requests
(see :data:`DEFAULT_MIN_REQUEST_INTERVAL_S`) for EVERY endpoint - it costs nothing
but wall-clock, and it is the difference between a run that finishes and a run that
dies in a 429 storm. Single-tool-call truncation, by contrast, is applied to NVIDIA
ONLY, because it is a real (if small) behavioural change and Groq does not need it;
see :func:`_install_endpoint_guards` for the failure it works around.

REASONING EFFORT
----------------
The recorded Week-0 baseline runs ``openai/gpt-oss-120b``, a reasoning model, at
``reasoning_effort="low"``: full-effort reasoning multiplies latency (and token
spend) per turn to the point where a multi-task AgentDojo run stops being feasible
on a free tier, while "low" still solved every user task in the recorded run. The
value is validated here rather than at the endpoint so a typo fails in the first
millisecond of the run instead of after the first billed request.

SECRETS
-------
The API key is read only from the environment variable the caller names (default
``GROQ_API_KEY`` at the runner level), never hardcoded, never logged, and never
placed in an error message: a missing-key error names the *variable to set*, never
a value. The base URL is an endpoint, carries no secret, and is safe to record -
and, unlike the key, safe to inspect (see :func:`_needs_single_tool_call`).
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from typing import Any, Final

import openai
from agentdojo.agent_pipeline import OpenAILLM
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage

__all__ = [
    "DEFAULT_MIN_REQUEST_INTERVAL_S",
    "REASONING_EFFORT_CHOICES",
    "CountingOpenAILLM",
    "OpenAICompatLLMError",
    "build_openai_compat_llm",
]

# The reasoning-effort values the OpenAI wire protocol accepts, mirroring the SDK's
# ``ChatCompletionReasoningEffort`` literal. Kept as a plain tuple (not the imported
# Literal) so it can be handed straight to argparse ``choices`` and listed in an
# error message; drift from the SDK would surface as a value rejected at build
# time, never as a silently mis-sent request.
REASONING_EFFORT_CHOICES: Final[tuple[str, ...]] = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)

# Minimum spacing between requests, to stay under a free-tier requests-per-minute
# limit. A fast model bursting through an AgentDojo run trips that limit easily
# (NVIDIA's free endpoint 429s llama-3.1-8b, which answers in ~1s); ~1.2s spacing
# keeps us near 50 req/min. This is a coarse guard - the SDK's Retry-After-aware
# retry (``max_retries``) absorbs any 429 that still slips through.
DEFAULT_MIN_REQUEST_INTERVAL_S: Final[float] = 1.2

# The coarse provider fingerprint that decides whether single-tool-call truncation
# is installed. See :func:`_needs_single_tool_call` for why a substring test on the
# base URL is acceptable here.
_SINGLE_TOOL_CALL_ENDPOINT_MARKER: Final[str] = "nvidia"


class OpenAICompatLLMError(RuntimeError):
    """Configuration error building the element - notably a missing API key.

    A named type, mirroring :class:`~evals.agentdojo.gemini_llm.Gemini3LLMError`,
    so a "you did not give me a key" failure is distinguishable from a transport
    error and assertable in a test. The key value is never part of this text.
    """


class _RequestCounter:
    """A one-field mutable counter shared by the client guard and the element.

    The endpoint guard is a closure installed on the CLIENT, which has to exist
    before the element that wraps it, so the two cannot simply share ``self``. A
    tiny counter object is created first and handed to both, which is what lets the
    element report attempts that were actually observed at the wire seam instead of
    inferring them from turns.
    """

    count: int

    def __init__(self) -> None:
        self.count = 0


class CountingOpenAILLM(OpenAILLM):  # type: ignore[misc]  # OpenAILLM is untyped (agentdojo has no py.typed)
    """AgentDojo's ``OpenAILLM`` plus :attr:`call_count` and :attr:`request_count`.

    Identical to the stock element in every behavioural respect - the entire tool
    loop is inherited unchanged - except that each :meth:`query` increments
    :attr:`call_count` before delegating. :attr:`request_count` is read from the
    counter the endpoint guard bumps, so it counts request ATTEMPTS (AgentDojo's
    tenacity retries included) rather than turns. See the module docstring for what
    each unit does and does not include.
    """

    call_count: int

    def __init__(
        self,
        client: openai.OpenAI,
        model: str,
        *,
        temperature: float | None = 0.0,
        reasoning_effort: str | None = None,
        request_counter: _RequestCounter | None = None,
    ) -> None:
        """Wrap ``client``/``model``; ``reasoning_effort=None`` omits the field.

        ``reasoning_effort`` is stored by the stock element and sent on every
        request; ``None`` means "do not send it at all", which is what a
        non-reasoning model wants. The value is NOT validated here - direct
        construction with a hand-made client (as in the tests) stays unencumbered;
        the checked entry point is :func:`build_openai_compat_llm`.

        ``request_counter`` is the counter the client's endpoint guard bumps;
        :func:`build_openai_compat_llm` builds the guard and the element together
        and threads it through. Omitting it - direct construction against an
        unguarded client - leaves :attr:`request_count` at 0, which is honest: with
        no guard installed, nothing is watching the wire.
        """
        super().__init__(
            client=client,
            model=model,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
        )
        self.call_count = 0
        self._request_counter = (
            request_counter if request_counter is not None else _RequestCounter()
        )
        # A non-None name so `AgentPipeline.from_config` has a name to log under
        # even before the runner overrides it.
        self.name = f"openai-compat-{model}"

    @property
    def request_count(self) -> int:
        """API request attempts seen at the client seam - retries included."""
        return self._request_counter.count

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env | None = None,
        messages: Sequence[ChatMessage] | None = None,
        extra_args: dict[str, Any] | None = None,
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict[str, Any]]:
        """Count this model turn, then run the stock query.

        The count is bumped BEFORE delegating so a turn whose request reaches the
        server and then errors is still counted - a failed attempt spends quota
        just like a successful one. The base signature's mutable defaults
        (``EmptyEnv()``, ``[]``, ``{}``) are replaced with ``None`` sentinels and
        materialised here; AgentDojo always passes these positionally, so behaviour
        is unchanged.
        """
        if env is None:
            env = EmptyEnv()
        if messages is None:
            messages = []
        if extra_args is None:
            extra_args = {}
        self.call_count += 1
        result: tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict[str, Any]] = (
            super().query(query, runtime, env, messages, extra_args)
        )
        return result


def _needs_single_tool_call(base_url: str) -> bool:
    """Does this endpoint need assistant turns truncated to a single tool call?

    Answered by substring-matching :data:`_SINGLE_TOOL_CALL_ENDPOINT_MARKER` against
    the base URL. That is deliberately coarse, and it is acceptable here for one
    reason: this is a COMPATIBILITY SHIM, NOT A SECURITY BOUNDARY. A false positive
    costs a slightly different tool-call cadence (sequential instead of parallel);
    a false negative costs a loud HTTP 500. Neither grants a capability, hides an
    input, or changes what the agent is allowed to do - nothing downstream trusts
    this answer. The base URL is also operator-supplied configuration, not
    attacker-controlled content. The precise alternative - probing the endpoint for
    its capabilities - would spend a live request to learn something we already know
    from the deployment we chose to point at.
    """
    return _SINGLE_TOOL_CALL_ENDPOINT_MARKER in base_url.lower()


def _install_endpoint_guards(
    client: openai.OpenAI,
    *,
    truncate_tools: bool,
    min_request_interval_s: float,
) -> _RequestCounter:
    """Patch the client's ``chat.completions.create`` in place with endpoint guards.

    Returns the :class:`_RequestCounter` the patched create bumps, so the element
    built around this client can report attempts it never sees itself.

    (1) REQUEST COUNT (always). Every call through the patched create is counted,
    which is precisely why the count includes AgentDojo's tenacity retries: a retry
    re-enters ``chat.completions.create`` and therefore re-enters this closure. The
    bump happens before the request is issued, for the same reason the turn counter
    bumps first - an attempt that errors in flight still spent quota.

    (2) RATE THROTTLE (on unless disabled). A fast model bursts past the free-tier
    RPM limit; we space calls by ``min_request_interval_s`` so a 429 is rare, and
    rely on the SDK's Retry-After-aware retry for the rest. A value <= 0 disables
    the throttle entirely - no clock is read and ``time.sleep`` is never called,
    which is what an offline test (or a paid tier with a generous limit) wants. The
    first call through a fresh guard never waits: there is no earlier request to
    space it from. The timestamp is stamped BEFORE the request goes out, never after
    it returns, so the spacing measured is request-START to request-START - a real
    requests-per-minute guard. Stamping after the response would instead impose a
    fixed GAP between requests, throttling a slow model twice over.

    (3) SINGLE TOOL CALL (only when ``truncate_tools``). NVIDIA's NIM prompt template
    for some models (notably meta/llama-3.1-8b-instruct) cannot represent an assistant
    turn that carries more than one ``tool_call``, and fails the *next* request with
    ``500 ... This model only supports single tool-calls at once!`` once such a turn is
    in the history. The endpoint also ACCEPTS BUT SILENTLY IGNORES
    ``parallel_tool_calls=False``, so the only reliable remedy is to keep just the
    first tool call in every response - converting parallel tool-calling into
    sequential tool-calling (the agent re-requests further tools on later turns). This
    changes the tool-call *cadence*, not the task semantics. Other OpenAI-compatible
    endpoints (Groq's, notably) handle multi-tool-call history fine, so they leave it
    off and let a capable model use parallel tool calls naturally - which is why this
    is a parameter and not an unconditional fix.

    We patch the instance's ``chat.completions.create`` (a cached resource, so the
    patch persists) rather than AgentDojo's request function, keeping the change local
    to this client and leaving the stock element untouched.
    """
    original_create = client.chat.completions.create
    throttled = min_request_interval_s > 0
    last_call_at: float | None = None
    counter = _RequestCounter()

    def _create(*args: Any, **kwargs: Any) -> Any:
        nonlocal last_call_at
        if throttled:
            if last_call_at is not None:
                wait = min_request_interval_s - (time.monotonic() - last_call_at)
                if wait > 0:
                    time.sleep(wait)
            last_call_at = time.monotonic()
        counter.count += 1
        response = original_create(*args, **kwargs)
        if truncate_tools:
            for choice in getattr(response, "choices", None) or []:
                message = getattr(choice, "message", None)
                tool_calls = getattr(message, "tool_calls", None)
                if message is not None and tool_calls and len(tool_calls) > 1:
                    message.tool_calls = tool_calls[:1]
        return response

    client.chat.completions.create = _create  # type: ignore[method-assign]
    return counter


def _validate_reasoning_effort(reasoning_effort: str | None) -> None:
    """Reject an effort level the wire protocol does not accept, at build time.

    ``None`` is allowed and means "omit the field". Anything else must be one of
    :data:`REASONING_EFFORT_CHOICES`. The rejected value IS quoted in the error: it
    is a configuration enum member, never secret, and naming it is the difference
    between a fixable typo and a mystery. Failing here rather than at the endpoint
    means a bad value costs zero requests.
    """
    if reasoning_effort is None:
        return
    if reasoning_effort not in REASONING_EFFORT_CHOICES:
        allowed = ", ".join(REASONING_EFFORT_CHOICES)
        raise OpenAICompatLLMError(
            f"invalid reasoning_effort {reasoning_effort!r}; allowed values are: {allowed} "
            "(or None to omit the field entirely)"
        )


def build_openai_compat_llm(
    *,
    model: str,
    base_url: str,
    api_key_env: str,
    timeout: float,
    reasoning_effort: str | None = None,
    min_request_interval_s: float = DEFAULT_MIN_REQUEST_INTERVAL_S,
) -> CountingOpenAILLM:
    """Read the key from ``api_key_env`` and build a counted OpenAI-compatible LLM.

    The key is read ONLY from the named environment variable; if it is unset (or
    empty) we fail with a message that names the variable to set and never contains
    a key value. The client is bounded on purpose: an explicit ``timeout`` plus
    ``max_retries=5``, so a cold or oversized model cannot hang the run forever while
    a rate-limited one still gets its Retry-After-aware retries.

    ``reasoning_effort`` is validated against :data:`REASONING_EFFORT_CHOICES` before
    any client is built (``None`` omits the field). ``min_request_interval_s`` is the
    minimum spacing between requests, stored on the guard closure; pass ``0`` to
    disable throttling entirely.

    This is also the only place that sees both the client and the element, so it is
    where the guard's request counter is joined to the element that reports it.
    """
    _validate_reasoning_effort(reasoning_effort)
    key = os.environ.get(api_key_env)
    if not key:
        raise OpenAICompatLLMError(
            f"no API key: set the {api_key_env} environment variable "
            "(it lives in .env, which is gitignored). "
            "This error intentionally contains no key value."
        )
    # max_retries=5: the SDK retries transient errors honoring the 429 Retry-After
    # header, which is the correct primary defense against a free-tier rate limit
    # (the throttle in the guard reduces how often we hit it in the first place).
    client = openai.OpenAI(
        base_url=base_url,
        api_key=key,
        timeout=timeout,
        max_retries=5,
    )
    # Throttling and counting apply to every endpoint; single-tool-call truncation
    # only to the one that cannot represent a multi-tool assistant turn.
    counter = _install_endpoint_guards(
        client,
        truncate_tools=_needs_single_tool_call(base_url),
        min_request_interval_s=min_request_interval_s,
    )
    return CountingOpenAILLM(
        client=client,
        model=model,
        reasoning_effort=reasoning_effort,
        request_counter=counter,
    )
