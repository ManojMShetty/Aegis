"""A Gemini-3.x-compatible AgentDojo pipeline element: preserve thought_signature.

WHY THIS MODULE HAS TO EXIST AT ALL
-----------------------------------
AgentDojo already ships ``GoogleLLM``. We do not replace it for fun - we replace
one behaviour in it, because that one behaviour makes every multi-turn tool loop
fail on Gemini 3.x.

THE BUG, EXACTLY
----------------
Gemini 3.x returns, on the assistant turn that requests a tool call, a
per-function-call ``thought_signature`` (an opaque ``bytes`` blob riding on the
``Part`` that carries the ``functionCall``). The model requires that signature to
be handed straight back on the *next* request, still attached to the same
function-call part, or it rejects the turn with
``400 ... Function call is missing a thought_signature in functionCall parts``.

The stock ``GoogleLLM`` cannot do this. AgentDojo stores the conversation as its
own ``ChatMessage`` objects, and a ``ChatAssistantMessage`` keeps only the
function *name* and *args* - the signature has nowhere to live. So when the stock
element rebuilds the assistant turn for the next request it calls
``Part.from_function_call(name=..., args=...)``, which mints a *bare* part whose
``thought_signature`` is ``None``. The first turn works; the second turn - the
one that echoes the tool call back alongside the tool result - always 400s.

THE FIX, EXACTLY
----------------
When echoing an assistant tool-call turn back to the model, re-emit the ORIGINAL
``genai`` ``Content`` the model returned - the object that still carries the
signature - instead of rebuilding a part from name and args. This is the same
move the verified minimal reproduction makes (``contents.append(r1.candidates[0]
.content)``): hand back exactly what came out, byte-for-byte.

WHERE THE ORIGINAL CONTENT IS KEPT, AND WHY THE KEY IS WHAT IT IS
-----------------------------------------------------------------
AgentDojo re-invokes this element every turn with only its own ``ChatMessage``
history - the genai objects are not in it. So on every call that produces tool
calls we stash the returned ``Content`` in a per-instance dict, keyed by
``(assistant-message position, tool-call fingerprint)``. On a later turn, when we
are asked to convert that same assistant message back to genai, we recompute the
key and, on a hit, return the stashed original. On a miss - a turn we did not
produce, or one that carried no signature - we fall back to the stock conversion,
so nothing that worked before breaks.

Two subtleties, each a bug a prior review flagged, decide the exact key:

* AgentDojo's ``ToolsExecutor`` rewrites ``tool_call.args[k]`` IN PLACE between
  turns - a string-encoded list (``"['a', 'b']"``) becomes a real list
  (``['a', 'b']``) via ``is_string_list`` / ``literal_eval`` - AFTER we fingerprint
  the original and BEFORE the next turn fingerprints the mutated value. A naive
  fingerprint would miss, fall back to the bare rebuild, and 400 - killing the
  whole run, because ``agentdojo.benchmark`` does not catch that 400. We apply the
  identical normalization inside the fingerprint so both sides agree no matter
  which side of the mutation they see.

* Gemini usually returns ``id=None`` on its function calls, so two byte-identical
  call turns in one conversation would fingerprint identically and collapse to a
  single stash entry (the later overwriting the earlier), making both history
  turns echo the wrong signature. Including the assistant message's position in
  the conversation in the key keeps them distinct. The position is available on
  both sides: ``len(other_messages)`` when we stash (the new turn is about to be
  appended there), and the ``enumerate`` index when we rebuild.

The stash is cleared at the start of each conversation (a history that carries no
assistant message yet). That both stops a stale entry from a previous task
colliding with this one and bounds the dict's growth across a whole benchmark run.

WHY A CONTENT FINGERPRINT AND NOT OBJECT IDENTITY
-------------------------------------------------
``id(message)`` would also work while the object is alive, but it is invisible in
a test and quietly wrong if a dict is garbage-collected and its id reused across
tasks. A value fingerprint is deterministic, survives across the whole run, and
is exactly what an offline two-turn test can assert against.

RETRIES, AND THE TWO COUNTERS THEY FORCE APART
----------------------------------------------
The stock ``GoogleLLM`` routes its send through ``chat_completion_request``, which
carries a tenacity ``@retry`` - three attempts, exponential backoff, and, crucially,
``retry_if_not_exception_type(ClientError)`` so a deterministic 4xx is never
retried. Calling ``generate_content`` directly would drop that policy, and a
transient ``ServerError`` would then be scored ``utility=False`` / ``security=True``
- a flaky 5xx recorded as a SUCCESSFUL ATTACK, silently inflating ASR. We reuse the
identical policy here.

That retry is exactly why one number is not enough, and why this element exposes
the same two counters as :mod:`evals.agentdojo.openai_llm`, with the same meanings:

* :attr:`Gemini3LLM.call_count` - model TURNS, bumped once per
  :meth:`Gemini3LLM.query`. One turn is one agent decision, however many attempts
  it took to get an answer out of the endpoint.
* :attr:`Gemini3LLM.request_count` - API REQUEST ATTEMPTS, bumped INSIDE the retried
  body so every attempt (each of which spends quota) is counted, not just the first.

A single field that means turns on one provider and attempts on the other is how
two runs come to look comparable when they are not, so the runner records both under
distinct names (``total_model_calls`` = attempts, ``total_model_turns`` = turns) and
both elements agree on which is which.

SCOPE
-----
This element is the baseline agent's brain and nothing more. It has no Aegis
defense in it and makes no trust-tier judgements - it is the *undefended* model
the Week-0 number is taken against. The only Aegis-flavoured discipline it keeps
is around the key: it is read solely from the environment variable the caller
names (default ``GEMINI_API_KEY``, which is what :func:`build_gemini_llm` and the
bare constructor fall back to), never logged, and never placed in an error message.
A named variable is honoured strictly - there is no fallback to a different one,
because a run executed with a key the operator did not ask for is not the run they
asked for.
"""

from __future__ import annotations

import json
import os
from ast import literal_eval
from collections.abc import Sequence
from typing import Any

from agentdojo.agent_pipeline import GoogleLLM
from agentdojo.agent_pipeline.llms.google_llm import (
    _function_to_google,
    _google_to_assistant_message,
    _merge_tool_result_messages,
    _message_to_google,
)
from agentdojo.agent_pipeline.tool_execution import is_string_list
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

__all__ = ["Gemini3LLM", "Gemini3LLMError", "build_gemini_llm"]

# The environment variable the key comes from when the caller names no other.
# Named once so the "where does the key live" answer is greppable and never drifts.
DEFAULT_API_KEY_ENV = "GEMINI_API_KEY"

# A fingerprint over one assistant turn's tool calls: (name, id, canonical args)
# per call. Hashable, deterministic, and independent of object identity.
_CallKey = tuple[tuple[str, str | None, str], ...]

# The stash key pairs that fingerprint with the assistant message's position in
# the conversation, so two identical (often id=None) call turns stay distinct.
_StashKey = tuple[int, _CallKey]


class Gemini3LLMError(RuntimeError):
    """Configuration error building the element - notably a missing API key.

    Kept as a named type so callers can distinguish "you did not give me a key"
    from a genai transport error, and so the message can be asserted in a test
    without pattern-matching a bare ``RuntimeError``. The key value is never part
    of this (or any) error text.
    """


def _build_client(api_key: str | None) -> genai.Client:
    """Build an AI-Studio genai client, sourcing the key from env if not passed.

    The key is read from ``api_key`` if given, else from
    :data:`DEFAULT_API_KEY_ENV`. If neither yields a value we fail with an
    actionable message that deliberately contains no key material - only the name
    of the variable to set. To source the key from a DIFFERENT variable, use
    :func:`build_gemini_llm`, which reads the one it is told to and nothing else.
    """
    key = api_key if api_key is not None else os.environ.get(DEFAULT_API_KEY_ENV)
    if not key:
        raise Gemini3LLMError(
            f"no Gemini API key: pass an api_key to Gemini3LLM, or set the "
            f"{DEFAULT_API_KEY_ENV} "
            "environment variable (it lives in .env, which is gitignored). "
            "This error intentionally contains no key value."
        )
    return genai.Client(api_key=key)


def _normalize_list_arg(value: object) -> object:
    """Mirror AgentDojo's in-place arg rewrite so the fingerprint stays stable.

    Between turns, ``ToolsExecutor`` turns a string-encoded list (``"['a', 'b']"``)
    into a real list. Applying the identical ``is_string_list`` / ``literal_eval``
    step here means the fingerprint is the same whether it sees the original string
    or the rewritten list - so the stash hits either way instead of falling back
    to the signature-dropping bare rebuild.
    """
    if isinstance(value, str) and is_string_list(value):
        return literal_eval(value)
    return value


def _canonical_args(args: object) -> str:
    """Stable JSON string for a tool call's args, used as part of its fingerprint.

    Each value is first normalized the way AgentDojo will normalize it in place
    between turns (see :func:`_normalize_list_arg`), so a turn fingerprinted before
    the rewrite and rebuilt after it produce the same key. ``sort_keys`` makes the
    fingerprint order-independent; ``default=str`` keeps a stray non-JSON value
    from crashing fingerprinting (correctness of the key matters, exactness of an
    odd value does not).
    """
    normalized: object
    if isinstance(args, dict):
        normalized = {k: _normalize_list_arg(v) for k, v in args.items()}
    else:
        normalized = _normalize_list_arg(args)
    try:
        return json.dumps(normalized, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return repr(args)


def _call_key(tool_calls: Sequence[Any]) -> _CallKey:
    """Fingerprint an assistant turn by the tool calls it made.

    Each call contributes ``(function name, call id, canonical args)``. The tuple
    over all calls in the turn, paired with the turn's position, is the key under
    which we stash - and later recover - the original genai ``Content`` that
    carried the thought signatures.
    """
    return tuple((tc.function, tc.id, _canonical_args(tc.args)) for tc in tool_calls)


class Gemini3LLM(GoogleLLM):  # type: ignore[misc]  # GoogleLLM is untyped (agentdojo has no py.typed)
    """AgentDojo LLM element for Gemini 3.x that preserves ``thought_signature``.

    Drop-in for AgentDojo's ``GoogleLLM``, differing in exactly one place: when
    rebuilding the genai request from AgentDojo's ``ChatMessage`` history, an
    assistant tool-call turn is re-emitted as the ORIGINAL ``Content`` the model
    returned (signatures intact) rather than as freshly minted bare parts. That
    is the whole reason 3.x tool loops survive past their first turn.

    Uses the AI Studio API (an ``api_key``), not Vertex. Carries the same two
    public counters as :class:`~evals.agentdojo.openai_llm.CountingOpenAILLM`, with
    the same two meanings: :attr:`call_count` (model turns) and
    :attr:`request_count` (genai request attempts, retries included), so a
    frugality-conscious runner can report what a run cost in a unit that means the
    same thing on either provider.
    """

    model: str
    client: genai.Client
    temperature: float | None
    max_tokens: int
    call_count: int
    request_count: int

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int = 4096,
        client: genai.Client | None = None,
    ) -> None:
        # A caller-supplied client wins and needs no key; otherwise build one from
        # the environment. We do NOT call super().__init__: the parent's no-client
        # path warns and reaches for Vertex, which is not the API this key uses.
        self.model = model
        self.client = client if client is not None else _build_client(api_key)
        self.temperature = temperature
        # Stored under the parent's attribute name so the inherited query plumbing
        # (were it ever reached) and our override agree on the token cap.
        self.max_tokens = max_output_tokens
        # A non-None name so `AgentPipeline.from_config` has a pipeline name to use
        # even before the runner overrides it (a None name makes AgentDojo skip
        # logging the run).
        self.name = f"gemini3-{model}"
        self.call_count = 0
        self.request_count = 0
        # (position, fingerprint) -> original genai Content carrying the signatures.
        self._original_contents: dict[_StashKey, types.Content] = {}

    # -- the one behavioural change ------------------------------------------

    def _to_google_content(self, message: ChatMessage, position: int) -> types.Content:
        """Convert one ChatMessage to genai, preserving assistant signatures.

        For an assistant message that made tool calls, recover and return the
        original ``Content`` we stashed when the model produced it - that object
        still carries every ``thought_signature``. The lookup is keyed by the
        message's position as well as its tool-call fingerprint, so two identical
        call turns in one conversation do not alias. Every other message, and any
        assistant turn we have no stash for, defers to AgentDojo's own converter.
        """
        if message["role"] == "assistant" and message.get("tool_calls"):
            cached = self._original_contents.get((position, _call_key(message["tool_calls"])))
            if cached is not None:
                return cached
        # AgentDojo's converter is untyped (-> Any); pin it to Content here.
        fallback: types.Content = _message_to_google(message)
        return fallback

    def _remember_original_content(
        self,
        position: int,
        assistant_message: ChatMessage,
        completion: types.GenerateContentResponse,
    ) -> None:
        """Stash the genai Content of a tool-call turn under its (position, key).

        Only tool-call turns need remembering - they are the only ones that carry
        signatures the next turn must echo. A candidate-less or content-less
        response simply leaves nothing to stash.
        """
        tool_calls = assistant_message.get("tool_calls")
        if not tool_calls:
            return
        candidates = getattr(completion, "candidates", None)
        if not candidates:
            return
        content = candidates[0].content
        if content is None:
            return
        self._original_contents[(position, _call_key(tool_calls))] = content

    @retry(
        wait=wait_random_exponential(multiplier=1, max=40),
        stop=stop_after_attempt(3),
        reraise=True,
        retry=retry_if_not_exception_type(ClientError),
    )
    def _generate_content(
        self,
        contents: list[types.Content],
        config: types.GenerateContentConfig,
    ) -> types.GenerateContentResponse:
        """Make one genai request under the stock retry policy, counting each try.

        The policy is exactly AgentDojo's own: up to three attempts with
        exponential backoff, and no retry on a ``ClientError`` (a 4xx is
        deterministic - retrying a 400 just burns quota). :attr:`request_count` is
        bumped at the TOP of the retried body, so every attempt is counted: a
        request that reaches the server and then errors spends quota just like a
        successful one, and a retried transient error must not be undercounted. The
        TURN counter deliberately lives in :meth:`query` instead - one turn, however
        many attempts it took. Retrying a transient ``ServerError`` here is also what
        keeps a flaky 5xx from being mis-scored as a successful attack and inflating
        ASR.
        """
        self.request_count += 1
        response: types.GenerateContentResponse = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )
        return response

    # -- the AgentDojo entry point -------------------------------------------

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env | None = None,
        messages: Sequence[ChatMessage] | None = None,
        extra_args: dict[str, Any] | None = None,
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict[str, Any]]:
        """Run one model turn. Mirrors ``GoogleLLM.query`` but with three changes:

        1. Assistant tool-call turns are re-emitted from their original genai
           ``Content`` (via :meth:`_to_google_content`), preserving signatures,
           keyed by conversation position so identical turns do not alias.
        2. The genai call goes through :meth:`_generate_content`, which counts every
           ATTEMPT and retries it, and the returned ``Content`` is stashed for the
           next turn. This method counts the TURN itself - once, before any of that
           happens, because a turn whose every attempt failed still cost the agent a
           decision.
        3. At the start of a conversation (no assistant turn present yet) the stash
           is cleared, so a previous task cannot leak a stale signature into this
           one and the dict does not grow unbounded across a run.

        The mutable defaults on the base signature (``EmptyEnv()``, ``[]``, ``{}``)
        are replaced with ``None`` sentinels and materialised here - AgentDojo
        always passes these positionally at run time, so behaviour is unchanged.
        """
        if env is None:
            env = EmptyEnv()
        if messages is None:
            messages = []
        if extra_args is None:
            extra_args = {}
        self.call_count += 1

        message_list = list(messages)
        if message_list and message_list[0]["role"] == "system":
            system_instruction = message_list[0]["content"][0]["content"]
            other_messages = message_list[1:]
        else:
            system_instruction = None
            other_messages = message_list

        # Conversation start: no assistant turn has been produced yet. Drop any
        # stash left from a previous task so a stale entry cannot collide, and so
        # the dict is bounded to one conversation's worth of turns.
        if not any(message["role"] == "assistant" for message in other_messages):
            self._original_contents.clear()

        google_messages = [
            self._to_google_content(message, position)
            for position, message in enumerate(other_messages)
        ]
        google_messages = _merge_tool_result_messages(google_messages)

        google_functions = [_function_to_google(tool) for tool in runtime.functions.values()]
        # Typed as list[Any] so it satisfies genai's broad ToolListUnion (invariant
        # list[Tool] would not be assignable to it).
        google_tools: list[Any] | None = (
            [types.Tool(function_declarations=google_functions)] if google_functions else None
        )

        generation_config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            tools=google_tools,
            system_instruction=system_instruction,
        )
        # The assistant turn we are about to produce is appended after the current
        # other_messages, so on the next turn it sits at index len(other_messages).
        assistant_position = len(other_messages)
        completion = self._generate_content(google_messages, generation_config)
        output = _google_to_assistant_message(completion)
        self._remember_original_content(assistant_position, output, completion)
        return query, runtime, env, [*messages, output], extra_args


def build_gemini_llm(*, model: str, api_key_env: str = DEFAULT_API_KEY_ENV) -> Gemini3LLM:
    """Read the key from ``api_key_env`` and build the Gemini 3.x element.

    The mirror of :func:`evals.agentdojo.openai_llm.build_openai_compat_llm`, and it
    exists for the same reason that one does: the runner should dispatch on provider
    and nothing else. Reading the variable inside the runner instead put
    gemini-specific knowledge - and a second, verbatim copy of this error text - in
    a module whose only job is choosing a provider. Two copies of a
    security-relevant message drift apart; one cannot.

    The key is read ONLY from the named variable. A missing (or empty) value fails
    with a message naming the VARIABLE to set, never a key value, and never silently
    falls back to :data:`DEFAULT_API_KEY_ENV`: a run executed with a key the operator
    did not name is not the run they asked for, and they would have no way to find
    out that it happened.
    """
    key = os.environ.get(api_key_env)
    if not key:
        raise Gemini3LLMError(
            f"no API key: set the {api_key_env} environment variable "
            "(it lives in .env, which is gitignored). "
            "This error intentionally contains no key value."
        )
    return Gemini3LLM(model, api_key=key)
