"""The AgentDojo shim over :mod:`aegis.middleware` - message shapes and two seats.

WHAT IS IN HERE, AND WHAT DELIBERATELY IS NOT
---------------------------------------------
The defense itself - the taint state and its per-conversation reset, the
value-based argument attribution, the layers, the decision path and the refusal
text - lives in :mod:`aegis.middleware` and knows nothing about AgentDojo. This
module is the part a SECOND framework would have to write for itself and nothing
more: how to recognise a tool result in a ``ChatMessage`` dict, how to put the
guarded spans back where they came from, how to turn a ``FunctionCall`` into a
name and an argument snapshot, and how to phrase a refusal as a message the
agent-under-test will actually be shown. Anything in this file that reasons about
trust tiers is a bug.

THE TWO HALVES, AND WHY THEY ARE TWO
------------------------------------
An Aegis-defended AgentDojo run has two halves, and they sit on opposite sides
of the tool call:

* the OUTPUT side - everything that happens to a tool RESULT on its way back
  into the model's context. That is L1 (provenance), L2 (spotlighting), L3
  (detection) and the optional L4 (quarantine), and it is
  :class:`AegisToolOutputGuard`.
* the CALL side - everything that happens to a tool CALL on its way to the
  environment. That is L5 (the capability gate), which has to run BEFORE
  ``ToolsExecutor`` calls ``runtime.run_function``, because by the time a result
  exists the side effect has already happened. That is
  :class:`AegisGatedToolsExecutor`.

They are separate elements, but they share one :class:`AegisMiddleware`: the gate
can only refuse a call for a reason the output side observed, so two middlewares
would silently produce a gate that never refuses - the failure mode that looks
exactly like a working pipeline. :func:`build_aegis_pipeline` is the only
supported way to wire them, precisely so that sharing cannot be forgotten, and it
is why the executor takes the middleware rather than a policy: there is no
correct way to construct that element on its own.

WHERE THE ELEMENTS GO IN THE PIPELINE
-------------------------------------
Inside ``ToolsExecutionLoop``, the gate REPLACES ``ToolsExecutor`` and the guard
sits immediately after it, before the LLM element::

    ToolsExecutionLoop([AegisGatedToolsExecutor(...), AegisToolOutputGuard(...), llm])

That ordering is the whole contract. The gate is a ``ToolsExecutor`` subclass
rather than an element in front of one because there is no seat in front of one:
``ToolsExecutor.query`` runs every call inline, so an element placed before it
could only rewrite the assistant message (erasing the evidence that the call was
attempted), and an element placed after it would be reasoning about a side effect
that has already fired. The guard then sees each tool result in the turn it is
produced and rewrites it before any model token is spent reading it - the same
seat AgentDojo's own ``TransformersBasedPIDetector`` occupies, so the defended
pipeline differs from the stock ``transformers_pi_detector`` pipeline in exactly
one element, which is what makes the comparison honest.

WHY BOTH HALVES OF A RESULT ARE GUARDED
---------------------------------------
``ToolsExecutor`` answers a failed call with an EMPTY content block and the
exception text in ``error``, and ``OpenAILLM`` renders
``message["error"] or <content blocks>`` - so on an errored call the model reads
the error string and never the content. A guard that inspected content alone
would record an empty result, scan nothing, fence nothing, and report the turn as
guarded while the only text the model actually saw went past every layer
untouched. Both halves are therefore marked; see :func:`_untrusted_text_of` for
why the text RECORDED as evidence is not quite the text marked.

FAILURE POLICY: NEVER RAISE
---------------------------
An exception thrown from inside the tool loop propagates out of
``benchmark_suite_*`` and ends the whole run. A run costs paid quota (the Groq
budget is ~25-30 task runs per day), so an unexpected message shape must be
passed through untouched, not asserted against. Anything unrecognised is left
exactly as it was found and counted in :attr:`AegisToolOutputGuard.failures` or
:attr:`AegisGatedToolsExecutor.failures`, so a silent pass-through is still
visible afterwards. The counters are kept per ELEMENT rather than on the shared
middleware, because "which seat failed" is the question the run report has to
answer.

The gate obeys the same rule for a second, independent reason: a REFUSED call is
returned to the model as a tool RESULT explaining the refusal, never as an
exception. AgentDojo's own detector aborts the episode with ``AbortAgentError``,
which scores the user's task as failed by construction - so a defense evaluated
that way reports a utility of roughly zero and an ASR of roughly zero, and the
pair says nothing. Utility is half the measurement; the agent has to be able to
hear "no" and carry on.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple, cast

from agentdojo.agent_pipeline import AgentPipeline, PipelineConfig
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage
from agentdojo.agent_pipeline.tool_execution import (
    ToolsExecutionLoop,
    ToolsExecutor,
    tool_result_to_str,
)
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage, text_content_block_from_string

from aegis.config.policy import SecurityPolicy
from aegis.middleware import (
    QUARANTINE_INSTRUCTION_FLAG,
    QUARANTINE_UNAVAILABLE_FLAG,
    AegisMiddleware,
    Decision,
    DefenseConfig,
    GateAction,
    GateEntry,
    Layer,
    QuarantineVerdict,
    TaintRecord,
    TaintState,
    ToolCall,
    ToolOutput,
    subtract_own_args,
)
from aegis.middleware import conversation_key as _key_of
from aegis.security.capabilities import AuthorizationContext, CapabilityGate
from aegis.security.quarantine import QuarantineExtractor
from aegis.security.spotlight import Spotlighter, SpotlightStyle, guidance_for_style

__all__ = [
    "AEGIS_GENERATED",
    "QUARANTINE_INSTRUCTION_FLAG",
    "QUARANTINE_UNAVAILABLE_FLAG",
    "AegisGatedToolsExecutor",
    "AegisMiddleware",
    "AegisPipeline",
    "AegisToolOutputGuard",
    "DefenseConfig",
    "GateAction",
    "GateEntry",
    "Layer",
    "QuarantineVerdict",
    "SpotlightStyle",
    "TaintRecord",
    "TaintState",
    "build_aegis_pipeline",
    "conversation_key",
]

AEGIS_GENERATED = "aegis_generated"
"""Key marking a message Aegis itself wrote into the conversation.

Only the gate's refusal results carry it. Without it the output guard would treat
its own sibling's refusal text as untrusted tool output: datamark it (making our
own explanation unreadable to the model), record a taint entry no tool produced,
and possibly flag it - because a refusal naturally contains words like "denied"
and the name of a side-effecting tool. Provenance is only useful if it says where
text actually came from.
"""


def conversation_key(query: str, messages: Sequence[ChatMessage]) -> str:
    """Identify the conversation a turn belongs to, from an AgentDojo history.

    The hashing rule is :func:`aegis.middleware.conversation_key`; the only part
    that is AgentDojo's is "the first message whose role is user". A history with
    no user message in it yet takes the library's ``None`` branch, which is a
    DIFFERENT key rather than the same key with an empty string appended.
    """
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            return _key_of(query, _text_of(message))
    return _key_of(query, None)


class AegisToolOutputGuard(BasePipelineElement):  # type: ignore[misc]  # agentdojo is untyped
    """L1/L2/L3(/L4) applied to tool results, in the seat after ToolsExecutor.

    One instance is reused for a whole benchmark run; all per-task state lives in
    the middleware and is reset per conversation.
    """

    def __init__(
        self,
        config: DefenseConfig,
        *,
        tool_names: Sequence[str] = (),
        state: TaintState | None = None,
        quarantine: QuarantineExtractor | None = None,
        middleware: AegisMiddleware | None = None,
    ) -> None:
        """Build the output-side element.

        ``middleware`` is how :func:`build_aegis_pipeline` hands this element the
        SAME runtime the gate holds. Without one a private middleware is built
        from the remaining arguments, which is right for an output-only use (and
        for a test): the guard writes taint that nothing reads.
        """
        self._config = config
        self._mw = (
            middleware
            if middleware is not None
            else AegisMiddleware(
                config,
                # The live suite's tool names are what make `tool_invocation_attempt`
                # able to fire at all - without them that pattern has nothing to
                # match, and the flag silently never appears.
                tool_names=tool_names,
                quarantine=quarantine,
                state=state,
            )
        )
        # Turns whose processing raised and were passed through untouched. A
        # silent pass-through is a defense that did not happen, so it is counted
        # rather than swallowed.
        self.failures = 0
        # A non-None name keeps AgentDojo's per-element logging from skipping it.
        self.name = config.label

    @property
    def config(self) -> DefenseConfig:
        return self._config

    @property
    def state(self) -> TaintState:
        """The current conversation's taint record - read by the L5 element."""
        return self._mw.state

    @property
    def quarantine_failures(self) -> int:
        """Tool results L4 was asked about but could not judge.

        Reported for the same reason the pass-throughs are: an arm where the
        extractor was down for half the run is not the arm it claims to be. It
        lives on the middleware, because L4 runs there.
        """
        return self._mw.quarantine_failures

    @property
    def _spotlighter(self) -> Spotlighter:
        """The middleware's L2, seen from the seat that applies it.

        One object owns the spotlighter - two would be an ablation arm that is on
        in one place and off in another. This forwards rather than copies so that
        substituting it here (which is how the never-raise backstop is tested)
        substitutes the one that actually runs.
        """
        return self._mw._spotlighter

    @_spotlighter.setter
    def _spotlighter(self, spotlighter: Spotlighter) -> None:
        self._mw._spotlighter = spotlighter

    # -- AgentDojo entry point ------------------------------------------------

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env | None = None,
        messages: Sequence[ChatMessage] | None = None,
        extra_args: dict[str, Any] | None = None,
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict[str, Any]]:
        """Guard every tool result that is new this turn, and never raise.

        The mutable defaults on the base signature (``EmptyEnv()``, ``[]``,
        ``{}``) are replaced with ``None`` sentinels and materialised here;
        AgentDojo always passes these positionally at run time, so behaviour is
        unchanged.

        The blanket except is deliberate and is explained in the module
        docstring: an exception here ends the benchmark run and burns the day's
        quota, and a tool result we failed to understand is no more dangerous
        unguarded than it would be if the run had never happened.
        """
        if env is None:
            env = EmptyEnv()
        if messages is None:
            messages = []
        if extra_args is None:
            extra_args = {}

        try:
            guarded = self._guard_turn(query, list(messages))
        except Exception:
            self.failures += 1
            return query, runtime, env, messages, extra_args
        return query, runtime, env, guarded, extra_args

    # -- internals ------------------------------------------------------------

    def _guard_turn(self, query: str, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Process only the messages appended since the last turn.

        Returns a NEW list. AgentDojo's own detector element builds a new list
        and then returns the old one - its redaction works only through an
        in-place mutation of the message dict, which is a trap worth not
        repeating. Here the rewritten messages are fresh dicts and the new list
        is what is returned, so the caller's list is never mutated behind its
        back and an off-arm can be compared byte-for-byte against its input.

        The message COUNT is the middleware's monotonic progress signal, and this
        seat is the only one that marks progress: the gate's own ``begin_turn``
        sees one more message than this one did (the assistant turn it is about to
        act on), which is exactly what makes "the history did not grow" able to
        catch a re-run of the same task.
        """
        self._mw.begin_turn(conversation_key(query, messages), len(messages))
        start = self._mw.state.processed_messages
        if start >= len(messages):
            return messages

        guarded = list(messages[:start])
        guarded.extend(self._guard_message(message) for message in messages[start:])
        self._mw.state.mark_processed(len(messages))
        return guarded

    def _guard_message(self, message: ChatMessage) -> ChatMessage:
        """Translate one message into a :class:`ToolOutput`, guard it, translate back.

        Anything that is not a tool result carrying model-readable text - in a
        content block or in ``error`` - is returned as the identical object: an
        assistant turn, a user turn, a result with no text at all, or a shape we do
        not recognise. So is EVERY message when L2 is off, which is what makes the
        control arm byte-identical rather than merely equal.
        """
        if not isinstance(message, dict) or message.get("role") != "tool":
            return message
        # Our own refusal text is not tool output; see AEGIS_GENERATED.
        if message.get(AEGIS_GENERATED):
            return message
        blocks = message.get("content")
        text_blocks = [b for b in blocks if _is_text_block(b)] if isinstance(blocks, list) else []
        error = message.get("error")
        error_text = error if isinstance(error, str) else ""
        if not text_blocks and not error_text:
            return message

        # The spans are what the model will read, in the order it will read them:
        # every text block, then the error. The evidence is what the layers judge,
        # which is not the same string - see _untrusted_text_of.
        spans = (*(_block_text(b) for b in text_blocks), *((error_text,) if error_text else ()))
        guarded = self._mw.guard(
            ToolOutput(
                tool_name=_tool_name_of(message),
                spans=spans,
                evidence=_untrusted_text_of(message),
                note="agentdojo tool result",
            )
        )

        if not self._config.spotlight:
            return message

        # Back into the message, positionally. Non-text blocks stay exactly where
        # they were: wrapping per block rather than joining is what makes that
        # possible, and it cannot drop content if a result ever arrives as more
        # than one block.
        rewritten = dict(message)
        marked: Iterator[str] = iter(guarded.spans[: len(text_blocks)])
        if isinstance(blocks, list):
            rewritten["content"] = [
                text_content_block_from_string(next(marked)) if _is_text_block(block) else block
                for block in blocks
            ]
        if error_text:
            # The same treatment, for the half the model will actually be shown.
            # Marking only the content would spotlight the string nobody reads.
            rewritten["error"] = guarded.spans[len(text_blocks)]
        return cast(ChatMessage, rewritten)


def _is_text_block(block: object) -> bool:
    if not isinstance(block, dict):
        return False
    return bool(block.get("type") == "text")


def _block_text(block: Any) -> str:
    content = block.get("content")
    return content if isinstance(content, str) else ""


def _untrusted_text_of(message: Any) -> str:
    """What the guard treats as untrusted text arriving from the tool boundary.

    The content blocks AND ``error``, because the LLM elements disagree about which
    one the model is shown and a result that errored carries its whole payload in
    ``error``. Separate from :func:`_text_of`, which answers the narrower question
    "what is in the content blocks" - the one the conversation key asks of a USER
    message.

    MINUS the agent's own arguments wherever the error quotes them back. That
    subtraction is the one piece of judgement that cannot leave this module,
    because it exists solely to undo AgentDojo's echo of the input dict: pydantic's
    ``ValidationError`` embeds the arguments verbatim, so a call that omits a
    required field comes back as an error string containing the agent's own
    recipient. :func:`aegis.middleware.subtract_own_args` says what that would cost
    if it were recorded as evidence.

    What the subtraction cannot remove - anything the ENVIRONMENT put into the
    exception - is exactly what stays, and that is the text this closes the gap on.
    Today no v1.2 suite tool interpolates environment prose into an exception, so
    the gap is structural rather than live; it is closed anyway, because "the guard
    saw what the model saw" is the invariant every layer above rests on.
    """
    error = message.get("error") if isinstance(message, dict) else None
    parts = (_text_of(message), _without_own_args(error, message) if error else None)
    return "\n".join(p for p in parts if isinstance(p, str) and p)


def _without_own_args(error: Any, message: Any) -> str:
    """Find the producing call's arguments on the message, then subtract them.

    Only the lookup is here; the subtraction rule and its residual are
    :func:`aegis.middleware.subtract_own_args`. A result whose ``tool_call``
    carries no argument mapping keeps its error whole - there is nothing to
    attribute the echo to, so removing anything would be a guess.
    """
    if not isinstance(error, str):
        return ""
    call = message.get("tool_call") if isinstance(message, dict) else None
    args = getattr(call, "args", None)
    if not isinstance(args, Mapping):
        return error
    return subtract_own_args(error, args)


def _text_of(message: Any) -> str:
    """Join a message's text blocks, tolerating any shape.

    AgentDojo's ``get_text_content_as_str`` would do this, but it assumes a
    well-formed list of blocks; this module's contract is that a malformed
    message never raises.
    """
    blocks = message.get("content") if isinstance(message, dict) else None
    if not isinstance(blocks, list):
        return ""
    return "\n".join(_block_text(b) for b in blocks if _is_text_block(b))


def _tool_name_of(message: Any) -> str:
    """Name the tool a result came from, for the provenance URI.

    ``ChatToolResultMessage`` has no ``name`` field - the name lives on the
    attached ``FunctionCall``. A result with no usable call still gets recorded,
    under the middleware's documented placeholder, because "some tool returned
    untrusted text" is the fact L1 exists to preserve.
    """
    tool_call = message.get("tool_call")
    function = getattr(tool_call, "function", None)
    return function if isinstance(function, str) and function else "unknown_tool"


# ---------------------------------------------------------------------------
# L5 - the CALL side.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _CallDecision:
    """A middleware decision bound to the AgentDojo call it was made about.

    The original ``FunctionCall`` object is kept rather than rebuilt from
    :class:`ToolCall`, and that is not an optimisation: the base
    ``ToolsExecutor.query`` rewrites ``tool_call.args`` IN PLACE (a string
    argument that should have been a list is ``literal_eval``-ed), and the object
    it rewrites has to be the one the outer assistant message holds.
    """

    call: Any
    """The ``FunctionCall``; untyped because agentdojo is."""

    decision: Decision

    @property
    def refused(self) -> bool:
        return self.decision.refused


def _tool_call_of(call: Any) -> ToolCall:
    """Snapshot one ``FunctionCall`` as a framework-neutral :class:`ToolCall`.

    ``dict(args)`` is a COPY, taken now, and the timing is load-bearing:
    ``ToolsExecutor.query`` later rewrites the live mapping in place, so a
    middleware holding a reference would match ``['a@evil.example']`` as a parsed
    list rather than as the 18-character string the model actually emitted - a
    different attribution outcome, reached silently.
    """
    raw_args = getattr(call, "args", None)
    return ToolCall(
        name=getattr(call, "function", "") or "",
        args=dict(raw_args) if isinstance(raw_args, Mapping) else {},
    )


class AegisGatedToolsExecutor(ToolsExecutor):  # type: ignore[misc]  # agentdojo is untyped
    """L5: consult the capability gate BEFORE each tool call fires.

    WHY THIS SUBCLASSES ToolsExecutor
    ---------------------------------
    ``ToolsExecutor.query`` calls ``runtime.run_function`` inline, so there is no
    element position from which a side effect can still be prevented. Taking its
    seat is the only place the answer can still be "no".

    Allowed calls are delegated to ``super().query()`` one call at a time rather
    than reimplemented. That matters more than it looks: the base method also
    handles empty function names, tools absent from the runtime, and a
    string-to-list argument rewrite that mutates the ``FunctionCall`` in place. A
    reimplementation would drift from those semantics, and the drift would show up
    as a utility difference attributed to the defense.

    WHAT IT ASKS THE MIDDLEWARE, AND WHAT IT DOES NOT DECIDE
    --------------------------------------------------------
    One question per turn: here are the pending calls and here is the set of tool
    names this runtime can actually execute - which of them may fire? The taint
    reconstruction, the read-only override, the CONFIRM-as-DENY rule and the
    refusal wording are all the middleware's, and are documented there. What is
    decided HERE is only the shape of the answer: a refused call comes back as a
    tool result rather than an exception, and the surviving calls are run through
    the base class one at a time.
    """

    def __init__(
        self,
        config: DefenseConfig,
        *,
        middleware: AegisMiddleware,
        tool_output_formatter: Any = tool_result_to_str,
    ) -> None:
        """Take the executor seat for one arm of the ablation.

        The middleware is required and is not constructible from here on purpose.
        This element only READS taint; the guard writes it. An executor holding a
        middleware of its own would consult an empty state and never refuse
        anything, which is indistinguishable from a working gate until the numbers
        are published.
        """
        super().__init__(tool_output_formatter)
        self._config = config
        self._mw = middleware
        # Turns whose gating raised; the calls were then executed ungated. Counted
        # for the same reason the guard counts its own: a defense that silently
        # did not run must not be reported as one that did.
        self.failures = 0
        self.name = f"{config.label}-gate"

    # -- accessors ------------------------------------------------------------

    @property
    def config(self) -> DefenseConfig:
        return self._config

    @property
    def state(self) -> TaintState:
        """The shared taint state; written by the output guard, read here."""
        return self._mw.state

    @property
    def authorization(self) -> AuthorizationContext:
        return self._mw.authorization

    @property
    def ledger(self) -> tuple[GateEntry, ...]:
        """Every decision the shared middleware made, oldest first.

        A copy, so a caller cannot append evidence of a refusal that never
        happened. It spans the whole run rather than one task - each entry
        carries its own ``conversation_key`` for splitting.
        """
        return self._mw.ledger

    @property
    def refusals(self) -> tuple[GateEntry, ...]:
        return self._mw.refusals

    def entries_for(self, tool_name: str) -> tuple[GateEntry, ...]:
        return self._mw.entries_for(tool_name)

    @property
    def _gate(self) -> CapabilityGate:
        """The middleware's L5, seen from the seat that consults it.

        Forwarding rather than holding a second one, for the same reason the guard
        forwards its spotlighter: one object per layer, so an arm cannot be on in
        one place and off in another, and so substituting the gate here (which is
        how the never-raise backstop is tested) substitutes the one that runs.
        """
        return self._mw.gate

    @_gate.setter
    def _gate(self, gate: CapabilityGate) -> None:
        self._mw._gate = gate

    # -- AgentDojo entry point ------------------------------------------------

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env | None = None,
        messages: Sequence[ChatMessage] | None = None,
        extra_args: dict[str, Any] | None = None,
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict[str, Any]]:
        """Gate the pending tool calls, then execute the ones that survive.

        Three paths reach ``super().query()`` with the ORIGINAL arguments, which
        is the exact baseline code path and is what makes the all-layers-off arm a
        valid control: L5 is off, nothing was refused, or gating itself raised.
        Only a refusal takes the slower per-call path.
        """
        if env is None:
            env = EmptyEnv()
        if messages is None:
            messages = []
        if extra_args is None:
            extra_args = {}

        if not self._config.gate:
            return self._delegate(query, runtime, env, messages, extra_args)

        try:
            decisions = self._decide(query, runtime, list(messages))
        except Exception:
            # Same policy as the guard: a crash inside the tool loop ends the run
            # and burns the day's quota. Ungated execution is the baseline
            # behaviour, and it is counted rather than hidden.
            self.failures += 1
            return self._delegate(query, runtime, env, messages, extra_args)

        if decisions is None or not any(d.refused for d in decisions):
            return self._delegate(query, runtime, env, messages, extra_args)

        try:
            return self._execute(query, runtime, env, list(messages), extra_args, decisions)
        except Exception:
            # The never-raise contract covers the substitution too, and this is the
            # one place that synthesises messages, so it is the least proven path,
            # guarded the same way the decision path above is.
            #
            # The fallback repeats the turn ungated, which can re-run a call
            # _execute had already run. That is the lesser harm by a wide margin: a
            # raise here propagates out of benchmark_suite_* and ends the run,
            # discarding every task still to come along with the day's quota, while
            # a repeated call costs one duplicated side effect in one task - and
            # `failures` says on which run it happened.
            self.failures += 1
            return self._delegate(query, runtime, env, messages, extra_args)

    # -- internals ------------------------------------------------------------

    def _delegate(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env,
        messages: Sequence[ChatMessage],
        extra_args: dict[str, Any],
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict[str, Any]]:
        """Run the stock ``ToolsExecutor`` body, unchanged."""
        result: tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict[str, Any]] = (
            super().query(query, runtime, env, messages, extra_args)
        )
        return result

    def _decide(
        self,
        query: str,
        runtime: FunctionsRuntime,
        messages: list[ChatMessage],
    ) -> list[_CallDecision] | None:
        """Ask the middleware about this turn's calls, or return None if there are none.

        The bail-out conditions mirror ``ToolsExecutor.query`` exactly, so a turn
        the base class would ignore is a turn this element ignores.

        The decisions come back positionally, one per pending call, and are paired
        straight back onto the ``FunctionCall`` objects they were made about.
        """
        if not messages:
            return None
        last = messages[-1]
        if not isinstance(last, dict) or last.get("role") != "assistant":
            return None
        calls = last.get("tool_calls")
        if not calls:
            return None
        # Materialise before anything walks it. The decisions are built from one
        # pass and zipped against a second, so a one-shot iterable would leave the
        # second pass empty, `zip(strict=True)` would raise, and query()'s
        # never-raise handler would count a failure and run the turn UNGATED. That
        # is a fail-OPEN direction in the one module whose job is to refuse, and it
        # costs a list() to remove. AgentDojo types this as list | None, so the
        # shape is not reachable from the benchmark - but this module's contract is
        # to survive shapes it did not expect, not to gate only the expected ones.
        calls = list(calls)

        # The gate reads taint recorded by the OUTPUT guard, which runs later in
        # the loop - so on the first tool call of a new task the state still holds
        # the PREVIOUS task's records unless it is reset here too. Inheriting them
        # would refuse a call for a reason belonging to another task: a defense
        # that reads as effective precisely because it fired for the wrong reason.
        # Resetting here is safe: within a task the executor always sees at least
        # one more message than the guard last marked (the LLM's assistant turn),
        # so the "history did not grow" signal cannot fire mid-conversation.
        self._mw.begin_turn(conversation_key(query, messages), len(messages))

        # Membership in the runtime is the ONLY registration test; there is no
        # special case for the empty function name, so a call the base class would
        # answer itself stays the base class's to answer.
        known = {f.name for f in runtime.functions.values()}
        decisions = self._mw.decide([_tool_call_of(call) for call in calls], known_tools=known)
        return [
            _CallDecision(call=call, decision=decision)
            for call, decision in zip(calls, decisions, strict=True)
        ]

    def _execute(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env,
        messages: list[ChatMessage],
        extra_args: dict[str, Any],
        decisions: list[_CallDecision],
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict[str, Any]]:
        """Run the surviving calls in order, substituting refusals for the rest.

        Each allowed call goes through ``super().query()`` on a one-call assistant
        message, so every allowed call gets exactly the base class's treatment.
        The original assistant message is left intact: rewriting it to hide the
        refused call would erase the evidence that the agent tried, which is the
        one thing this whole apparatus exists to record.
        """
        results: list[ChatMessage] = []
        for decision in decisions:
            if decision.refused:
                results.append(self._refusal_message(decision))
                continue
            single = cast(
                ChatMessage,
                {"role": "assistant", "content": None, "tool_calls": [decision.call]},
            )
            _, runtime, env, produced, extra_args = self._delegate(
                query, runtime, env, [single], extra_args
            )
            results.extend(produced[1:])
        return query, runtime, env, [*messages, *results], extra_args

    def _refusal_message(self, decision: _CallDecision) -> ChatMessage:
        """The refusal, as a tool RESULT the agent can read and work around.

        The text lands in BOTH ``content`` and ``error`` because AgentDojo's LLM
        elements disagree about which one they show: ``OpenAILLM`` renders
        ``message["error"] or <content>``, so a refusal placed only in ``content``
        would be invisible to the primary agent-under-test.
        """
        text = decision.decision.refusal_text
        call = decision.call
        message: dict[str, Any] = {
            "role": "tool",
            "content": [text_content_block_from_string(text)],
            "tool_call": call,
            "tool_call_id": getattr(call, "id", None),
            "error": text,
            AEGIS_GENERATED: True,
        }
        return cast(ChatMessage, message)


# ---------------------------------------------------------------------------
# Pipeline construction.
# ---------------------------------------------------------------------------


class AegisPipeline(NamedTuple):
    """What :func:`build_aegis_pipeline` hands back.

    The pipeline alone would be enough to RUN, but not enough to report: the
    ledger and the taint state live on the elements, and the whole point of the
    ledger is that it is read after the run. The runner's own ``BuiltPipeline``
    already sets this precedent - it carries the LLM element along so its counters
    can be read afterwards - and the alternative, digging the elements back out of
    ``pipeline.elements[3].elements[0]``, is a positional dependency on AgentDojo's
    internals that would break silently.
    """

    pipeline: Any
    executor: AegisGatedToolsExecutor
    guard: AegisToolOutputGuard
    state: TaintState
    defense: str
    """``DefenseConfig.label`` - what the result JSON should record as the defense."""


def build_aegis_pipeline(
    llm: Any,
    config: PipelineConfig,
    defense: DefenseConfig,
    *,
    policy: SecurityPolicy | None = None,
    tool_names: Sequence[str] | None = None,
    authorization: AuthorizationContext | None = None,
    quarantine: QuarantineExtractor | None = None,
    max_iters: int = 15,
) -> AegisPipeline:
    """Compose the Aegis-defended pipeline, mirroring ``AgentPipeline.from_config``.

    ``from_config`` ends in ``raise ValueError("Invalid defense name")``, so a new
    defense cannot be registered without patching AgentDojo. Rather than
    monkeypatch a benchmark harness - which would make every number this project
    reports depend on a mutation of the thing measuring it - the element list is
    built here, deliberately identical to the ``defense=None`` branch except for
    the two Aegis elements inside the tool loop::

        baseline: [SystemMessage, InitQuery, llm, Loop([ToolsExecutor,        llm])]
        aegis:    [SystemMessage, InitQuery, llm, Loop([AegisGated..., Guard, llm])]

    The one other difference is the system message: when L2 is on, the style's
    marker guidance is appended to it, because a mark the model was never taught to
    read is not spotlighting (AgentDojo's own ``spotlighting_with_delimiting``
    defense edits the system message for the same reason). That edit is gated on
    the L2 toggle, so with ``DefenseConfig.none()`` both Aegis elements delegate to
    exactly the base behaviour AND the prompt is byte-identical - the all-off arm
    is a genuine control rather than an approximation of one.
    ``tests/evals/test_defense.py`` asserts the two pipelines produce identical
    conversations and identical side effects.

    Args:
        llm: The agent-under-test element. It appears twice, as in every AgentDojo
            pipeline: once to open the turn and once inside the loop.
        config: Supplies the system message and tool output format. ``config.llm``
            is used only for naming; ``llm`` is what actually runs.
        defense: Which layers are on. Also names the pipeline, because AgentDojo
            caches results per ``pipeline.name`` and two arms sharing a name can
            replay each other's numbers.
        policy: Loaded security policy. Defaults to ``config/trust_tiers.yaml``,
            which is the only place tool sinks are declared - this module contains
            no list of tool names.
        tool_names: Names fed to the L3 detector so ``tool_invocation_attempt`` can
            fire. Defaults to the policy's tools; pass the live suite's names when
            they differ.
        authorization: What the user granted. Defaults to ``allow_all=True``,
            which is the only honest setting for an unattended benchmark: the
            harness cannot ask the human what this task authorises, and a gate
            that refused every call for lack of authorization would drive utility
            to zero and make the accompanying ASR meaningless. It is the ablation
            escape hatch documented on :class:`AuthorizationContext`, never a
            deployment setting.
        quarantine: The L4 extractor. Required when ``defense.quarantine`` is on.
        max_iters: Tool-loop cap, as in AgentDojo.
    """
    if config.system_message is None:  # pragma: no cover - PipelineConfig fills it in
        raise ValueError("PipelineConfig.system_message must be set before building a pipeline")

    # L2 is marking PLUS the prompt-side convention that says what the marks mean;
    # marking alone is the expensive half of spotlighting and buys nothing, because
    # AgentDojo's stock system prompt mentions neither the fence nor the datamark.
    # Appended (never rewritten) so the defended prompt still opens with the
    # baseline one verbatim, and gated on the L2 toggle so the all-off arm's system
    # message stays byte-identical to the undefended baseline's.
    system_message: str = config.system_message
    if defense.spotlight:
        system_message = f"{system_message}\n{guidance_for_style(defense.spotlight_style)}"

    loaded = policy if policy is not None else SecurityPolicy.load()
    formatter = (
        partial(tool_result_to_str, dump_fn=json.dumps)
        if config.tool_output_format == "json"
        else tool_result_to_str
    )

    # ONE middleware, shared. The gate can only refuse a call for a reason the
    # guard observed, so two would silently produce a gate that never refuses
    # anything - the failure mode that looks exactly like a working pipeline.
    middleware = AegisMiddleware(
        defense,
        policy=loaded,
        tool_names=tuple(tool_names) if tool_names is not None else loaded.tool_names,
        authorization=authorization,
        quarantine=quarantine,
    )
    executor = AegisGatedToolsExecutor(
        defense,
        middleware=middleware,
        tool_output_formatter=formatter,
    )
    guard = AegisToolOutputGuard(defense, middleware=middleware)

    tools_loop = ToolsExecutionLoop([executor, guard, llm], max_iters=max_iters)
    pipeline = AgentPipeline([SystemMessage(system_message), InitQuery(), llm, tools_loop])
    llm_name = config.llm if isinstance(config.llm, str) else getattr(llm, "name", None)
    pipeline.name = f"{llm_name}-{defense.label}" if llm_name else defense.label
    return AegisPipeline(
        pipeline=pipeline,
        executor=executor,
        guard=guard,
        state=middleware.state,
        defense=defense.label,
    )
