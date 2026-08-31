"""Aegis inside LangGraph - the second adapter, and what writing it found.

    from aegis.adapters.langgraph import AegisToolNode

    builder = StateGraph(MessagesState)
    builder.add_node("tools", AegisToolNode(tools))

WHY A SECOND ADAPTER EXISTS AT ALL
----------------------------------
``README.md`` states the limit plainly: one adapter cannot prove an interface is
framework-neutral, only refute it. Every shape in :mod:`aegis.middleware` was
designed while looking at AgentDojo, so the honest test is to write the same
integration against a framework that does things differently, then report what
the first one quietly paid for.

WHAT THE MIDDLEWARE GOT RIGHT
-----------------------------
*The conversation id came for free.* AgentDojo has no conversation identity, so
that adapter hashes the user query together with the first user message
(``conversation_key``). LangGraph hands every run a ``thread_id`` in its config,
which is exactly what :meth:`AegisMiddleware.begin_turn` asks for. Requesting "a
stable id and a rising counter" rather than a message list is why this needed no
new concept on either side.

*Spans as a tuple.* A ``ToolMessage`` carries ``str | list[str | dict]``, so the
same flattening the AgentDojo shim performs had to be written again. That is the
one piece a third adapter will also write - and it is the argument for
:class:`~aegis.middleware.types.ToolOutput` taking a TUPLE of spans, because a
single string would force every adapter to join the blocks and hand back
something the framework cannot put where it found it.

WHAT IT COST, HONESTLY
----------------------
*Two seats became one.* AgentDojo splits the work across a pipeline element that
guards tool output and a replacement tools-executor that gates calls. LangGraph
runs tool calls in a single node, so both halves live here. That is the EASIER
shape, not the harder one, and saying so is more useful than claiming the
interface bent gracefully under strain.

*Callbacks cannot do this job.* The obvious LangChain seat is a
``BaseCallbackHandler`` with ``on_tool_end``, and it does not work:
``on_tool_end`` returns ``None``, so a callback may OBSERVE a tool result but
never replace it, and nothing in the callback surface can stop a call from
running. Guarding has to sit where the value actually flows through. That is
LangChain's design rather than a gap here, but an adopter who reaches for
callbacks first will lose an afternoon to it.

*The framework reads your type annotations, and this repository's house style
hides them.* LangGraph decides what to inject into a node by comparing the
node's ``config`` annotation against the real ``RunnableConfig`` type. Under
``from __future__ import annotations`` - which every other module here uses, and
should - the annotation is a STRING, the comparison fails, and LangGraph skips
injecting the config with only a warning that reads as a tautology. The config
then arrives as ``None``, ``thread_id`` is never seen, every run collapses onto
one conversation id, and one conversation's taint survives into the next.

That is the single most valuable thing this adapter found, and it is worth being
precise about whose fault it is: not the middleware's, whose ``begin_turn``
contract is exactly right, and not really LangGraph's either. It is what happens
when a library that resolves types statically meets a framework that resolves
them at runtime, and it is invisible to every test that uses one instance per
run. See the comment above the imports.

*Both invocation paths must exist.* A graph driven with ``ainvoke`` calls the
node's ``ainvoke``. A synchronous-only node does not merely run slower - the
async path fails outright, so the whole adapter is unusable to half of
LangGraph's users. The middleware being pure and synchronous is what makes
supporting both cheap: only the delegated execution differs, so
:meth:`AegisToolNode._plan` and :meth:`AegisToolNode._assemble` are shared and
neither knows which path called it.

*Delegation ties this to a compiled graph.* Execution is handed to a real
``ToolNode``, which in LangGraph 1.x requires a runtime that only exists inside a
compiled graph - so this node cannot be invoked standalone either. That is
correct drop-in behaviour rather than a defect, but it means the tests drive a
real graph instead of calling the node directly, and so should yours.
"""

# NO `from __future__ import annotations` IN THIS MODULE - it breaks the framework.
#
# It is this repository's house style and it is correct everywhere else. Here it
# turns every annotation into a STRING, and LangGraph inspects annotations at
# runtime to decide what to inject into a node: `_runnable.py` compares
# `p.annotation not in typ` against the real `RunnableConfig` type object, a
# string never matches, and it silently skips injecting the config entirely.
#
# The symptom is not a crash. `config` arrives as None, `thread_id` is never
# seen, every run collapses onto DEFAULT_THREAD_ID, and one conversation's taint
# survives into the next - which `aegis.middleware.taint` calls manufacturing a
# defense that was never triggered. Verified: with the future import present the
# node saw 'aegis-langgraph-default' for all three turns of two different
# threads; without it, each thread's own id arrives.
#
# A warning IS emitted, and it reads "should be typed as 'RunnableConfig | None',
# not 'RunnableConfig | None'" - the same text twice, because one side is the
# type and the other is its string. Easy to dismiss as a framework quirk.

from collections.abc import Sequence
from typing import Any

try:
    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_core.runnables import RunnableConfig
    from langchain_core.tools import BaseTool
    from langgraph.prebuilt import ToolNode
except ImportError as exc:  # pragma: no cover - depends on the install
    # ImportError rather than ModuleNotFoundError: a half-installed framework,
    # or an import hook of the kind the tests use, raises the parent class.
    # Catching only the subclass let the unhelpful original escape.
    #
    # Name the extra rather than the transitive package the caller never asked
    # for. Without this the error is "No module named 'langchain_core'", which
    # tells an adopter nothing about which install line would fix it - the same
    # defect this project already fixed once for the Gemini provider and the
    # `llm` extra.
    raise ModuleNotFoundError(
        "aegis.adapters.langgraph needs the 'langchain' extra. Install it with "
        "`pip install aegis-rag[langchain]`. Nothing else in aegis requires an "
        "agent framework: the trust lattice, the capability gate and the "
        "middleware all run without one, which is the point of keeping adapters "
        "in their own package."
    ) from exc

from aegis.middleware import (
    AegisMiddleware,
    Decision,
    DefenseConfig,
    ToolCall,
    ToolOutput,
)

__all__ = ["DEFAULT_THREAD_ID", "AegisToolNode"]

DEFAULT_THREAD_ID = "aegis-langgraph-default"
"""Used when a graph is invoked with no ``thread_id`` in its config.

A constant rather than a fresh random value on purpose: an unidentified run is
ONE conversation as far as the taint state is concerned, which is the
conservative reading. Randomising per invocation would reset between the turns of
a single run and quietly disarm the gate - the failure
:class:`aegis.middleware.taint.TaintState` describes as manufacturing a defense
that was never triggered.
"""

_Plan = tuple[list[dict[str, Any]], list[Decision], list[dict[str, Any]], dict[str, Any]]


class AegisToolNode:
    """A drop-in replacement for ``langgraph.prebuilt.ToolNode``, carrying L1-L5.

    Holds ONE middleware for the life of the node, because the output side
    records what the tools returned and the call side reads it back. A fresh
    instance per invocation could never refuse anything, which looks exactly like
    a working integration - so the sharing is not an optimisation.

    Execution of the calls it allows is delegated to a real ``ToolNode`` rather
    than reimplemented, which keeps injected state, error handling and every
    other LangGraph behaviour LangGraph's problem.
    """

    def __init__(
        self,
        tools: Sequence[BaseTool | Any],
        *,
        middleware: AegisMiddleware | None = None,
        config: DefenseConfig | None = None,
        messages_key: str = "messages",
        **tool_node_kwargs: Any,
    ) -> None:
        self._inner = ToolNode(tools, messages_key=messages_key, **tool_node_kwargs)
        self._messages_key = messages_key
        self._known: frozenset[str] = frozenset(self._inner.tools_by_name)
        self._middleware = middleware or AegisMiddleware(
            config or DefenseConfig.all_layers(),
            tool_names=sorted(self._known),
        )

    @property
    def middleware(self) -> AegisMiddleware:
        """The shared runtime, so a caller can read its ledger after a run."""
        return self._middleware

    @property
    def tools_by_name(self) -> dict[str, Any]:
        """Mirrors ``ToolNode`` so this stays a drop-in replacement."""
        return dict(self._inner.tools_by_name)

    # -- the node, both ways -------------------------------------------------

    def invoke(self, state: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        plan = self._plan(state, config)
        if plan is None:
            return {self._messages_key: []}
        calls, decisions, allowed, trimmed = plan
        executed = self._inner.invoke(trimmed, config) if allowed else None
        return self._assemble(calls, decisions, executed, len(state[self._messages_key]))

    async def ainvoke(
        self, state: dict[str, Any], config: RunnableConfig | None = None
    ) -> dict[str, Any]:
        """The async twin. Not optional - see the module docstring."""
        plan = self._plan(state, config)
        if plan is None:
            return {self._messages_key: []}
        calls, decisions, allowed, trimmed = plan
        executed = await self._inner.ainvoke(trimmed, config) if allowed else None
        return self._assemble(calls, decisions, executed, len(state[self._messages_key]))

    __call__ = invoke

    # -- shared between both paths -------------------------------------------

    def _plan(self, state: dict[str, Any], config: RunnableConfig | None) -> _Plan | None:
        """Decide every pending call, and build the state the allowed ones run in.

        Everything here is pure and synchronous, which is what lets the sync and
        async paths share it: the only thing that differs between them is how the
        delegated execution is awaited.
        """
        messages = list(state.get(self._messages_key) or [])
        last = messages[-1] if messages else None
        if not isinstance(last, AIMessage):
            return None
        # `list[Any]` on purpose: LangChain's ToolCall is a TypedDict that mypy
        # narrows to `object` through AIMessage.tool_calls, so dict(c) has no
        # matching overload without widening here first.
        raw: list[Any] = list(last.tool_calls or [])
        calls: list[dict[str, Any]] = [dict(c) for c in raw]
        if not calls:
            return None

        configurable = (config or {}).get("configurable") or {}
        thread_id = str(configurable.get("thread_id") or DEFAULT_THREAD_ID)
        self._middleware.begin_turn(thread_id, len(messages))

        decisions = self._middleware.decide(
            [ToolCall(name=str(c["name"]), args=dict(c.get("args") or {})) for c in calls],
            known_tools=self._known,
        )
        allowed = [c for c, d in zip(calls, decisions, strict=True) if not d.refused]

        # The assistant message is rebuilt carrying ONLY the allowed calls, so the
        # inner node never sees a refused one. Handing it the original and
        # discarding results afterwards would perform the side effect and then
        # hide it, which is the opposite of a gate.
        trimmed = {
            **state,
            self._messages_key: [*messages[:-1], last.model_copy(update={"tool_calls": allowed})],
        }
        return calls, decisions, allowed, trimmed

    def _assemble(
        self,
        calls: list[dict[str, Any]],
        decisions: list[Decision],
        executed: dict[str, Any] | None,
        progress: int,
    ) -> dict[str, Any]:
        """Guard what ran, refuse what did not, and keep the original order."""
        produced: dict[str, ToolMessage] = {}
        for message in (executed or {}).get(self._messages_key) or []:
            if isinstance(message, ToolMessage):
                produced[str(message.tool_call_id)] = message

        out: list[ToolMessage] = []
        for call, decision in zip(calls, decisions, strict=True):
            if decision.refused:
                out.append(self._refusal(call, decision))
                continue
            result = produced.get(str(call.get("id") or ""))
            out.append(self._guard(result) if result is not None else self._missing(call))

        # The second reset signal. Without it a checkpointed thread replayed from
        # an earlier state keeps the taint of the run it replaces, because the id
        # has not changed - and stale evidence refusing a later call is a defense
        # that never fired being reported as one that did.
        self._middleware.state.mark_processed(progress)
        return {self._messages_key: out}

    def _guard(self, message: ToolMessage) -> ToolMessage:
        spans = _spans_of(message)
        guarded = self._middleware.guard(
            ToolOutput(
                tool_name=str(message.name or "unknown_tool"),
                spans=spans,
                evidence="\n".join(s for s in spans if s),
                note="langgraph tool result",
            )
        )
        return _with_spans(message, guarded.spans)

    def _refusal(self, call: dict[str, Any], decision: Decision) -> ToolMessage:
        """What the model reads in place of the result it expected.

        ``status="error"`` because the call genuinely did not happen, and a model
        that reads success here will report the side effect to the user as done.
        The text carries the rest of the contract - do not retry, continue with
        the parts of the task that do not need this - which is what keeps a
        refusal from costing the whole turn.
        """
        return ToolMessage(
            content=decision.refusal_text,
            name=str(call["name"]),
            tool_call_id=str(call.get("id") or ""),
            status="error",
        )

    def _missing(self, call: dict[str, Any]) -> ToolMessage:
        """An allowed call the inner node returned no message for.

        Should not happen. If it does, say so rather than drop the call: a
        ``tool_call`` with no matching ``ToolMessage`` makes most providers reject
        the very next request, which would surface far away from the cause.
        """
        return ToolMessage(
            content=f"No result was produced for {call['name']}.",
            name=str(call["name"]),
            tool_call_id=str(call.get("id") or ""),
            status="error",
        )


# ---------------------------------------------------------------------------
# Content blocks - the part every adapter has to write for itself.
# ---------------------------------------------------------------------------


def _spans_of(message: ToolMessage) -> tuple[str, ...]:
    """The model-visible text fragments of a tool result, in order.

    A non-text block contributes an EMPTY span rather than being skipped, so the
    tuple stays positionally aligned with the content list and
    :func:`_with_spans` can put the guarded text back exactly where it came from.
    """
    content = message.content
    if isinstance(content, str):
        return (content,)
    spans: list[str] = []
    for block in content:
        if isinstance(block, str):
            spans.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            spans.append(str(block.get("text", "")))
        else:
            spans.append("")
    return tuple(spans)


def _with_spans(message: ToolMessage, spans: tuple[str, ...]) -> ToolMessage:
    """Put the guarded spans back where they were found."""
    content = message.content
    if isinstance(content, str):
        return message.model_copy(update={"content": spans[0]})
    rebuilt: list[Any] = []
    for block, span in zip(content, spans, strict=True):
        if isinstance(block, str):
            rebuilt.append(span)
        elif isinstance(block, dict) and block.get("type") == "text":
            rebuilt.append({**block, "text": span})
        else:
            rebuilt.append(block)
    return message.model_copy(update={"content": rebuilt})
