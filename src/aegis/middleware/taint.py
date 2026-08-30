"""What the tools returned this conversation, and when to forget it.

The gate can only refuse a call for a reason the output side observed, so the two
halves of the middleware share exactly one of these. Everything here is plain
Python: no framework, no message shapes, no model.

THE FABRICATION HAZARD
----------------------
A middleware instance is built ONCE and reused across many conversations.
Per-conversation state that is not reset therefore leaks: conversation B would
inherit A's recorded flags, and a later gate consulting them would refuse a call
for a reason that belongs somewhere else. That does not merely add noise - it
manufactures a defense that was never triggered. :meth:`TaintState.begin_turn`
therefore resets on two independent signals, because the cost of a missed reset
is a fabricated result and the cost of a spurious reset is nothing worse than
re-scanning a message.
"""

from __future__ import annotations

from dataclasses import dataclass

from aegis.domain.trust import Tainted, TrustTier, sha256_of
from aegis.security.detector import DetectionResult

__all__ = ["TaintRecord", "TaintState", "conversation_key"]


@dataclass(frozen=True, slots=True)
class TaintRecord:
    """One tool result, as the defense saw it.

    Frozen because a taint record is evidence: a mutable one could be edited
    after the fact by the very code whose decisions it justifies.
    """

    tool_name: str
    """The tool whose output this was, or ``"unknown_tool"`` if the caller could
    not name it."""

    tainted: Tainted[str]
    """The RAW output at T0_UNTRUSTED, carrying provenance and any L3 flags.

    RAW, never the spotlighted rewrite. The gate matches arguments against this
    text, so recording the post-L2 form would let L2 change what L5 decides and
    the ablation could no longer attribute a result to one layer.
    """

    detection: DetectionResult | None = None
    """The L3 result, or ``None`` when L3 was off for this run - which is a
    different statement from "L3 ran and found nothing"."""

    @property
    def tier(self) -> TrustTier:
        return self.tainted.tier

    @property
    def flags(self) -> tuple[str, ...]:
        return self.tainted.detector_flags


def conversation_key(query: str, first_user_text: str | None) -> str:
    """Identify the conversation a turn belongs to.

    Keyed on the request plus the text of the first user message, because those
    two are fixed for the whole of one conversation and differ between
    conversations. Hashed rather than kept verbatim so the state object never
    holds a copy of attacker-controlled prose that could later be logged by
    accident.

    ``first_user_text`` is ``None`` when the caller has no user message to point
    at, which is a DIFFERENT key rather than the same key with an empty string
    appended: a conversation that has not reached its user turn yet must not
    collide with one whose first user message happens to be blank.
    """
    if first_user_text is None:
        return sha256_of(query)
    return sha256_of(query + "\x00" + first_user_text)


class TaintState:
    """Per-conversation record of what the tools returned.

    Holds two things the later capability gate needs: which tools produced
    untrusted output, and which detector flags fired on it. Both are meaningless
    - worse, actively misleading - if they survive into the next conversation, so
    see :meth:`begin_turn` for the reset rule.
    """

    __slots__ = ("_conversation_key", "_processed_messages", "_records")

    def __init__(self) -> None:
        self._conversation_key = ""
        self._processed_messages = 0
        self._records: list[TaintRecord] = []

    # -- lifecycle -----------------------------------------------------------

    def reset(self, conversation_key: str = "") -> None:
        """Drop everything and adopt a new conversation identity."""
        self._conversation_key = conversation_key
        self._processed_messages = 0
        self._records = []

    def begin_turn(self, conversation_key: str, message_count: int) -> bool:
        """Note the start of a turn; reset first if it starts a new conversation.

        Returns True if a reset happened.

        Two independent signals, because either one alone has a blind spot and a
        missed reset fabricates a defense (see the module docstring):

        * the conversation key changed - a different conversation, unless two
          share both the request and the first user message;
        * the history did not grow. Within one conversation the message count only
          ever rises, so a count that is not strictly greater than the last one we
          saw cannot be a continuation of it. This catches the case the key
          misses, including a re-run of the very same task.

        The cost of a false reset is re-scanning messages we already scanned; the
        cost of a missed reset is a wrong number in a paper. The asymmetry is why
        both checks are here.

        The second signal is not hypothetical: AgentDojo's ``task_suite.py`` runs
        every injection variant of a user task against ONE reused pipeline with
        the same ``user_task.PROMPT``, so the key is byte-identical across the
        couples of a task. Without the count check, every couple after the first
        would find ``processed_messages`` already past the end of its own short
        history, be returned untouched, and run UNDEFENDED under a defended label.
        """
        if conversation_key != self._conversation_key or message_count <= self._processed_messages:
            self.reset(conversation_key)
            return True
        return False

    def mark_processed(self, message_count: int) -> None:
        """Record how much of the history has been inspected.

        For a caller whose history is an append-only list, a count is enough to
        find the new entries next turn, and it stops a result being spotlighted
        twice (which would nest one fence inside another).
        """
        self._processed_messages = max(self._processed_messages, message_count)

    # -- recording -----------------------------------------------------------

    def record(self, record: TaintRecord) -> None:
        self._records.append(record)

    # -- queries -------------------------------------------------------------

    @property
    def conversation_key(self) -> str:
        return self._conversation_key

    @property
    def processed_messages(self) -> int:
        return self._processed_messages

    @property
    def records(self) -> tuple[TaintRecord, ...]:
        """Copy, so a caller cannot append evidence that no tool produced."""
        return tuple(self._records)

    @property
    def is_empty(self) -> bool:
        return not self._records

    @property
    def tainted_tools(self) -> frozenset[str]:
        """Tools whose output entered the context as untrusted data."""
        return frozenset(r.tool_name for r in self._records)

    @property
    def flags(self) -> tuple[str, ...]:
        """Every distinct L3 flag raised this conversation, in the order first seen."""
        seen: dict[str, None] = {}
        for record in self._records:
            for flag in record.flags:
                seen.setdefault(flag, None)
        return tuple(seen)

    def flags_for(self, tool_name: str) -> tuple[str, ...]:
        """Distinct flags raised on one tool's output, in the order first seen."""
        seen: dict[str, None] = {}
        for record in self._records:
            if record.tool_name == tool_name:
                for flag in record.flags:
                    seen.setdefault(flag, None)
        return tuple(seen)
