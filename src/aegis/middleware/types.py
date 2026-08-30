"""The whole vocabulary an adopter has to speak.

Four types, and every field in them is something a caller already has at the
moment it calls. Deliberately absent: message lists, environments, runtimes,
content blocks, tool-call ids, and anything else shaped like one framework's
conversation - those are the things a second adopter would otherwise have to
fake in order to use this package.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from aegis.middleware.ledger import GateEntry, refusal_text
from aegis.middleware.taint import TaintRecord
from aegis.security.capabilities import GateDecision

__all__ = ["Decision", "GuardedOutput", "ToolCall", "ToolOutput"]


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A call the agent wants to make: a name and the arguments it chose.

    Nothing else. There is no call id here because the middleware never reads
    one - correlating a decision back to a framework's own call object is the
    adapter's job, and a field this package would only carry around is a field
    that invites an adapter to think it matters.

    ``args`` must be a SNAPSHOT taken at the moment the call was seen, not a live
    view of the framework's own mapping. Some tool loops rewrite arguments in
    place before executing them (AgentDojo turns the string ``"['a@b.example']"``
    into a list), and a middleware that read the mapping afterwards would match
    different text than the model actually emitted - silently, and only for the
    calls that were rewritten.
    """

    name: str
    args: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolOutput:
    """A tool result on its way back into the model's context.

    Three fields, because the text a caller SHOWS the model and the text the
    defense should JUDGE are not always the same string.
    """

    tool_name: str
    """The tool that produced this. The caller just ran it, so it knows; when it
    genuinely cannot say, ``"unknown_tool"`` is the documented fallback - "some
    tool returned untrusted text" is the fact L1 exists to preserve, and losing
    the record entirely is worse than losing the name."""

    spans: tuple[str, ...]
    """The model-visible fragments, each spotlighted INDEPENDENTLY and returned in
    the same order.

    A caller with one string passes a one-tuple. The field is a sequence because
    joining several fragments, marking once and handing back one string would
    change what the model reads: a result that arrives as several blocks would
    come back as one, and any non-text part of it would have to be dropped to make
    that work.
    """

    evidence: str
    """The text L1/L3/L4 judge, and that the gate later matches arguments against.

    Usually the spans joined, which is what :meth:`of` does. It is a separate
    field for the one case where they differ: a framework whose error strings echo
    the agent's own arguments back at it (see
    :func:`aegis.middleware.attribution.subtract_own_args`). Recording that echo
    as evidence would trace the agent's own corrected retry to "tool output" and
    refuse a benign task over a value no tool ever produced.
    """

    note: str = "tool result"
    """Audit breadcrumb recorded on the provenance of this output.

    Free-form and never read by any decision; it exists so a ledger dumped months
    later says which integration wrote the record.
    """

    @classmethod
    def of(cls, tool_name: str, *spans: str, note: str = "tool result") -> ToolOutput:
        """The common case: what the model will read is also what we judge."""
        return cls(
            tool_name=tool_name,
            spans=spans,
            evidence="\n".join(s for s in spans if s),
            note=note,
        )


class GuardedOutput(NamedTuple):
    """What :meth:`AegisMiddleware.guard` hands back.

    ``spans`` has the same length and order as the :class:`ToolOutput` that went
    in, so a caller reinserts them positionally rather than parsing anything. When
    L2 is off it is the very same tuple object, which is how an all-layers-off arm
    stays byte-identical to an undefended run.
    """

    spans: tuple[str, ...]
    record: TaintRecord


@dataclass(frozen=True, slots=True)
class Decision:
    """The answer for one pending call.

    Carries the ledger entry rather than only a boolean because "refused" alone
    cannot be written up: the reason, the tier, the codes and which arguments were
    traced to untrusted output are what separate a defense from a coincidence.
    """

    call: ToolCall
    entry: GateEntry
    gate: GateDecision | None = None
    """The gate's full answer, or ``None`` when the call was not gated at all.

    Not gated means the tool is not in ``known_tools``: there is no side effect to
    prevent, the caller will answer it with its own "no such tool" error, and
    crediting the gate with refusing a hallucinated name would pad the ledger with
    attacks that never had a tool to reach.
    """

    @property
    def refused(self) -> bool:
        return self.entry.refused

    @property
    def refusal_text(self) -> str:
        """What to hand back to the model in place of the result it expected."""
        return refusal_text(self.entry)
