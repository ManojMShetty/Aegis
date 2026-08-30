"""What the middleware DID, kept as evidence, and what it tells the model.

A gate that refuses without a ledger is indistinguishable from a model that
happened not to take the bait - both look like "the attack did not land". The
entry is what makes the first claim checkable, and the refusal text is what makes
the refusal survivable: an agent that cannot hear "no" and carry on costs the
whole task, which is the other half of any honest measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from aegis.domain.trust import TrustTier
from aegis.security.capabilities import Verdict as GateVerdict
from aegis.security.capabilities import ViolationCode

__all__ = ["TAINT_VIOLATIONS", "GateAction", "GateEntry", "refusal_text"]

TAINT_VIOLATIONS = frozenset(
    {
        ViolationCode.TIER_TOO_LOW,
        ViolationCode.FLAGGED_ARGUMENT,
        ViolationCode.TAINTED_SIDE_EFFECT,
    }
)
"""The violation codes that are consequences of TAINT rather than of intent.

A read-only tool is never refused for one of these; see
:meth:`aegis.middleware.runtime.AegisMiddleware._action_for`.
"""


class GateAction(StrEnum):
    """What the middleware actually DID with a call, as opposed to what the gate said.

    Separate from :class:`aegis.security.capabilities.Verdict` because the two
    genuinely differ twice: CONFIRM cannot be executed where there is no human to
    ask, and a DENY on a read-only tool is overridden. A ledger that recorded only
    one of the two could not tell "the gate objected" from "the call did not
    happen".
    """

    EXECUTED = "executed"
    REFUSED = "refused"


@dataclass(frozen=True, slots=True)
class GateEntry:
    """One gate decision, kept as evidence.

    This is the record that distinguishes "the attack failed because the gate
    refused the call" from "the attack failed because the model happened not to
    take the bait" - two outcomes that are indistinguishable in a benchmark's
    own success boolean, and only one of which is a defense. Without a ledger, an
    ASR of 0 is an anecdote.

    Frozen for the same reason :class:`aegis.middleware.taint.TaintRecord` is:
    evidence that the reporting code can edit is not evidence.
    """

    conversation_key: str
    """Which conversation this decision belongs to, so the ledger can be split."""

    tool_name: str
    verdict: GateVerdict
    """What :meth:`aegis.security.capabilities.CapabilityGate.check` returned."""

    action: GateAction
    """What the middleware did about it."""

    effective_tier: TrustTier
    codes: tuple[ViolationCode, ...] = ()
    reason: str = ""
    """The gate's own ``explain()`` output, verbatim."""

    tainted_args: tuple[str, ...] = ()
    """Arguments traced back to untrusted tool output, in call order."""

    note: str = ""
    """Why action and verdict differ, when they do."""

    @property
    def refused(self) -> bool:
        return self.action is GateAction.REFUSED

    @property
    def independent_block_count(self) -> int:
        """How many DISTINCT rules objected - the defense-in-depth number."""
        return len(set(self.codes))


def refusal_text(entry: GateEntry) -> str:
    """Explain the refusal to the model without inviting a retry loop.

    Three things have to be in here. That nothing happened, so the agent does not
    assume the side effect landed and report it to the user as done. That retrying
    is pointless, because a refusal the model reads as a transient error costs the
    rest of the turn budget in identical retries. And that the task is not over,
    because the user's own request usually has parts this call was not needed for -
    utility is half the measurement.
    """
    parts = [
        "Refused by the Aegis capability gate. This call was NOT executed and nothing "
        "in the environment changed.",
        entry.reason,
    ]
    if entry.note:
        parts.append(entry.note)
    parts.append(
        "This is a security policy decision, not a transient tool error: the identical "
        "call will be refused again, so do not retry it. Continue with any part of the "
        "user's request that does not require this action, and tell the user plainly "
        "which action was refused and why."
    )
    return "\n\n".join(p for p in parts if p)
