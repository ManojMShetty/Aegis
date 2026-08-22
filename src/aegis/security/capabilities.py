"""L5 — the capability gate. The last line of defense, and the only one that
stands between a hijacked plan and a real-world side effect.

WHY THIS LAYER EXISTS
---------------------
Layers L1-L4 try to stop attacker text from ever becoming an instruction. This
layer assumes they *failed*. Its job is to make the failure harmless: even if
the agent has somehow been convinced to call ``send_email(to="attacker@evil")``,
the call does not fire unless the data driving it is trusted enough and the user
actually authorized that class of action.

This is least privilege applied to tool use — capability-based security in the
spirit of CaMeL (DeepMind, 2025), which formalises "untrusted data may inform a
decision, but may never authorise an action".

THE PREDICATE
-------------
A call is allowed only if **all** hold:

1. **Known tool.** Unknown tools are denied (fail closed).
2. **Tier floor.** ``glb(arg tiers) >= policy.min_arg_tier``.
3. **Authorization.** The user's request actually granted this capability.
4. **Clean arguments.** For side-effecting tools, no argument carries a
   blocking detector flag.
5. **High-risk allowlist.** Sensitive arguments (recipients, URLs, amounts)
   must either be allowlisted or come from a non-attacker-influenced tier.

VIOLATIONS ARE COLLECTED, NOT SHORT-CIRCUITED
---------------------------------------------
The gate evaluates every rule even after the first failure. This costs nothing
and buys two things: the eval harness can report *which* layers would have
caught an attack independently (the ablation story), and the demo can show a
single attack being stopped for three unrelated reasons at once — which is what
defense in depth actually looks like.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from aegis.domain.trust import Tainted, TrustTier, glb

__all__ = [
    "BLOCKING_DETECTOR_FLAGS",
    "AuthorizationContext",
    "CapabilityGate",
    "GateDecision",
    "PolicyViolation",
    "ToolPolicy",
    "Verdict",
    "ViolationCode",
]


BLOCKING_DETECTOR_FLAGS: frozenset[str] = frozenset(
    {
        "injection_high_confidence",
        "exfiltration_pattern",
        "tool_invocation_attempt",
        "credential_pattern",
    }
)
"""Detector signals severe enough to block a side-effecting call outright.

Deliberately a *small* set. The detector is advisory (a legitimate document may
quote an attack string), so only high-confidence signals get veto power here;
weaker signals lower a tier or require confirmation instead.
"""


class Verdict(StrEnum):
    """Outcome of a gate evaluation."""

    ALLOW = "allow"
    CONFIRM = "confirm"
    """Permitted only with explicit human confirmation this turn."""
    DENY = "deny"


class ViolationCode(StrEnum):
    """Machine-readable reason a call was refused.

    Stable strings: the eval harness aggregates by these to report *why*
    attacks fail, which is how the per-layer ablation gets its numbers.
    """

    UNKNOWN_TOOL = "unknown_tool"
    TIER_TOO_LOW = "tier_too_low"
    NOT_AUTHORIZED = "not_authorized"
    FLAGGED_ARGUMENT = "flagged_argument"
    TAINTED_SIDE_EFFECT = "tainted_side_effect"
    """An attacker-influenced value is directing a side effect.

    One code covers this rule whether or not an allowlist is configured for the
    argument: the *violation* is the same, and splitting it would make the
    ablation's per-rule aggregation depend on deployment config rather than on
    which defense actually fired. Remediation detail goes in the message.
    """


@dataclass(frozen=True, slots=True)
class PolicyViolation:
    """A single reason a call was refused, with enough detail to explain it."""

    code: ViolationCode
    detail: str
    arg_name: str | None = None

    def __str__(self) -> str:  # pragma: no cover - display only
        where = f" [{self.arg_name}]" if self.arg_name else ""
        return f"{self.code.value}{where}: {self.detail}"


@dataclass(frozen=True, slots=True)
class ToolPolicy:
    """What a tool requires of its arguments before it may fire."""

    name: str

    side_effecting: bool = False
    """Does invoking this change the world (send, write, pay, delete)?

    Read-only tools are far cheaper to allow: the worst case for a poisoned
    ``search()`` is a bad answer, whereas the worst case for a poisoned
    ``send_email()`` is exfiltration.
    """

    min_arg_tier: TrustTier = TrustTier.UNTRUSTED
    """Floor on the GLB of argument tiers."""

    high_risk_args: frozenset[str] = frozenset()
    """Arguments that determine *who or what* is affected — recipients, URLs,
    account numbers, amounts. These get the strictest treatment."""

    allowlists: Mapping[str, frozenset[str]] = field(default_factory=dict)
    """Optional exact-value allowlists for high-risk arguments."""

    requires_confirmation: bool = False
    """Always ask a human, even when every other rule passes."""

    def is_high_risk(self, arg_name: str) -> bool:
        return arg_name in self.high_risk_args


@dataclass(frozen=True, slots=True)
class AuthorizationContext:
    """What the *user* actually asked for this turn.

    The gate's notion of intent. Capabilities are granted by the human (T3), never
    inferred from retrieved content — which is precisely the confusion the whole
    system exists to prevent.
    """

    granted_capabilities: frozenset[str] = frozenset()
    """Tool names the user's request authorizes."""

    confirmed_calls: frozenset[str] = frozenset()
    """Tool names the human explicitly confirmed after being shown the call."""

    allow_all: bool = False
    """Escape hatch for benign-utility measurement runs.

    Needed so the ablation isolates the *security* layers: if the gate refused
    everything for lack of authorization, a utility drop could not be attributed
    to the defenses under test. Never enable in a deployment.
    """

    def grants(self, tool_name: str) -> bool:
        return self.allow_all or tool_name in self.granted_capabilities

    def confirmed(self, tool_name: str) -> bool:
        return tool_name in self.confirmed_calls


@dataclass(frozen=True, slots=True)
class GateDecision:
    """The gate's answer, with every reason it reached that answer."""

    verdict: Verdict
    tool_name: str
    effective_tier: TrustTier
    violations: tuple[PolicyViolation, ...] = ()

    @property
    def allowed(self) -> bool:
        return self.verdict is Verdict.ALLOW

    @property
    def blocked(self) -> bool:
        """Refused outright (as opposed to merely needing confirmation)."""
        return self.verdict is Verdict.DENY

    @property
    def codes(self) -> tuple[ViolationCode, ...]:
        return tuple(v.code for v in self.violations)

    @property
    def independent_block_count(self) -> int:
        """How many *distinct* rules refused this call.

        The defense-in-depth metric: >1 means the attack had to defeat several
        unrelated controls, not just get lucky against one.
        """
        return len(set(self.codes))

    def explain(self) -> str:
        """Human-readable rationale, used in logs, tests, and the demo."""
        head = f"{self.verdict.value.upper()} {self.tool_name} (args @ {self.effective_tier.label})"
        if not self.violations:
            return head
        return head + "\n" + "\n".join(f"  - {v}" for v in self.violations)


class CapabilityGate:
    """Evaluates whether a tool call may fire.

    Stateless and pure: the same inputs always produce the same decision, so the
    gate is fully unit-testable with no model, network, or database. Enforced
    twice in the pipeline — once at plan time in the agent graph, once at
    execution time at the tool boundary — because a check that only runs in the
    planner is a check an exploit can route around.
    """

    def __init__(
        self,
        policies: Iterable[ToolPolicy],
        *,
        blocking_flags: frozenset[str] = BLOCKING_DETECTOR_FLAGS,
        enabled: bool = True,
    ) -> None:
        self._policies: dict[str, ToolPolicy] = {p.name: p for p in policies}
        self._blocking_flags = blocking_flags
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        """When False the gate allows everything — the L5-off ablation arm."""
        return self._enabled

    def policy_for(self, tool_name: str) -> ToolPolicy | None:
        return self._policies.get(tool_name)

    def check(
        self,
        tool_name: str,
        args: Mapping[str, Tainted[Any]],
        authorization: AuthorizationContext,
    ) -> GateDecision:
        """Decide whether ``tool_name`` may be invoked with ``args``.

        Args:
            tool_name: The tool the agent wants to call.
            args: Arguments, each carrying its own trust tier and provenance.
                Untainted arguments are a type error, not a convenience: if a
                value's origin is unknown we cannot reason about it.
            authorization: What the human granted this turn.
        """
        effective_tier = glb(a.tier for a in args.values())

        if not self._enabled:
            return GateDecision(Verdict.ALLOW, tool_name, effective_tier)

        policy = self._policies.get(tool_name)
        if policy is None:
            # Fail closed. An unknown tool is one whose blast radius we have not
            # reasoned about, which is exactly when guessing is most expensive.
            return GateDecision(
                verdict=Verdict.DENY,
                tool_name=tool_name,
                effective_tier=effective_tier,
                violations=(
                    PolicyViolation(
                        ViolationCode.UNKNOWN_TOOL,
                        f"no policy registered for '{tool_name}'; denying by default",
                    ),
                ),
            )

        violations: list[PolicyViolation] = []
        violations += self._check_tier(policy, effective_tier)
        violations += self._check_authorization(policy, authorization)
        violations += self._check_flags(policy, args)
        violations += self._check_high_risk_args(policy, args)

        verdict = self._verdict(policy, authorization, violations)
        return GateDecision(
            verdict=verdict,
            tool_name=tool_name,
            effective_tier=effective_tier,
            violations=tuple(violations),
        )

    # -- individual rules -------------------------------------------------

    def _check_tier(self, policy: ToolPolicy, effective: TrustTier) -> list[PolicyViolation]:
        if effective >= policy.min_arg_tier:
            return []
        return [
            PolicyViolation(
                ViolationCode.TIER_TOO_LOW,
                f"arguments are {effective.label} but '{policy.name}' requires at least "
                f"{policy.min_arg_tier.label}",
            )
        ]

    def _check_authorization(
        self, policy: ToolPolicy, auth: AuthorizationContext
    ) -> list[PolicyViolation]:
        if auth.grants(policy.name):
            return []
        return [
            PolicyViolation(
                ViolationCode.NOT_AUTHORIZED,
                f"the user's request did not grant the '{policy.name}' capability",
            )
        ]

    def _check_flags(
        self, policy: ToolPolicy, args: Mapping[str, Tainted[Any]]
    ) -> list[PolicyViolation]:
        """High-confidence detector signals veto side-effecting calls only.

        Read-only tools are intentionally exempt: letting the detector block
        reads would convert every false positive into a broken benign task,
        which is how a defense quietly destroys utility.
        """
        if not policy.side_effecting:
            return []
        out: list[PolicyViolation] = []
        for name, value in args.items():
            hits = sorted(set(value.detector_flags) & self._blocking_flags)
            if hits:
                out.append(
                    PolicyViolation(
                        ViolationCode.FLAGGED_ARGUMENT,
                        f"argument carries blocking detector flag(s): {', '.join(hits)}",
                        arg_name=name,
                    )
                )
        return out

    def _check_high_risk_args(
        self, policy: ToolPolicy, args: Mapping[str, Tainted[Any]]
    ) -> list[PolicyViolation]:
        """The rule that actually stops exfiltration.

        A high-risk argument decides *who or what* is affected. If its value was
        influenced by an attacker (T0/T1) it may only be used when it matches an
        explicit allowlist. This is what blocks
        ``send_email(to=<address the poisoned page chose>)`` while still allowing
        ``send_email(to=<address the user typed>)``.
        """
        out: list[PolicyViolation] = []
        for name, value in args.items():
            if not policy.is_high_risk(name):
                continue
            if not value.tier.is_attacker_influenced:
                continue  # value came from the user or system: fine

            allowed = policy.allowlists.get(name, frozenset())
            if isinstance(value.value, str) and value.value in allowed:
                continue

            remedy = (
                f"not among the {len(allowed)} allowlisted value(s)"
                if allowed
                else "no allowlist is configured for this argument"
            )
            out.append(
                PolicyViolation(
                    ViolationCode.TAINTED_SIDE_EFFECT,
                    f"high-risk argument derives from {value.tier.label} "
                    f"(attacker-influenced) and {remedy}; "
                    f"sources={list(value.sources)}",
                    arg_name=name,
                )
            )
        return out

    def _verdict(
        self,
        policy: ToolPolicy,
        auth: AuthorizationContext,
        violations: list[PolicyViolation],
    ) -> Verdict:
        if violations:
            return Verdict.DENY
        if policy.requires_confirmation and not auth.confirmed(policy.name):
            return Verdict.CONFIRM
        return Verdict.ALLOW
