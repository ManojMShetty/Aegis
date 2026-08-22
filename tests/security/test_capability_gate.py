"""L5 capability gate — the invariants that make a hijacked plan harmless.

These tests assume layers L1-L4 have already failed and the agent is trying to
make the attacker's tool call. Everything here is about whether the call fires.
"""

from __future__ import annotations

import pytest

from aegis.domain.trust import Tainted, TrustTier
from aegis.security.capabilities import (
    AuthorizationContext,
    CapabilityGate,
    ToolPolicy,
    Verdict,
    ViolationCode,
)

pytestmark = pytest.mark.security


SEND_EMAIL = ToolPolicy(
    name="send_email",
    side_effecting=True,
    min_arg_tier=TrustTier.QUARANTINE_DERIVED,
    high_risk_args=frozenset({"to"}),
    allowlists={"to": frozenset({"teammate@corp.test"})},
)

SEARCH = ToolPolicy(name="search", side_effecting=False, min_arg_tier=TrustTier.UNTRUSTED)

DELETE_ALL = ToolPolicy(
    name="delete_account",
    side_effecting=True,
    min_arg_tier=TrustTier.USER,
    requires_confirmation=True,
)


@pytest.fixture
def gate() -> CapabilityGate:
    return CapabilityGate([SEND_EMAIL, SEARCH, DELETE_ALL])


def user_arg(value: str) -> Tainted[str]:
    return Tainted.trusted(value, TrustTier.USER, source_uri="session://user-turn")


def poisoned_arg(value: str) -> Tainted[str]:
    return Tainted.untrusted(value, source_uri="https://evil.test/poisoned-page")


# --------------------------------------------------------------------------
# THE WORKED EXAMPLE — the attack this whole project exists to stop
# --------------------------------------------------------------------------


def test_poisoned_page_cannot_exfiltrate_via_send_email(gate: CapabilityGate) -> None:
    """A poisoned page convinced the agent to email an attacker. The gate refuses.

    Critically it refuses for *several independent reasons*: the recipient came
    from untrusted content, the content was flagged, and the user never granted
    the email capability. An attacker would have to defeat all three.
    """
    decision = gate.check(
        "send_email",
        {
            "to": poisoned_arg("attacker@evil.test").flagged("exfiltration_pattern"),
            "body": poisoned_arg("<all the user's files>"),
        },
        AuthorizationContext(granted_capabilities=frozenset({"search"})),
    )

    assert decision.verdict is Verdict.DENY
    assert decision.blocked
    assert decision.independent_block_count >= 3, decision.explain()

    codes = set(decision.codes)
    assert ViolationCode.TIER_TOO_LOW in codes  # args are T0, tool needs >= T1
    assert ViolationCode.NOT_AUTHORIZED in codes  # user only asked to search
    assert ViolationCode.FLAGGED_ARGUMENT in codes  # detector saw exfiltration
    assert ViolationCode.TAINTED_SIDE_EFFECT in codes  # recipient is attacker-chosen


def test_the_same_call_is_allowed_when_the_user_actually_asked_for_it(
    gate: CapabilityGate,
) -> None:
    """The mirror image — the defense must not break the legitimate case.

    Same tool, same shape, but the recipient came from the human and the human
    granted the capability. A defense that blocks this too is useless.
    """
    decision = gate.check(
        "send_email",
        {"to": user_arg("teammate@corp.test"), "body": user_arg("here are the notes")},
        AuthorizationContext(granted_capabilities=frozenset({"send_email"})),
    )
    assert decision.verdict is Verdict.ALLOW, decision.explain()


# --------------------------------------------------------------------------
# Rule 1 — fail closed
# --------------------------------------------------------------------------


def test_unknown_tool_is_denied(gate: CapabilityGate) -> None:
    decision = gate.check("exfiltrate", {}, AuthorizationContext(allow_all=True))
    assert decision.verdict is Verdict.DENY
    assert decision.codes == (ViolationCode.UNKNOWN_TOOL,)


# --------------------------------------------------------------------------
# Rule 2 — tier floor
# --------------------------------------------------------------------------


def test_untrusted_args_cannot_reach_a_tool_requiring_higher_tier(
    gate: CapabilityGate,
) -> None:
    decision = gate.check(
        "send_email",
        {"to": poisoned_arg("teammate@corp.test")},  # allowlisted value, still T0
        AuthorizationContext(allow_all=True),
    )
    assert ViolationCode.TIER_TOO_LOW in set(decision.codes)


def test_effective_tier_is_the_minimum_across_arguments(gate: CapabilityGate) -> None:
    """One untrusted argument taints the whole call — GLB, not average."""
    decision = gate.check(
        "send_email",
        {"to": user_arg("teammate@corp.test"), "body": poisoned_arg("hostile")},
        AuthorizationContext(allow_all=True),
    )
    assert decision.effective_tier is TrustTier.UNTRUSTED


def test_read_only_tool_accepts_untrusted_args(gate: CapabilityGate) -> None:
    """Reads are cheap to allow: worst case is a bad answer, not exfiltration."""
    decision = gate.check(
        "search",
        {"query": poisoned_arg("anything")},
        AuthorizationContext(granted_capabilities=frozenset({"search"})),
    )
    assert decision.verdict is Verdict.ALLOW


# --------------------------------------------------------------------------
# Rule 3 — authorization comes from the human, never from content
# --------------------------------------------------------------------------


def test_capability_not_granted_is_denied(gate: CapabilityGate) -> None:
    decision = gate.check(
        "send_email",
        {"to": user_arg("teammate@corp.test")},
        AuthorizationContext(granted_capabilities=frozenset({"search"})),
    )
    assert ViolationCode.NOT_AUTHORIZED in set(decision.codes)


def test_allow_all_is_scoped_to_utility_runs(gate: CapabilityGate) -> None:
    """The utility-measurement escape hatch relaxes authorization only.

    It must NOT disable the taint rules, or a benign-utility run would silently
    be measuring an undefended system.
    """
    decision = gate.check(
        "send_email",
        {"to": poisoned_arg("attacker@evil.test")},
        AuthorizationContext(allow_all=True),
    )
    assert decision.verdict is Verdict.DENY
    assert ViolationCode.NOT_AUTHORIZED not in set(decision.codes)
    assert ViolationCode.TAINTED_SIDE_EFFECT in set(decision.codes)


# --------------------------------------------------------------------------
# Rule 4 — detector flags veto side effects, but never block reads
# --------------------------------------------------------------------------


def test_blocking_flag_stops_a_side_effecting_call(gate: CapabilityGate) -> None:
    decision = gate.check(
        "send_email",
        {"to": user_arg("teammate@corp.test").flagged("injection_high_confidence")},
        AuthorizationContext(granted_capabilities=frozenset({"send_email"})),
    )
    assert ViolationCode.FLAGGED_ARGUMENT in set(decision.codes)


def test_flags_do_not_block_read_only_tools(gate: CapabilityGate) -> None:
    """False positives must not destroy benign utility.

    A legitimate document may quote 'ignore previous instructions'; refusing to
    read it would trade security for a broken product.
    """
    decision = gate.check(
        "search",
        {"query": poisoned_arg("q").flagged("injection_high_confidence")},
        AuthorizationContext(granted_capabilities=frozenset({"search"})),
    )
    assert decision.verdict is Verdict.ALLOW


def test_low_confidence_flags_do_not_veto(gate: CapabilityGate) -> None:
    """Only high-confidence signals get veto power; weak ones are advisory."""
    decision = gate.check(
        "send_email",
        {"to": user_arg("teammate@corp.test").flagged("mild_suspicion")},
        AuthorizationContext(granted_capabilities=frozenset({"send_email"})),
    )
    assert decision.verdict is Verdict.ALLOW


# --------------------------------------------------------------------------
# Rule 5 — high-risk arguments
# --------------------------------------------------------------------------


def test_quarantine_derived_recipient_is_still_refused(gate: CapabilityGate) -> None:
    """The subtlest and most important case in the whole gate.

    A T1 value passed schema validation — it is a well-formed email address. It
    is *still* the address the attacker chose. Typing constrains shape, not
    intent, so T1 does not earn the right to direct a side effect.
    """
    laundered = Tainted(
        value="attacker@evil.test",
        tier=TrustTier.QUARANTINE_DERIVED,
        provenance=poisoned_arg("x").provenance,
    )
    decision = gate.check(
        "send_email",
        {"to": laundered},
        AuthorizationContext(granted_capabilities=frozenset({"send_email"})),
    )
    assert decision.verdict is Verdict.DENY
    assert ViolationCode.TAINTED_SIDE_EFFECT in set(decision.codes)


def test_allowlisted_value_passes_even_when_attacker_influenced(
    gate: CapabilityGate,
) -> None:
    """The allowlist is the controlled escape hatch that keeps the system usable."""
    laundered = Tainted(
        value="teammate@corp.test",
        tier=TrustTier.QUARANTINE_DERIVED,
        provenance=poisoned_arg("x").provenance,
    )
    decision = gate.check(
        "send_email",
        {"to": laundered},
        AuthorizationContext(granted_capabilities=frozenset({"send_email"})),
    )
    assert decision.verdict is Verdict.ALLOW, decision.explain()


def test_non_high_risk_untrusted_arg_does_not_trigger_the_taint_rule(
    gate: CapabilityGate,
) -> None:
    """Body text may be attacker-influenced; the *recipient* is what matters."""
    decision = gate.check(
        "send_email",
        {
            "to": user_arg("teammate@corp.test"),
            "body": Tainted(
                value="summary of the page",
                tier=TrustTier.QUARANTINE_DERIVED,
                provenance=poisoned_arg("x").provenance,
            ),
        },
        AuthorizationContext(granted_capabilities=frozenset({"send_email"})),
    )
    assert ViolationCode.TAINTED_SIDE_EFFECT not in set(decision.codes)
    assert decision.verdict is Verdict.ALLOW, decision.explain()


# --------------------------------------------------------------------------
# Confirmation
# --------------------------------------------------------------------------


def test_destructive_tool_requires_confirmation(gate: CapabilityGate) -> None:
    decision = gate.check(
        "delete_account",
        {"account": user_arg("acct-1")},
        AuthorizationContext(granted_capabilities=frozenset({"delete_account"})),
    )
    assert decision.verdict is Verdict.CONFIRM
    assert not decision.allowed
    assert not decision.blocked  # CONFIRM is distinct from DENY


def test_confirmation_satisfied_allows(gate: CapabilityGate) -> None:
    decision = gate.check(
        "delete_account",
        {"account": user_arg("acct-1")},
        AuthorizationContext(
            granted_capabilities=frozenset({"delete_account"}),
            confirmed_calls=frozenset({"delete_account"}),
        ),
    )
    assert decision.verdict is Verdict.ALLOW


def test_violations_outrank_confirmation(gate: CapabilityGate) -> None:
    """A confirmed call with a taint violation is still denied, not merely asked about."""
    decision = gate.check(
        "delete_account",
        {"account": poisoned_arg("acct-1")},
        AuthorizationContext(
            granted_capabilities=frozenset({"delete_account"}),
            confirmed_calls=frozenset({"delete_account"}),
        ),
    )
    assert decision.verdict is Verdict.DENY


# --------------------------------------------------------------------------
# Ablation support — the gate must be cleanly switchable off
# --------------------------------------------------------------------------


def test_disabled_gate_allows_everything() -> None:
    """The L5-off arm of the ablation. Also the shape of the baseline run."""
    off = CapabilityGate([SEND_EMAIL], enabled=False)
    decision = off.check(
        "send_email",
        {"to": poisoned_arg("attacker@evil.test").flagged("exfiltration_pattern")},
        AuthorizationContext(),
    )
    assert decision.verdict is Verdict.ALLOW
    assert decision.violations == ()


def test_disabled_gate_still_reports_effective_tier() -> None:
    """Even switched off it must observe, so the baseline run yields taint telemetry."""
    off = CapabilityGate([SEND_EMAIL], enabled=False)
    decision = off.check("send_email", {"to": poisoned_arg("x")}, AuthorizationContext())
    assert decision.effective_tier is TrustTier.UNTRUSTED


def test_explain_is_human_readable(gate: CapabilityGate) -> None:
    decision = gate.check(
        "send_email",
        {"to": poisoned_arg("attacker@evil.test")},
        AuthorizationContext(),
    )
    text = decision.explain()
    assert "DENY send_email" in text
    assert "T0_UNTRUSTED" in text
    assert text.count("\n  - ") == len(decision.violations)


def test_no_args_yields_lattice_top(gate: CapabilityGate) -> None:
    """A call with no arguments has nothing untrusted in it; GLB of nothing is top."""
    decision = gate.check("search", {}, AuthorizationContext(allow_all=True))
    assert decision.effective_tier is TrustTier.SYSTEM
    assert decision.verdict is Verdict.ALLOW
