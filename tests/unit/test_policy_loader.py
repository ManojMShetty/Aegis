"""Security policy loading — fail-closed behaviour and shipped-config validation."""

from __future__ import annotations

from typing import Any

import pytest

from aegis.config.policy import DEFAULT_POLICY_PATH, PolicyError, SecurityPolicy
from aegis.domain.trust import Tainted, TrustTier
from aegis.security.capabilities import AuthorizationContext, Verdict

pytestmark = pytest.mark.security


MINIMAL: dict[str, Any] = {
    "version": 1,
    "sources": [
        {"match": "kb://*", "tier": "T2_CURATED"},
        {"match": "https://*", "tier": "T0_UNTRUSTED"},
    ],
    "default_tier": "T0_UNTRUSTED",
    "blocking_flags": ["injection_high_confidence"],
    "tools": {
        "search": {"side_effecting": False, "min_arg_tier": "T0_UNTRUSTED"},
        "send_email": {
            "side_effecting": True,
            "min_arg_tier": "T1_QUARANTINE_DERIVED",
            "high_risk_args": ["to"],
            "allowlists": {"to": ["ok@corp.test"]},
        },
    },
}


def policy(**overrides: Any) -> SecurityPolicy:
    raw = {**MINIMAL, **overrides}
    return SecurityPolicy.from_mapping(raw)


# --------------------------------------------------------------------------
# Source resolution
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("kb://handbook/page-1", TrustTier.CURATED),
        ("https://evil.test/page", TrustTier.UNTRUSTED),
        ("gopher://unknown/thing", TrustTier.UNTRUSTED),  # unmatched -> default
    ],
)
def test_resolve_tier(uri: str, expected: TrustTier) -> None:
    assert policy().resolve_tier(uri) is expected


def test_unmatched_source_fails_closed() -> None:
    """A source we could not classify is one we do not understand."""
    p = policy(sources=[])
    assert p.resolve_tier("some://brand-new-scheme") is TrustTier.UNTRUSTED


def test_first_matching_rule_wins() -> None:
    """Ordering is meaningful: specific rules must be able to precede general ones."""
    p = policy(
        sources=[
            {"match": "https://internal.corp.test/*", "tier": "T2_CURATED"},
            {"match": "https://*", "tier": "T0_UNTRUSTED"},
        ]
    )
    assert p.resolve_tier("https://internal.corp.test/doc") is TrustTier.CURATED
    assert p.resolve_tier("https://evil.test/doc") is TrustTier.UNTRUSTED


def test_source_matching_is_case_sensitive() -> None:
    """Case-insensitive matching would be an easy way to smuggle a source past policy."""
    p = policy(sources=[{"match": "kb://*", "tier": "T2_CURATED"}])
    assert p.resolve_tier("KB://handbook") is TrustTier.UNTRUSTED


def test_tier_labels_parse_in_both_forms() -> None:
    assert TrustTier.from_label("T2_CURATED") is TrustTier.CURATED
    assert TrustTier.from_label("curated") is TrustTier.CURATED


def test_unknown_tier_name_raises() -> None:
    with pytest.raises(PolicyError, match="unknown trust tier"):
        policy(default_tier="T9_SUPERTRUSTED")


# --------------------------------------------------------------------------
# Malformed policy must fail loudly at load time
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"sources": [{"tier": "T0_UNTRUSTED"}]}, "missing required key 'match'"),
        ({"sources": [{"match": "x://*"}]}, "missing required key 'tier'"),
        ({"sources": "not-a-list"}, "must be a list"),
        ({"tools": "not-a-mapping"}, "must be a mapping"),
    ],
)
def test_malformed_policy_raises(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(PolicyError, match=message):
        policy(**overrides)


def test_allowlist_on_non_high_risk_arg_is_rejected() -> None:
    """Catches a policy that looks stricter than it is."""
    with pytest.raises(PolicyError, match="have no effect"):
        policy(
            tools={
                "send_email": {
                    "side_effecting": True,
                    "min_arg_tier": "T1_QUARANTINE_DERIVED",
                    "high_risk_args": ["to"],
                    "allowlists": {"body": ["hello"]},
                }
            }
        )


def test_ungateable_side_effecting_tool_is_rejected() -> None:
    """A side-effecting tool the gate could never block is a policy bug, not a choice."""
    with pytest.raises(PolicyError, match="could never block it"):
        policy(
            tools={
                "send_email": {
                    "side_effecting": True,
                    "min_arg_tier": "T0_UNTRUSTED",
                    "high_risk_args": [],
                }
            }
        )


def test_missing_policy_file_raises() -> None:
    with pytest.raises(PolicyError, match="not found"):
        SecurityPolicy.load("does/not/exist.yaml")


# --------------------------------------------------------------------------
# The loaded policy drives a working gate
# --------------------------------------------------------------------------


def test_built_gate_enforces_the_configured_policy() -> None:
    gate = policy().build_gate()
    decision = gate.check(
        "send_email",
        {"to": Tainted.untrusted("attacker@evil.test", source_uri="https://evil.test/p")},
        AuthorizationContext(allow_all=True),
    )
    assert decision.verdict is Verdict.DENY


def test_built_gate_honours_configured_allowlist() -> None:
    gate = policy().build_gate()
    laundered = Tainted(
        value="ok@corp.test",
        tier=TrustTier.QUARANTINE_DERIVED,
        provenance=Tainted.untrusted("x", source_uri="https://evil.test/p").provenance,
    )
    decision = gate.check("send_email", {"to": laundered}, AuthorizationContext(allow_all=True))
    assert decision.verdict is Verdict.ALLOW, decision.explain()


def test_build_gate_disabled_is_the_ablation_arm() -> None:
    """Baseline and defended runs must read the same policy file."""
    gate = policy().build_gate(enabled=False)
    decision = gate.check(
        "send_email",
        {"to": Tainted.untrusted("attacker@evil.test", source_uri="https://evil.test/p")},
        AuthorizationContext(),
    )
    assert decision.verdict is Verdict.ALLOW


def test_blocking_flags_flow_from_config_into_the_gate() -> None:
    gate = policy(blocking_flags=["custom_signal"]).build_gate()
    decision = gate.check(
        "send_email",
        {
            "to": Tainted.trusted(
                "ok@corp.test", TrustTier.USER, source_uri="session://user-turn"
            ).flagged("custom_signal")
        },
        AuthorizationContext(allow_all=True),
    )
    assert decision.verdict is Verdict.DENY


# --------------------------------------------------------------------------
# The policy we actually ship must be valid and sane
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def shipped() -> SecurityPolicy:
    return SecurityPolicy.load()


def test_shipped_policy_file_exists_and_loads(shipped: SecurityPolicy) -> None:
    assert DEFAULT_POLICY_PATH.is_file(), f"expected policy at {DEFAULT_POLICY_PATH}"
    assert shipped.tool_names


def test_shipped_policy_fails_closed_by_default(shipped: SecurityPolicy) -> None:
    assert shipped.default_tier is TrustTier.UNTRUSTED


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("https://evil.test/page", TrustTier.UNTRUSTED),
        ("email://inbound/msg-1", TrustTier.UNTRUSTED),
        ("tool://web_search/result-3", TrustTier.UNTRUSTED),
        ("kb://handbook/onboarding", TrustTier.CURATED),
        ("session://user-turn", TrustTier.USER),
        ("config://system-prompt", TrustTier.SYSTEM),
    ],
)
def test_shipped_policy_classifies_the_obvious_sources(
    shipped: SecurityPolicy, uri: str, expected: TrustTier
) -> None:
    assert shipped.resolve_tier(uri) is expected


def test_every_shipped_side_effecting_tool_is_gateable(shipped: SecurityPolicy) -> None:
    """Each side-effecting tool must have *some* rule that can refuse it.

    Guards against a future edit quietly adding an unprotected tool.
    """
    for p in shipped.tool_policies:
        if not p.side_effecting:
            continue
        assert p.min_arg_tier > TrustTier.UNTRUSTED or p.high_risk_args, (
            f"{p.name} is side-effecting but nothing could block it"
        )


def test_shipped_money_and_delete_tools_demand_user_tier(shipped: SecurityPolicy) -> None:
    """Irreversible actions must not run on quarantine-derived data."""
    for name in ("send_money", "delete_file"):
        p = shipped.policy_for(name)
        assert p is not None, f"{name} missing from shipped policy"
        assert p.min_arg_tier >= TrustTier.USER
        assert p.requires_confirmation
