"""Loads ``config/trust_tiers.yaml`` into the objects L1 and L5 actually use.

Security posture should be auditable as *data*. A reviewer reads one YAML file
to know what the system trusts and what each tool requires, rather than tracing
Python. This module is the bridge, and it is deliberately strict:

* An unmatched source resolves to :attr:`TrustTier.UNTRUSTED` (fail closed).
* An unlisted tool is denied by the gate (fail closed).
* A malformed policy raises :class:`PolicyError` at load time rather than
  degrading silently — a typo in a security policy must break the build, not
  quietly widen the attack surface.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

import yaml

from aegis.domain.trust import TrustTier
from aegis.security.capabilities import CapabilityGate, ToolPolicy

__all__ = ["PolicyError", "SecurityPolicy", "SourceRule"]

# The policy ships INSIDE the package, and is found relative to this module rather
# than to a repository layout.
#
# It used to live at the repository root and be reached with `parents[3]`, which
# works from a checkout and silently does not work from an install: from
# site-packages that walks to `<venv>/Lib/config/trust_tiers.yaml`, a path nothing
# creates. Every test passed, because every test runs from the checkout. The
# failure was reserved for the first person to `pip install aegis` and call the
# middleware - which is precisely the audience the middleware was extracted for.
#
# A security policy is also not incidental data: it is the file that decides which
# tools are sinks and which arguments are high-risk, so shipping the code without
# it would leave an installed adopter fail-closed on every call with no way to see
# why. It belongs in the distribution.
DEFAULT_POLICY_PATH = Path(__file__).resolve().parent / "trust_tiers.yaml"


class PolicyError(ValueError):
    """The security policy file is malformed or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class SourceRule:
    """Maps a source-URI glob to the tier content from that source starts at."""

    match: str
    tier: TrustTier
    note: str = ""

    def matches(self, source_uri: str) -> bool:
        # Case-sensitive on purpose: URI schemes and paths are case-sensitive,
        # and a case-insensitive match is an easy way to smuggle a source past
        # a policy that looks correct.
        return fnmatchcase(source_uri, self.match)


@dataclass(frozen=True, slots=True)
class SecurityPolicy:
    """The whole deployment security posture, loaded from YAML."""

    source_rules: tuple[SourceRule, ...]
    default_tier: TrustTier
    blocking_flags: frozenset[str]
    tool_policies: tuple[ToolPolicy, ...]
    version: int = 1

    # -- construction ----------------------------------------------------

    @classmethod
    def load(cls, path: Path | str | None = None) -> SecurityPolicy:
        """Load and validate a policy file."""
        p = Path(path) if path is not None else DEFAULT_POLICY_PATH
        if not p.is_file():
            raise PolicyError(f"security policy not found at {p}")
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise PolicyError(f"{p} is not valid YAML: {exc}") from exc
        if not isinstance(raw, Mapping):
            raise PolicyError(f"{p} must contain a mapping at the top level")
        return cls.from_mapping(raw, origin=str(p))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, origin: str = "<memory>") -> SecurityPolicy:
        """Build from an already-parsed mapping (used by tests and by :meth:`load`)."""
        try:
            return cls(
                version=int(raw.get("version", 1)),
                source_rules=_parse_source_rules(raw.get("sources", [])),
                default_tier=_parse_tier(raw.get("default_tier", "T0_UNTRUSTED"), "default_tier"),
                blocking_flags=frozenset(raw.get("blocking_flags", []) or []),
                tool_policies=_parse_tools(raw.get("tools", {}) or {}),
            )
        except PolicyError as exc:
            raise PolicyError(f"{origin}: {exc}") from exc

    # -- use -------------------------------------------------------------

    def resolve_tier(self, source_uri: str) -> TrustTier:
        """The tier content from ``source_uri`` starts at.

        First matching rule wins, so ordering in the YAML is meaningful. An
        unmatched source gets :attr:`default_tier`, which the loader requires to
        be UNTRUSTED — a source we could not classify is one we do not
        understand, and guessing upward is exactly the mistake this system
        exists to prevent.
        """
        for rule in self.source_rules:
            if rule.matches(source_uri):
                return rule.tier
        return self.default_tier

    def build_gate(self, *, enabled: bool = True) -> CapabilityGate:
        """Construct the L5 gate this policy describes.

        ``enabled=False`` yields the ablation's L5-off arm without touching the
        policy file, so the baseline and defended runs read identical config.
        """
        return CapabilityGate(
            self.tool_policies,
            blocking_flags=self.blocking_flags,
            enabled=enabled,
        )

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.tool_policies)

    def policy_for(self, tool_name: str) -> ToolPolicy | None:
        return next((p for p in self.tool_policies if p.name == tool_name), None)


# ---------------------------------------------------------------------------
# parsing helpers
# ---------------------------------------------------------------------------


def _parse_tier(value: Any, where: str) -> TrustTier:
    if not isinstance(value, str):
        raise PolicyError(f"{where}: expected a tier name string, got {type(value).__name__}")
    try:
        return TrustTier.from_label(value)
    except ValueError as exc:
        raise PolicyError(f"{where}: {exc}") from exc


def _parse_source_rules(raw: Any) -> tuple[SourceRule, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, str):
        raise PolicyError("'sources' must be a list of {match, tier} entries")
    rules: list[SourceRule] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, Mapping):
            raise PolicyError(f"sources[{i}]: expected a mapping")
        if "match" not in entry:
            raise PolicyError(f"sources[{i}]: missing required key 'match'")
        if "tier" not in entry:
            raise PolicyError(f"sources[{i}]: missing required key 'tier'")
        rules.append(
            SourceRule(
                match=str(entry["match"]),
                tier=_parse_tier(entry["tier"], f"sources[{i}].tier"),
                note=str(entry.get("note", "")),
            )
        )
    return tuple(rules)


def _parse_tools(raw: Any) -> tuple[ToolPolicy, ...]:
    if not isinstance(raw, Mapping):
        raise PolicyError("'tools' must be a mapping of tool name -> policy")

    policies: list[ToolPolicy] = []
    for name, spec in raw.items():
        if not isinstance(spec, Mapping):
            raise PolicyError(f"tools.{name}: expected a mapping")

        high_risk = frozenset(str(a) for a in (spec.get("high_risk_args") or []))
        allowlists = _parse_allowlists(name, spec.get("allowlists") or {})

        # A configuration mistake worth catching at load time: allowlisting an
        # argument that is not marked high-risk has no effect, which makes the
        # policy look stricter than it is.
        stray = set(allowlists) - high_risk
        if stray:
            raise PolicyError(
                f"tools.{name}: allowlist(s) for {sorted(stray)} have no effect because those "
                f"arguments are not in high_risk_args {sorted(high_risk)}"
            )

        side_effecting = bool(spec.get("side_effecting", False))
        min_tier = _parse_tier(
            spec.get("min_arg_tier", "T0_UNTRUSTED"), f"tools.{name}.min_arg_tier"
        )

        # Another load-time smell: a side-effecting tool that accepts raw
        # untrusted arguments and names nothing high-risk has, in effect, no
        # protection at all. Surface it rather than let it pass review.
        if side_effecting and min_tier is TrustTier.UNTRUSTED and not high_risk:
            raise PolicyError(
                f"tools.{name}: side-effecting tool accepts {TrustTier.UNTRUSTED.label} arguments "
                "and declares no high_risk_args, so the gate could never block it. Raise "
                "min_arg_tier or declare which arguments are high-risk."
            )

        policies.append(
            ToolPolicy(
                name=str(name),
                side_effecting=side_effecting,
                min_arg_tier=min_tier,
                high_risk_args=high_risk,
                allowlists=allowlists,
                requires_confirmation=bool(spec.get("requires_confirmation", False)),
            )
        )
    return tuple(policies)


def _parse_allowlists(tool: str, raw: Any) -> dict[str, frozenset[str]]:
    if not isinstance(raw, Mapping):
        raise PolicyError(f"tools.{tool}.allowlists: expected a mapping of arg -> [values]")
    out: dict[str, frozenset[str]] = {}
    for arg, values in raw.items():
        if values is None:
            out[str(arg)] = frozenset()
            continue
        if isinstance(values, str) or not isinstance(values, Sequence):
            raise PolicyError(f"tools.{tool}.allowlists.{arg}: expected a list of values")
        out[str(arg)] = frozenset(str(v) for v in values)
    return out
