"""L3 - the injection / poisoning detector. Advisory, never the wall.

WHY "ADVISORY" IS THE WHOLE POINT
---------------------------------
A pattern detector cannot be the thing that stops attacks, for two reasons that
pull in opposite directions:

* False positives - a *legitimate* security document may quote "ignore all
  previous instructions" as an example. If a detector hit blocked reads, every
  such document would break the product. So detector hits must not veto benign,
  read-only work.
* False negatives - an adaptive attacker rewrites around any fixed pattern. So
  the detector cannot be trusted to catch everything.

Therefore L3 produces *signals*, and the layer that acts on them is the capability
gate (L5), which lets a small set of high-confidence flags veto side-effecting calls.
:meth:`HeuristicDetector.apply` with ``strict=True`` also offers an ingest-time
downgrade of flagged content, but nothing here calls it: the ingest pipeline never
runs the detector, and the middleware only ever calls :meth:`scan`. The
detector's job is to raise the cost of an attack and to sharpen the signal the
structural layers act on - not to be load-bearing.

This module is the heuristic tier of the intended cascade
(heuristics -> local classifier -> LLM judge). The heuristics are free,
deterministic, and fully offline; the later tiers plug in behind the same
``Detector`` protocol without changing any caller.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, TypeVar

from aegis.domain.trust import Tainted, TrustTier
from aegis.security.spotlight import looks_like_marker

__all__ = [
    "DetectionResult",
    "Detector",
    "HeuristicDetector",
    "Severity",
    "Signal",
    "Verdict",
]

T = TypeVar("T")


class Verdict(StrEnum):
    """Overall read on a piece of content."""

    CLEAN = "clean"
    SUSPECT = "suspect"
    MALICIOUS = "malicious"


class Severity(StrEnum):
    """How much weight a single signal carries."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


_WEIGHT = {Severity.LOW: 0.2, Severity.MEDIUM: 0.5, Severity.HIGH: 0.9}


@dataclass(frozen=True, slots=True)
class Signal:
    """One thing the detector noticed, with the evidence for it."""

    flag: str
    """Stable flag name. High-severity flags reuse the vocabulary the capability
    gate treats as blocking (see ``BLOCKING_DETECTOR_FLAGS``)."""

    category: str
    severity: Severity
    evidence: str
    """The matched snippet, truncated - so a human (or a test) can see *why*."""

    @property
    def weight(self) -> float:
        return _WEIGHT[self.severity]


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Everything the detector concluded about one piece of content."""

    signals: tuple[Signal, ...] = ()
    score: float = 0.0
    verdict: Verdict = Verdict.CLEAN

    @property
    def flags(self) -> tuple[str, ...]:
        """Distinct flag names, order-preserving - what gets attached to a value."""
        return tuple(dict.fromkeys(s.flag for s in self.signals))

    @property
    def is_clean(self) -> bool:
        return self.verdict is Verdict.CLEAN

    def blocking_flags(self, blocking: Iterable[str]) -> tuple[str, ...]:
        """Subset of this result's flags that the gate would treat as blocking."""
        block = set(blocking)
        return tuple(f for f in self.flags if f in block)

    def explain(self) -> str:
        head = f"{self.verdict.value.upper()} (score={self.score:.2f}, {len(self.signals)} signals)"
        if not self.signals:
            return head
        lines = [f"  - [{s.severity.value}] {s.flag}: {s.evidence}" for s in self.signals]
        return head + "\n" + "\n".join(lines)


class Detector(Protocol):
    """The interface every cascade tier implements.

    Keeping this a Protocol is what lets a local classifier or an LLM judge slot
    in later without touching the pipeline: they scan the same text and return
    the same result shape.
    """

    def scan(self, text: str) -> DetectionResult: ...


# ---------------------------------------------------------------------------
# Pattern library. Each entry: (flag, category, severity, compiled pattern).
# Patterns are bounded on purpose - this runs on every retrieved chunk and must
# stay cheap and predictable (no nested unbounded quantifiers).
# ---------------------------------------------------------------------------

_EMAIL = r"[\w.+-]+@[\w-]+\.[\w.-]+"

_PatternSpec = tuple[str, str, Severity, "re.Pattern[str]"]


def _p(flag: str, category: str, severity: Severity, pattern: str) -> _PatternSpec:
    return (flag, category, severity, re.compile(pattern, re.IGNORECASE))


_PATTERNS: tuple[_PatternSpec, ...] = (
    # -- instruction override -------------------------------------------------
    _p(
        "instruction_override",
        "override",
        Severity.HIGH,
        r"\b(?:ignore|disregard|forget|override)\b[^.\n]{0,40}\b"
        r"(?:previous|prior|above|preceding|earlier|all|any)\b[^.\n]{0,20}\b"
        r"(?:instruction|prompt|command|direction|rule|context)s?\b",
    ),
    _p(
        "instruction_override",
        "override",
        Severity.MEDIUM,
        r"\b(?:new|updated|revised|real|actual)\s+(?:instruction|task|prompt|objective)s?\s*:",
    ),
    # -- role / system-prefix manipulation -----------------------------------
    _p(
        "role_manipulation",
        "role",
        Severity.HIGH,
        r"(?:^|\n)\s*(?:system|assistant|developer)\s*:",
    ),
    _p("role_manipulation", "role", Severity.MEDIUM, r"\byou\s+are\s+now\b"),
    _p(
        "role_manipulation",
        "role",
        Severity.MEDIUM,
        r"\b(?:maintenance|developer|debug|admin|god|jailbreak|dan)\s+mode\b",
    ),
    _p(
        "role_manipulation",
        "role",
        Severity.LOW,
        r"\b(?:act\s+as|pretend\s+(?:to\s+be|you))\b",
    ),
    # -- exfiltration --------------------------------------------------------
    _p(
        "exfiltration_pattern",
        "exfiltration",
        Severity.HIGH,
        r"\b(?:forward|send|email|e-mail|leak|exfiltrate|upload|post|transmit|share)\b"
        r"[^.\n]{0,50}?" + _EMAIL,
    ),
    _p(
        "exfiltration_pattern",
        "exfiltration",
        Severity.MEDIUM,
        r"\b(?:forward|send|email|leak|exfiltrate|upload|transmit)\b[^.\n]{0,30}\b"
        r"all\s+(?:the\s+|your\s+|my\s+)?(?:files?|emails?|messages?|data|documents?|"
        r"contacts?|history|secrets?)\b",
    ),
    # -- credentials ---------------------------------------------------------
    _p(
        "credential_pattern",
        "credential",
        Severity.MEDIUM,
        r"\b(?:api[_\s-]?key|secret[_\s-]?key|access[_\s-]?token|auth[_\s-]?token|"
        r"password|passwd|bearer\s+token|private[_\s-]?key)\b",
    ),
    _p(
        "credential_pattern",
        "credential",
        Severity.HIGH,
        r"\b(?:sk-[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{20,}|xox[baprs]-[A-Za-z0-9-]{10,}|"
        r"AKIA[A-Z0-9]{12,})\b",
    ),
    # -- hidden / obfuscated content -----------------------------------------
    _p("hidden_content", "hidden", Severity.MEDIUM, r"<!--.{0,400}?-->"),
    _p(
        "hidden_content",
        "hidden",
        Severity.MEDIUM,
        r"[​‌‍‎‏⁠﻿]",
    ),
    _p("encoded_content", "encoded", Severity.LOW, r"\b[A-Za-z0-9+/]{60,}={0,2}\b"),
)

# When these categories co-occur, we are looking at a coordinated attempt, not a
# stray phrase - escalate to the gate-blocking high-confidence flag.
_ESCALATION_CATEGORIES = {"override", "role", "exfiltration"}


class HeuristicDetector:
    """The free, offline, deterministic tier of the detector cascade."""

    def __init__(
        self,
        *,
        tool_names: Sequence[str] = (),
        enabled: bool = True,
        suspect_threshold: float = 0.35,
        malicious_threshold: float = 0.75,
    ) -> None:
        self._enabled = enabled
        self._suspect = suspect_threshold
        self._malicious = malicious_threshold
        self._tool_patterns = _build_tool_patterns(tool_names)

    @property
    def enabled(self) -> bool:
        """When False, everything scans CLEAN - the L3-off ablation arm."""
        return self._enabled

    def scan(self, text: str) -> DetectionResult:
        if not self._enabled or not text:
            return DetectionResult()

        signals: list[Signal] = []
        for flag, category, severity, pattern in (*_PATTERNS, *self._tool_patterns):
            m = pattern.search(text)
            if m:
                signals.append(
                    Signal(
                        flag=flag,
                        category=category,
                        severity=severity,
                        evidence=_snippet(m.group(0)),
                    )
                )

        if looks_like_marker(text):
            signals.append(
                Signal(
                    flag="spotlight_marker_present",
                    category="marker",
                    severity=Severity.HIGH,
                    evidence="content is shaped like a spotlight fence (break-out attempt)",
                )
            )

        signals = _escalate(signals)
        score = _score(signals)
        verdict = (
            Verdict.MALICIOUS
            if score >= self._malicious
            else Verdict.SUSPECT
            if score >= self._suspect
            else Verdict.CLEAN
        )
        return DetectionResult(signals=tuple(signals), score=score, verdict=verdict)

    def apply(self, tainted: Tainted[T], *, strict: bool = False) -> Tainted[T]:
        """Scan a tainted value and attach any flags found (advisory).

        In ``strict`` mode, a MALICIOUS verdict on above-untrusted content also
        downgrades it to UNTRUSTED: a curated document that reads like an attack
        should not keep its curated standing. Never raises trust (it cannot).
        """
        text = tainted.value if isinstance(tainted.value, str) else str(tainted.value)
        result = self.scan(text)
        flagged = tainted.flagged(*result.flags) if result.flags else tainted
        if strict and result.verdict is Verdict.MALICIOUS and flagged.tier > TrustTier.UNTRUSTED:
            return flagged.downgrade(
                TrustTier.UNTRUSTED, note=f"L3 strict downgrade: {result.verdict.value}"
            )
        return flagged


def _build_tool_patterns(tool_names: Sequence[str]) -> tuple[_PatternSpec, ...]:
    """Detect attempts to invoke the *actual* configured tools by name."""
    names = [re.escape(n) for n in tool_names if n]
    if not names:
        return ()
    alt = "|".join(names)
    return (
        _p(
            "tool_invocation_attempt",
            "tool",
            Severity.HIGH,
            rf"\b(?:use|call|invoke|execute|run|trigger)\s+(?:the\s+)?(?:{alt})\b",
        ),
        _p("tool_invocation_attempt", "tool", Severity.MEDIUM, rf"\b(?:{alt})\s*\("),
    )


def _escalate(signals: list[Signal]) -> list[Signal]:
    """Two or more distinct attack categories => coordinated injection."""
    categories = {s.category for s in signals} & _ESCALATION_CATEGORIES
    if len(categories) >= 2 and not any(s.flag == "injection_high_confidence" for s in signals):
        signals.append(
            Signal(
                flag="injection_high_confidence",
                category="composite",
                severity=Severity.HIGH,
                evidence=f"co-occurring attack categories: {', '.join(sorted(categories))}",
            )
        )
    return signals


def _score(signals: Iterable[Signal]) -> float:
    """Combine weights so any single HIGH signal already reads as malicious,
    while several weak signals accumulate. Saturating, capped at 1.0."""
    product = 1.0
    for s in signals:
        product *= 1.0 - s.weight
    return round(1.0 - product, 4)


def _snippet(text: str, limit: int = 80) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."
