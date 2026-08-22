"""Defense layers L1-L5.

Each layer is independently toggleable so the ablation is "flip one node" rather
than "maintain a fork". No layer is load-bearing alone; that is the point.

    L1 provenance  -> aegis.domain.trust (the lattice itself)
    L2 spotlight   -> aegis.security.spotlight
    L3 detector    -> aegis.security.detector
    L4 quarantine  -> aegis.security.quarantine
    L5 capability  -> aegis.security.capabilities
"""

from aegis.security.capabilities import (
    AuthorizationContext,
    CapabilityGate,
    GateDecision,
    PolicyViolation,
    ToolPolicy,
    Verdict,
    ViolationCode,
)
from aegis.security.detector import (
    DetectionResult,
    Detector,
    HeuristicDetector,
    Severity,
    Signal,
)
from aegis.security.detector import Verdict as DetectorVerdict
from aegis.security.quarantine import (
    QuarantineError,
    QuarantineExtractor,
    gemini_schema_for,
)
from aegis.security.spotlight import (
    DEFAULT_DATAMARK,
    SpotlightedText,
    Spotlighter,
    SpotlightStyle,
)

__all__ = [
    "DEFAULT_DATAMARK",
    "AuthorizationContext",
    "CapabilityGate",
    "DetectionResult",
    "Detector",
    "DetectorVerdict",
    "GateDecision",
    "HeuristicDetector",
    "PolicyViolation",
    "QuarantineError",
    "QuarantineExtractor",
    "Severity",
    "Signal",
    "SpotlightStyle",
    "SpotlightedText",
    "Spotlighter",
    "ToolPolicy",
    "Verdict",
    "ViolationCode",
    "gemini_schema_for",
]
