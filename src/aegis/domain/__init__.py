"""Pure domain model — no I/O, no network, no framework imports.

Everything here is deterministic and unit-testable without Docker, Postgres,
or an API key. The security invariants live here on purpose: they must be
verifiable in isolation.
"""

from aegis.domain.trust import (
    DeclassificationError,
    Provenance,
    QuarantineAttestation,
    Tainted,
    TrustTier,
    combine_all,
    declassify_via_quarantine,
    glb,
    sha256_of,
)

__all__ = [
    "DeclassificationError",
    "Provenance",
    "QuarantineAttestation",
    "Tainted",
    "TrustTier",
    "combine_all",
    "declassify_via_quarantine",
    "glb",
    "sha256_of",
]
