"""A deliberately inert stand-in for a real tool, to demonstrate the import guard.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
This module sends nothing. It has no network client, no credential handling and
no dependency that could acquire either. It exists for two reasons:

1. To make :mod:`aegis.tools.guard` demonstrable and testable. A guard with no
   guarded module is an untested assertion; this gives
   ``tests/security/test_no_egress.py`` something whose import must fail.
2. To be the template. The guard call is the FIRST statement in the module body,
   above every import that could build a client and above every definition, so
   that a refused import leaves nothing behind. A real ``send_email`` added to
   this package copies that shape.

The guard is intentionally written and tested before any real tool exists - see
the module docstring of :mod:`aegis.tools.guard` for why that ordering matters.
Until one does, :func:`send_email` raises, because an inert example that quietly
returned a plausible-looking success would be a worse lie than no example at all.
"""

from __future__ import annotations

from aegis.tools.guard import require_real_tools_enabled

# FIRST, before any definition below is bound. Importing this module without
# AEGIS_ALLOW_REAL_CREDENTIALS=true raises here, and `send_email` never exists.
require_real_tools_enabled(__name__)

__all__ = ["send_email"]


def send_email(to: str, subject: str, body: str) -> None:
    """Would send mail if this were real. It is not, and it never sends anything.

    Raises unconditionally: the point of this module is the guard above it, and a
    stub that returned ``None`` would let a caller believe a message was sent.
    """
    raise NotImplementedError(
        "aegis.tools.real_example is an inert demonstration of the import guard; "
        "no real tool implementation exists in this repository."
    )
