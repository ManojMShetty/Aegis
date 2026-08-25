"""The import guard: a real tool implementation may not even be IMPORTED by default.

WHY AN IMPORT GUARD AND NOT A CALL-TIME CHECK
---------------------------------------------
The tripwire in :mod:`aegis.config.sandbox` runs once, at startup, and asks
whether the environment looks dangerous. This module asks a narrower question at
a different moment: is this specific module - the one that can actually send an
email - allowed to exist in this process at all?

Import time is the right moment because import is where the damage becomes
possible. A module that opens an SMTP session, constructs an authenticated
client, or registers a tool into a dispatch table has already widened the blast
radius by the time anyone calls it; and a tool registry populated at import is
exactly how a "mock" run acquires a real ``send_email`` without any line of code
saying so. Refusing the import means the dangerous callable never enters the
process, so nothing downstream - no registry, no getattr, no agent tool loop -
can reach it by mistake.

The two guards are deliberately independent. The tripwire can be satisfied
(nobody exported a credential) while the import guard still refuses, and that is
correct: "no credential is lying around" is not the same statement as "this
process may execute real side effects".

WHY THE GUARD EXISTS BEFORE THE THING IT GUARDS
------------------------------------------------
There are no real tool implementations in this repository. This guard is written
first, ON PURPOSE, and that ordering is the point rather than an accident of
scheduling. A guard added after the first real tool lands is a guard written
under deadline pressure, by whoever is trying to get that tool working, in the
one review where "just let it import" is the path of least resistance. Written
first, it is instead a precondition the first real tool has to satisfy:
:mod:`aegis.tools.real_example` shows exactly what that costs (one call, at the
top of the module), so the answer to "how do I add a real tool?" already includes
the guard.

``AEGIS_ALLOW_REAL_CREDENTIALS=true`` is the single escape hatch, shared with the
startup tripwire, so that one auditable variable governs every path from this lab
to a real side effect.
"""

from __future__ import annotations

from collections.abc import Mapping

from aegis.config.sandbox import ALLOW_REAL_CREDENTIALS_ENV, real_credentials_allowed

__all__ = ["RealToolImportError", "require_real_tools_enabled"]


class RealToolImportError(ImportError):
    """A real tool module was imported without the deliberate opt-in.

    Subclasses :class:`ImportError` so that the failure reads, in a traceback and
    to any caller doing ``except ImportError``, as what it is: this module is not
    available in this process.
    """


def require_real_tools_enabled(module_name: str, *, env: Mapping[str, str] | None = None) -> None:
    """Refuse the import unless real tools were deliberately enabled.

    Call at MODULE level, before defining anything that can cause a side effect,
    so the refusal happens while the module body is still executing and the
    dangerous names are never bound.
    """
    if real_credentials_allowed(env):
        return
    raise RealToolImportError(
        f"{module_name} is a REAL tool implementation and cannot be imported "
        f"unless {ALLOW_REAL_CREDENTIALS_ENV}=true. This repository runs "
        "prompt-injection attacks against mock suites; a real tool in the same "
        "process turns a benchmark result into a real-world side effect. If you "
        "meant it, set the variable deliberately - it is deliberately not implied "
        "by anything else."
    )
