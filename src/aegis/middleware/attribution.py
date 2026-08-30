"""Reconstructing taint at the tool boundary, by VALUE.

An agent framework carries no trust: by the time a tool call exists its arguments
are plain JSON, and whatever the model was reading when it chose them is gone. So
an argument is treated as untrusted when its text appears verbatim in tool output
this conversation already returned, and it inherits that output's tier,
provenance and detector flags. Otherwise it is attributed to the user's request.

The alternative - "once any untrusted output is in the context, every later
argument is untrusted" - is the strictly sound reading of the propagation rule,
and it is unusable: it denies every side effect in every task that reads
something first, which is nearly all of them. Value matching is the weaker rule
that keeps a deployment (and a measurement) meaningful, and it is well matched to
this threat model: the parameters that make an injection dangerous - the
attacker's address, the file id, the IBAN - are values the attacker had to write
into the poisoned text, so they are exactly what matches verbatim.

Everything here is a pure function. There is no state, no model and no framework
in this module, which is why the two floors below can be argued about on their
own terms rather than as a property of one harness.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from aegis.security.spotlight import DEFAULT_DATAMARK

__all__ = [
    "CONTEXT_ARG",
    "MIN_MATCH_CHARS",
    "MIN_SCALAR_MATCH_CHARS",
    "USER_TURN_SOURCE",
    "appears_in",
    "leaf_texts",
    "normalise",
    "subtract_own_args",
]

USER_TURN_SOURCE = "session://user-turn"
"""Provenance URI for an argument we could not trace to any tool output.

``config/trust_tiers.yaml`` maps it to T3_USER. It is an assertion about the only
other place the value could have come from - the user's request, read through the
model - and it is the reason the tier lattice is usable at all here: if every
argument were untrusted the moment the conversation contained one untrusted byte,
the gate would refuse every side effect in every task and the utility number
would be zero by construction.
"""

CONTEXT_ARG = "<conversation>"
"""Synthetic argument name standing in for "the untrusted context of this task".

``glb`` over an EMPTY mapping is the lattice's top (SYSTEM), so a side-effecting
call with no arguments at all would clear every tier floor unconditionally, and
the per-argument rules would have no argument to look at - a gate that can be
bypassed by calling a tool that takes no parameters is not a gate. When a
side-effecting call carries no arguments and untrusted output has already entered
the conversation, this pseudo-argument carries that context's tier AND its
detector flags, so both the tier floor and the blocking-flag rule still have
something to judge.

Under the shipped policy the FLAGS are the half that does the work: no sink there
asks for a floor above T0 any more, because a reachable floor over the GLB of all
arguments is a ban on every side effect that touches anything a tool returned
(see ``config/trust_tiers.yaml``). The tier half stays for the deployments whose
sinks do set one, and for the money/deletion tools that demand T3.
"""

MIN_MATCH_CHARS = 4
"""Shortest argument value that may be attributed to a tool output by matching.

Below this, agreement is coincidence: ``true``, ``1``, ``rw`` and ``id`` appear in
almost any text, and taint attributed by coincidence is a refusal attributed to
nothing.
"""

MIN_SCALAR_MATCH_CHARS = 8
"""The same floor, raised for a value that is not a string.

A string argument may be text the model copied out of a tool result. A number or a
boolean is not copied text at all - it is a RENDERING, and its agreement with
prose is correspondingly cheaper. ``str(False)`` is ``'False'``, which occurs in
any YAML carrying ``recurring: false``; ``str(2024)`` occurs in any document that
mentions the year; both clear :data:`MIN_MATCH_CHARS` comfortably. Attributing
taint that way is not merely noise now that the sinks carry no tier floor: the
coincidentally-matched argument also inherits the matched output's DETECTOR FLAGS,
so a calendar call with ``all_day=False`` would be refused because some flagged
document elsewhere in the task contained the word "false".

Eight characters is where a rendered scalar stops being a plausible coincidence
and starts being an identifier: a year, an hour, a small amount and both booleans
are all shorter, while an account number, an epoch timestamp or a long id is not.
"""

_WHITESPACE = re.compile(r"\s+")

# Zero-width characters the spotlighter's marker neutralisation inserts, plus the
# BOM. They are removed rather than turned into spaces because they are inserted
# INSIDE a word to break it up; turning them into spaces would break the match
# that removing them restores.
_INVISIBLE = ("\u200b", "\ufeff", "\u00ad")


def normalise(text: str) -> str:
    """Fold text into the form arguments and tool outputs are compared in.

    Case, whitespace runs, the spotlight datamark and the zero-width characters
    the marker neutralisation inserts are all removed. The datamark handling is
    the load-bearing part: without it, turning L2 on would silently change what L5
    decides, and the ablation could no longer attribute a result to one layer.
    """
    cleaned = text.replace(DEFAULT_DATAMARK, " ")
    for char in _INVISIBLE:
        cleaned = cleaned.replace(char, "")
    return _WHITESPACE.sub(" ", cleaned).strip().casefold()


def leaf_texts(value: Any) -> list[str]:
    """Every scalar inside an argument value, as text.

    Containers are walked rather than stringified so that
    ``recipients=["attacker@evil.com"]`` is matched on the address itself, not on
    a repr that no tool output would ever contain verbatim.
    """
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [text for item in value.values() for text in leaf_texts(item)]
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [text for item in value for text in leaf_texts(item)]
    # A non-string scalar is only evidence when it is long enough to identify
    # something; see MIN_SCALAR_MATCH_CHARS.
    rendered = str(value)
    return [rendered] if len(rendered) >= MIN_SCALAR_MATCH_CHARS else []


def appears_in(value: Any, normalised_output: str) -> bool:
    """Did any scalar of ``value`` come verbatim out of this tool output?"""
    for text in leaf_texts(value):
        needle = normalise(text)
        if len(needle) >= MIN_MATCH_CHARS and needle in normalised_output:
            return True
    return False


def subtract_own_args(error: str, args: Mapping[str, Any]) -> str:
    """Blank out a call's own argument values inside the error string it produced.

    A plain replace on the raw text: validation errors quote arguments verbatim -
    that is what makes the echo recognisable at all - and anything left behind is no
    worse than not having tried. Values too short to be attributed anyway are left
    alone, because :func:`appears_in` already refuses to trace taint to them.

    WHY THIS SUBTRACTION EXISTS
    ---------------------------
    A framework that answers a call missing a required field with pydantic's
    ``ValidationError`` embeds the input dict VERBATIM, so
    ``send_email(recipients=["dana@corp.example"])`` with no body comes back as an
    error string containing ``dana@corp.example``. Recorded whole, that address
    would sit in the text the gate matches later arguments against, so the agent's
    CORRECTED retry would trace its own recipient to "tool output", find it
    high-risk and attacker-influenced, and be refused - a benign task lost over a
    value no tool ever produced. It is the rule :data:`MIN_MATCH_CHARS` exists for,
    applied to a second source of false attribution: text that agrees with an
    argument for a reason other than provenance is not provenance.

    THE RESIDUAL, STATED
    --------------------
    Subtraction can over-remove. An attack delivered ONLY through an error string,
    whose wording happens to coincide with one of the caller's own arguments, has
    that overlap blanked before the detector ever scans it, so it raises no flags.
    What is NOT weakened is L2: the whole error, subtraction or not, is still
    fenced and marked before the model reads it, and the gate still sees the call.

    What the subtraction cannot remove - anything the ENVIRONMENT put into the
    exception - is exactly what stays, and that is the text this closes the gap on.
    """
    stripped = error
    for text in leaf_texts(dict(args)):
        if len(text) >= MIN_MATCH_CHARS:
            stripped = stripped.replace(text, " ")
    return stripped
