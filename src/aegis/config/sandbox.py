"""The configuration tripwire: refuse to run mock-tool attacks beside a real tool credential.

WHY THIS MODULE EXISTS
----------------------
This repository runs indirect-prompt-injection attacks on purpose. Every one of
them is aimed at AgentDojo's simulated tools, whose ``send_email`` mutates a
dictionary and whose ``delete_file`` deletes nothing. That property - "the tools
are fake" - is the only thing standing between a successful attack in this lab
and a real-world side effect.

It is also a property nobody can see. A mock ``send_email`` and a real one look
identical from the outside, and the difference between them is one import and one
credential. So the failure this module is built for is mundane: an operator who
has just exported a Gmail or GitHub or AWS credential for some unrelated task
starts an attack run in the same shell. Nothing in the run announces that the
blast radius has changed.

The guard is therefore a startup check, not a runtime one, and it fails CLOSED:
under mock tools, a real-tool-shaped credential in the environment stops the
process before the first attack is generated.

WHERE THE LINE IS DRAWN, AND WHY THAT IS THE HARD PART
------------------------------------------------------
A guard that refuses every secret is useless here, because the eval harness
legitimately needs one: the agent under test IS a remote model reached over the
internet, and reaching it needs ``GROQ_API_KEY``. A guard that allows every
secret is theatre. So the boundary is drawn by what a credential BUYS:

* **Model-provider keys** (:data:`MODEL_PROVIDER_ENV_VARS`) buy exactly one
  capability - text completion from a remote endpoint. That is the agent under
  test, by design, in every recorded run. Allowed under mock tools.
* **Tool credentials** - Gmail, GitHub, AWS, Slack, anything else shaped like a
  secret - buy side effects in somebody's real account. The mock suites never
  need one. Their presence under ``AEGIS_TOOLS=mock`` is either a mistake or a
  live credential sitting one import away from an attack loop, and the guard
  cannot tell which. It refuses either way.

Detection is by SHAPE, not by a list of blessed names, because the failure mode
is the credential nobody thought to enumerate: any variable NAME ending in
``API``/``SECRET``/``TOKEN``/``PASSWORD``/``CREDENTIAL``/``KEY``, or any VALUE
carrying a well-known issuer prefix (``ghp_``, ``AKIA``, ``xoxb-``, ...). The
value prefixes are checked even inside an allowlisted model-provider variable: a
GitHub token pasted into ``GROQ_API_KEY`` is still a GitHub token.

VALUES ARE NEVER ECHOED
-----------------------
A tripwire that prints the secret it found is a worse leak than the one it
prevents - it moves the credential into a CI log, a terminal scrollback, and a
bug report. So :class:`CredentialFinding` stores the variable NAME and a reason,
never the value; values are read into a local, compared against a prefix, and
dropped. Nothing here logs, and no error message this module raises can contain a
secret.

THE ESCAPE HATCH IS DELIBERATE, AND SAYS SO
-------------------------------------------
``AEGIS_ALLOW_REAL_CREDENTIALS=true`` turns the guard off. It has to exist: the
day real tool implementations land, somebody has to be able to run them. It is a
separate, explicitly named variable precisely so that turning it on is a decision
a reader can audit in a shell history, rather than a side effect of some other
setting. The refusal message names it, because a guard whose bypass is
undocumented gets bypassed by deleting the guard.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "ALLOW_REAL_CREDENTIALS_ENV",
    "MODEL_PROVIDER_ENV_VARS",
    "TOOL_MODE_ENV",
    "CredentialFinding",
    "FindingKind",
    "SandboxViolation",
    "ToolMode",
    "enforce_sandbox",
    "real_credentials_allowed",
    "resolve_tool_mode",
    "scan_environment",
]

TOOL_MODE_ENV = "AEGIS_TOOLS"
"""Which tool implementations the process intends to use. Default: mock."""

ALLOW_REAL_CREDENTIALS_ENV = "AEGIS_ALLOW_REAL_CREDENTIALS"
"""The one deliberate escape hatch. Nothing else in this repository disables the guard."""


class ToolMode(StrEnum):
    """What the process intends to call. Anything else is a configuration error."""

    MOCK = "mock"
    REAL = "real"


class FindingKind(StrEnum):
    """Why a variable was flagged. Both kinds describe SHAPE, never content."""

    NAME_SHAPE = "name-shape"
    VALUE_PREFIX = "value-prefix"


MODEL_PROVIDER_ENV_VARS: frozenset[str] = frozenset(
    {
        # Every provider this repository can drive the agent under test with.
        # These buy remote text completion and nothing else, which is the one
        # category of secret that is legitimate under mock tools.
        "ANTHROPIC_API_KEY",
        "DEEPSEEK_API_KEY",
        "FIREWORKS_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GROQ_API_KEY",
        "MISTRAL_API_KEY",
        "NVIDIA_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "TOGETHER_API_KEY",
        "XAI_API_KEY",
    }
)
"""Model-side variables: allowed under mock tools because the model is remote by design.

Deliberately NOT here: ``HF_TOKEN`` and friends. A Hugging Face token writes to a
real account, so it is a tool credential wearing a model provider's coat.
"""

_CONTROL_ENV_VARS: frozenset[str] = frozenset(
    {
        # This guard's own switches. ALLOW_REAL_CREDENTIALS ends in "CREDENTIALS"
        # and would otherwise trip the name pattern on itself.
        TOOL_MODE_ENV,
        ALLOW_REAL_CREDENTIALS_ENV,
        "AEGIS_RUN_COSTLY",
    }
)

# A name that ENDS in a secret-ish word, on an underscore boundary so that
# "MONKEY" is not a key and "AWS_SECRET_ACCESS_KEY" is.
_CREDENTIAL_NAME_RE = re.compile(r"(?:^|_)(?:API|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIALS?|KEY)$")

# Issuer prefixes that identify a credential regardless of what it was named.
# Public, documented constants - naming the issuer in an error leaks nothing.
_TOOL_VALUE_PREFIXES: tuple[tuple[str, str], ...] = (
    ("github_pat_", "GitHub"),
    ("ghp_", "GitHub"),
    ("gho_", "GitHub"),
    ("ghu_", "GitHub"),
    ("ghs_", "GitHub"),
    ("glpat-", "GitLab"),
    ("AKIA", "AWS"),
    ("ASIA", "AWS"),
    ("xoxb-", "Slack"),
    ("xoxp-", "Slack"),
    ("xoxa-", "Slack"),
    ("xapp-", "Slack"),
    ("ya29.", "Google OAuth"),
    ("SG.", "SendGrid"),
    ("shpat_", "Shopify"),
)

_MODEL_VALUE_PREFIXES: tuple[tuple[str, str], ...] = (
    ("sk-ant-", "Anthropic"),
    ("sk-", "OpenAI-compatible"),
    ("gsk_", "Groq"),
    ("nvapi-", "NVIDIA"),
    ("AIza", "Google AI Studio"),
)


@dataclass(frozen=True, slots=True)
class CredentialFinding:
    """One flagged variable. Carries the NAME and the reason - never the value.

    The value is deliberately absent from the fields, so that no repr, log line,
    traceback or pytest assertion diff of this object can leak a secret.
    """

    name: str
    kind: FindingKind
    issuer: str = ""

    def describe(self) -> str:
        """One human line, safe to print anywhere."""
        if self.kind is FindingKind.VALUE_PREFIX:
            return f"{self.name} (value carries a well-known {self.issuer} credential prefix)"
        return f"{self.name} (name is shaped like a credential)"


class SandboxViolation(RuntimeError):
    """The process refused to start. The message names variables and no values."""

    def __init__(self, message: str, findings: tuple[CredentialFinding, ...] = ()) -> None:
        super().__init__(message)
        self.findings = findings


def resolve_tool_mode(env: Mapping[str, str] | None = None) -> ToolMode:
    """Read :data:`TOOL_MODE_ENV`, defaulting to mock.

    An unrecognised value is a :class:`SandboxViolation`, not a silent fallback:
    ``AEGIS_TOOLS=mocks`` must never be read as "real tools, guard disabled".
    """
    source = os.environ if env is None else env
    raw = source.get(TOOL_MODE_ENV, ToolMode.MOCK.value).strip().lower()
    try:
        return ToolMode(raw)
    except ValueError:
        allowed = ", ".join(mode.value for mode in ToolMode)
        raise SandboxViolation(
            f"{TOOL_MODE_ENV}={raw!r} is not a tool mode (expected one of: {allowed}). "
            "Refusing to guess, because guessing wrong means running with the guard off."
        ) from None


def real_credentials_allowed(env: Mapping[str, str] | None = None) -> bool:
    """True only if the operator deliberately set the escape hatch."""
    source = os.environ if env is None else env
    return source.get(ALLOW_REAL_CREDENTIALS_ENV, "").strip().lower() in {"true", "1"}


def scan_environment(
    env: Mapping[str, str] | None = None,
    *,
    extra_model_provider_vars: frozenset[str] = frozenset(),
) -> tuple[CredentialFinding, ...]:
    """Return every variable shaped like a REAL TOOL credential, by name only.

    Mode-independent on purpose: a caller that wants the policy asks
    :func:`enforce_sandbox`, and a caller that wants only the observation (the
    test suite's own environment-hygiene fixture) gets it without a mode.

    ``extra_model_provider_vars`` widens the model-side allowlist for a run that
    declares its own key variable - the eval runner's ``--api-key-env`` seam.
    Widening it cannot admit a tool credential: an issuer prefix still fires.
    """
    source = os.environ if env is None else env
    model_vars = MODEL_PROVIDER_ENV_VARS | extra_model_provider_vars
    findings: list[CredentialFinding] = []
    for name in sorted(source):
        if name in _CONTROL_ENV_VARS:
            continue
        value = source[name].strip()
        if not value:
            # An exported-but-empty variable holds no credential. Flagging it
            # would only train operators to ignore this guard.
            continue
        issuer = _match_prefix(value, _TOOL_VALUE_PREFIXES)
        if issuer is not None:
            # Checked BEFORE the model allowlist: a GitHub token pasted into
            # GROQ_API_KEY is a GitHub token in the attack loop.
            findings.append(CredentialFinding(name, FindingKind.VALUE_PREFIX, issuer))
            continue
        if name in model_vars:
            continue
        issuer = _match_prefix(value, _MODEL_VALUE_PREFIXES)
        if issuer is not None:
            # A provider secret under a name no provider in this repo reads: we
            # cannot tell what it unlocks, so it counts.
            findings.append(CredentialFinding(name, FindingKind.VALUE_PREFIX, issuer))
        elif _CREDENTIAL_NAME_RE.search(name.upper()):
            findings.append(CredentialFinding(name, FindingKind.NAME_SHAPE))
    return tuple(findings)


def enforce_sandbox(
    env: Mapping[str, str] | None = None,
    *,
    extra_model_provider_vars: frozenset[str] = frozenset(),
) -> ToolMode:
    """Fail-closed startup guard. Returns the validated mode, or raises.

    Two refusals, one principle - never run side-effecting code the operator did
    not ask for:

    * ``mock`` plus a real-tool-shaped credential: the credential should not be
      here, so stop before the first attack is generated.
    * ``real`` without the escape hatch: real tools may only run when somebody
      said so out loud, in the variable whose whole job is saying so.
    """
    mode = resolve_tool_mode(env)
    if real_credentials_allowed(env):
        return mode
    if mode is ToolMode.REAL:
        raise SandboxViolation(
            f"{TOOL_MODE_ENV}={mode.value} requires {ALLOW_REAL_CREDENTIALS_ENV}=true. "
            "Real tools cause real side effects, so running them is an explicit "
            "decision, never a default."
        )
    findings = scan_environment(env, extra_model_provider_vars=extra_model_provider_vars)
    if not findings:
        return mode
    listed = "; ".join(finding.describe() for finding in findings)
    raise SandboxViolation(
        f"{TOOL_MODE_ENV}={mode.value}, but the environment carries "
        f"{len(findings)} variable(s) shaped like a real tool credential: {listed}. "
        "The mock suites never need one, so this is either a mistake or a live "
        "credential one import away from an attack loop - and this guard cannot "
        f"tell which. Unset them, or set {ALLOW_REAL_CREDENTIALS_ENV}=true to say "
        "you meant it. (Only names are reported; values are never logged or echoed.)",
        findings,
    )


def _match_prefix(value: str, prefixes: tuple[tuple[str, str], ...]) -> str | None:
    """Return the issuer whose prefix ``value`` carries, or None. Never returns the value."""
    for prefix, issuer in prefixes:
        if value.startswith(prefix):
            return issuer
    return None
