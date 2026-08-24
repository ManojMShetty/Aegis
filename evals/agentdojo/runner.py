"""The AgentDojo runner: produce the undefended baseline number - and the defended
one it is compared against - frugally.

WHAT IT PRODUCES
----------------
One JSON file and one printed table with three numbers for an undefended
agent-under-test on an AgentDojo suite:

* ``utility``               - fraction of user tasks solved with no attack present.
* ``asr``                   - attack success rate: fraction of (user, injection)
                              couples where the injected instruction hijacked the
                              agent (AgentDojo's ``security_results`` is ``True``
                              when the ATTACK succeeded).
* ``utility_under_attack``  - fraction of the same injected couples where the
                              user's own task still got done.

THE PROVIDER SEAM
-----------------
The agent-under-test is chosen with ``--provider``:

* ``openai-compat`` (the default, primary path) runs AgentDojo's built-in,
  battle-tested ``OpenAILLM`` against any OpenAI-compatible endpoint. Nothing
  custom sits between the runner and the model; see
  :mod:`evals.agentdojo.openai_llm`.
* ``gemini`` (secondary, experimental) runs the ``thought_signature``-preserving
  element in :mod:`evals.agentdojo.gemini_llm`, which Gemini 3.x needs to survive a
  multi-turn tool loop at all.

Selecting a provider switches the element AND the sensible defaults for the model
and the key's environment variable; an explicit ``--model`` (or ``--api-key-env``)
is always honoured for whichever provider is selected. Each provider module owns
its own key handling and exposes one builder, so :func:`_build_element` is a pure
two-way dispatch and no provider-specific knowledge (or error text) lives here.

WHY THE DEFAULTS ARE GROQ + gpt-oss-120b
-----------------------------------------
These defaults are not a guess - they are the configuration the recorded Week-0
baseline ran on: ``openai/gpt-oss-120b`` via ``https://api.groq.com/openai/v1`` at
``--reasoning-effort low``. That combination is what a security baseline needs: a
model capable enough to actually complete the user tasks, AND measurably
hijackable, so a later defended run has a real number to improve on.

The recorded numbers - utility 100%, ASR 8.3% over 12 injected couples - came from
that configuration run at ``--max-tasks 3 --max-injection-tasks 4`` (3 x 4 = 12
couples). The DEFAULT caps here are smaller (:data:`DEFAULT_MAX_TASKS` and
:data:`DEFAULT_MAX_INJECTION_TASKS`), because the default spend profile is
deliberately cheap; reproducing the quoted number means passing those two flags.

The NVIDIA path (``meta/llama-3.1-8b-instruct`` via
``https://integrate.api.nvidia.com/v1``) remains fully supported through flags and
its endpoint quirks are still handled, but it is NOT a valid security baseline: an
8b model that cannot reliably drive the tool loop scores near-0% ASR out of
INCAPABILITY, not out of resistance, and "the attack failed because the agent failed"
is not a defensive result. Never quote a number from it as a baseline.

WHY EVERY RUN IS A FRESH RUN BY DEFAULT
----------------------------------------
AgentDojo caches each task result on disk and, unless told to force a rerun, it
REPLAYS that cache instead of calling the model. Its cache key is
``logdir/pipeline_name/suite/user_task/attack/{injection_task}.json`` - so the only
thing separating one run's results from another's is ``pipeline.name``. Two
defenses keep a replay from being written up as a measurement:

* ``force_rerun`` is TRUE by default at both benchmark call sites. ``--resume`` opts
  back into the cache, which is what an interrupted long run wants and nothing else
  does. The effective value is recorded as ``force_rerun`` in the result JSON.
* ``pipeline.name`` carries a fingerprint of the configuration that could change a
  score (see :func:`_pipeline_name`), not just the model. Otherwise a re-run at a
  different reasoning effort - or against a different endpoint with the same model
  id - would silently inherit the previous run's numbers and stamp the NEW settings
  on them.

Belt and braces: if a run ends having made zero model requests while still
reporting metrics, it was served entirely from cache. That is recorded as
``"replayed": true`` and shouted about in the console summary rather than passed
off as a fresh measurement.

WHY FRUGALITY IS ENFORCED HERE, NOT IN aegis.llm.budget
-------------------------------------------------------
Aegis has a budget guard, but AgentDojo drives the model through its OWN client, so
that guard never sees these calls. The spend ceiling therefore has to live in this
runner. Three mechanisms do it:

* a hard ``--max-tasks`` cap that bounds how many user tasks run in each phase - the
  runner refuses a non-positive value and never runs more than the cap;
* a ``--max-injection-tasks`` cap on the injection set; and
* the element's own counters, reported at the end, so every run states exactly how
  much model traffic it made.

The injected phase is NOT simply (tasks x injections):
``benchmark_suite_with_injections`` first runs each injection task once STANDALONE
as a user task, to check the injection goal is achievable at all, so the real cost
is (tasks x injections) + injections task runs. At the defaults that is 4 clean + 2
standalone + 8 injected = 14 task runs, and each task run is a whole multi-turn tool
loop, not one request.

The two caps are independent on purpose. Passing ``--max-injection-tasks`` IS the
operator's explicit consent to that much spend, so it is not silently clamped to
``--max-tasks``; the DEFAULT is kept small (2) so the default spend profile of a
plain ``python -m evals.agentdojo.runner`` is unchanged.

Two traffic counters are recorded because two honest units exist and they are not
the same number: ``total_model_calls`` counts API request ATTEMPTS (retries
included - that is what quota is spent on) and ``total_model_turns`` counts model
TURNS (agent decisions). Both elements define them identically, so the two fields
mean the same thing whichever provider produced them.

Task ids are chosen by numeric order of their trailing index, because AgentDojo
stores them in registration order (``user_task_0, user_task_1, user_task_3, ...``)
- "the first N" has to mean 0..N-1, not whatever order a dict happens to yield.

THE DEFENSE SEAM
----------------
``--defense`` selects what sits around the agent:

* ``none`` (the default) builds AgentDojo's stock ``defense=None`` pipeline. The
  recorded baseline command is un-flagged, so it keeps behaving exactly as it did
  before the seam was filled - including the pipeline NAME it caches under.
* ``aegis`` builds the defended pipeline from :mod:`evals.agentdojo.defense`,
  whose layers are chosen with ``--defense-layers``.

``--defense-layers`` takes a comma list of ``spotlight``, ``detect`` and ``gate``
(L2/L3/L5), plus the two aliases ``all`` and ``none``; it is REFUSED with
``--defense none``, the same way the openai-compat-only flags are refused for
gemini, because a flag that cannot do anything is a lie about the run. L1
(provenance) is not listed: it is not toggleable. L4 (quarantine) is not listed
either - it needs an extractor model and costs one extra call per tool result,
which is a spend decision this runner does not make on the operator's behalf.

Three things follow the defense identity out of this module, and each one exists
to stop a specific way of publishing a wrong number:

* the result JSON records ``defense_name`` and the exact ``defense_layers`` set,
  because two runs are comparable only if a reader can see what was on;
* that identity is fingerprinted into ``pipeline.name`` (see
  :func:`_pipeline_name`), so a defended run can NEVER be served from the
  undefended run's on-disk cache - the fingerprint is the only mechanism
  AgentDojo gives us for that;
* the gate's ledger is summarised into ``gate_ledger``. AgentDojo's
  ``security_results`` says only whether the attack succeeded; it cannot
  distinguish "the gate refused the call" from "the model happened not to take
  the bait", and only the first of those is a defense.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, NamedTuple

from agentdojo.agent_pipeline import AgentPipeline, PipelineConfig
from agentdojo.attacks import load_attack
from agentdojo.benchmark import (
    benchmark_suite_with_injections,
    benchmark_suite_without_injections,
)
from agentdojo.logging import OutputLogger
from agentdojo.task_suite.load_suites import get_suite

from aegis.security.capabilities import Verdict as GateVerdict
from evals.agentdojo.defense import (
    AegisPipeline,
    DefenseConfig,
    GateAction,
    build_aegis_pipeline,
)
from evals.agentdojo.gemini_llm import build_gemini_llm
from evals.agentdojo.openai_llm import (
    DEFAULT_MIN_REQUEST_INTERVAL_S,
    REASONING_EFFORT_CHOICES,
    build_openai_compat_llm,
)


class Provider(StrEnum):
    """The agent-under-test backends the runner knows how to wire up.

    ``openai-compat`` is the primary path (AgentDojo's stock ``OpenAILLM`` over an
    OpenAI-compatible endpoint); ``gemini`` is the secondary, experimental path
    (the ``thought_signature`` element). The string values are exactly the
    ``--provider`` choices.
    """

    OPENAI_COMPAT = "openai-compat"
    GEMINI = "gemini"


class Defense(StrEnum):
    """What sits around the agent-under-test. The values are the ``--defense``
    choices.

    ``none`` is AgentDojo's stock undefended pipeline and is the default, so the
    recorded baseline command keeps measuring the same thing it always did.
    """

    NONE = "none"
    AEGIS = "aegis"


class DefenseLayer(StrEnum):
    """The Aegis layers ``--defense-layers`` can turn on, one CLI name each.

    L1 (provenance) is absent because it is not toggleable - it changes nothing
    the model reads, so there is no arm in which it is off. L4 (quarantine) is
    absent for a different reason: it needs an extractor model and costs one extra
    model call PER TOOL RESULT, which is a spend decision that belongs to whoever
    is paying, not to a flag that reads like the others.
    """

    SPOTLIGHT = "spotlight"
    DETECT = "detect"
    GATE = "gate"


# Defaults, named so the CLI help and the values agree in one place.
DEFAULT_PROVIDER = Provider.OPENAI_COMPAT
DEFAULT_SUITE = "workspace"
# The recorded Week-0 baseline configuration: capable enough to solve the user
# tasks AND measurably hijackable (see the module docstring). Changing any of
# these three changes what "the baseline" means, so they move together.
DEFAULT_OPENAI_MODEL = "openai/gpt-oss-120b"
DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_OPENAI_KEY_ENV = "GROQ_API_KEY"
# gpt-oss-120b is a reasoning model; "low" is what made a multi-task run feasible
# on a free tier without costing utility (100% in the recorded run).
DEFAULT_REASONING_EFFORT = "low"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
DEFAULT_GEMINI_KEY_ENV = "GEMINI_API_KEY"
DEFAULT_TIMEOUT = 90.0
DEFAULT_ATTACK = "important_instructions"
DEFAULT_MAX_TASKS = 4
# The injected phase runs (user tasks) x (injection tasks) PLUS one standalone run
# per injection task. Two keeps the default spend profile small;
# --max-injection-tasks raises it deliberately.
DEFAULT_MAX_INJECTION_TASKS = 2
DEFAULT_BENCHMARK_VERSION = "v1.2"
# Undefended by default: an un-flagged run is the recorded Week-0 baseline, and it
# must keep behaving exactly as it did before this seam was filled.
DEFAULT_DEFENSE = Defense.NONE
# What --defense aegis means when --defense-layers is not given: every toggleable
# layer this runner exposes. An ablation arm is then always an explicit request.
DEFAULT_DEFENSE_LAYERS: tuple[DefenseLayer, ...] = tuple(DefenseLayer)
# The two aliases --defense-layers accepts alongside the layer names. They are
# whole-list words, not layers, so they cannot be combined with anything.
DEFENSE_LAYERS_ALL = "all"
DEFENSE_LAYERS_NONE = "none"
DEFAULT_LOGDIR = Path("results/raw/agentdojo_logs")
DEFAULT_OUT = Path("results/week0_baseline.json")

# The fully-supported secondary endpoint, named here only so the --help usage
# example and this module's docstring cannot drift from each other.
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_KEY_ENV = "NVIDIA_API_KEY"
NVIDIA_MODEL = "meta/llama-3.1-8b-instruct"

# The caps the recorded Week-0 number was measured at: 3 x 4 = the 12 injected
# couples the quoted ASR is a fraction of. Named so the docs, the epilog and any
# reproduction instruction cannot drift from each other - these are NOT the
# defaults (see DEFAULT_MAX_TASKS), because the default spend profile is smaller.
RECORDED_BASELINE_MAX_TASKS = 3
RECORDED_BASELINE_MAX_INJECTION_TASKS = 4

# Per-provider defaults, resolved when --model / --api-key-env are not supplied.
_PROVIDER_DEFAULT_MODEL = {
    Provider.OPENAI_COMPAT: DEFAULT_OPENAI_MODEL,
    Provider.GEMINI: DEFAULT_GEMINI_MODEL,
}
_PROVIDER_DEFAULT_KEY_ENV = {
    Provider.OPENAI_COMPAT: DEFAULT_OPENAI_KEY_ENV,
    Provider.GEMINI: DEFAULT_GEMINI_KEY_ENV,
}

_TRAILING_INT = re.compile(r"(\d+)$")

# -- the defense selection ------------------------------------------------------


def _valid_layer_names() -> str:
    """The layer vocabulary, for an error message that can be acted on."""
    names = ", ".join(layer.value for layer in DefenseLayer)
    return f"{names} (or the whole-list aliases '{DEFENSE_LAYERS_ALL}', '{DEFENSE_LAYERS_NONE}')"


def parse_defense_layers(spec: str) -> tuple[DefenseLayer, ...]:
    """Turn a ``--defense-layers`` spec into the layers it names, in layer order.

    Raises ``ValueError`` naming every valid layer, which the CLI turns into a
    ``parser.error``. Refusing an unknown name is not pedantry: a typo that parsed
    as "layer off" would run an ablation arm nobody asked for and label it with the
    arm they did ask for, which is a wrong number rather than a failed run.

    ``all`` and ``none`` are whole-list words and may not be combined with anything
    else - ``none,gate`` is a contradiction and ``all,gate`` is a claim about
    precedence that this function should not have to guess at.
    """
    tokens = [token.strip().casefold() for token in spec.split(",")]
    if any(not token for token in tokens):
        raise ValueError(
            f"--defense-layers has an empty entry in {spec!r}; "
            f"pass a comma-separated list of: {_valid_layer_names()}"
        )
    aliases = {DEFENSE_LAYERS_ALL: DEFAULT_DEFENSE_LAYERS, DEFENSE_LAYERS_NONE: ()}
    for alias, layers in aliases.items():
        if alias in tokens:
            if len(tokens) > 1:
                raise ValueError(
                    f"--defense-layers '{alias}' names the whole set and cannot be combined "
                    f"with other entries (got {spec!r}); pass either '{alias}' alone or an "
                    f"explicit list of: {_valid_layer_names()}"
                )
            return layers
    known = {layer.value: layer for layer in DefenseLayer}
    unknown = [token for token in tokens if token not in known]
    if unknown:
        raise ValueError(
            f"--defense-layers got unknown layer(s) {', '.join(repr(u) for u in unknown)}; "
            f"valid layers are: {_valid_layer_names()}"
        )
    selected = {known[token] for token in tokens}
    return tuple(layer for layer in DefenseLayer if layer in selected)


def _resolve_defense_layers(
    defense: Defense, layers: Sequence[str] | None
) -> tuple[DefenseLayer, ...]:
    """Normalise the layer set for a defense, refusing the combinations that lie.

    The same two rules the CLI enforces, applied again for callers that reach
    ``run_baseline`` directly - the ``--max-tasks`` guard is doubled up for exactly
    this reason. Naming layers for the undefended pipeline is refused rather than
    ignored, because a result JSON that recorded them would describe a defense that
    was never built.
    """
    if defense is Defense.NONE:
        if layers:
            raise ValueError(
                "--defense-layers applies to --defense aegis only; the undefended "
                f"pipeline has no layers to turn on (got {list(layers)})"
            )
        return ()
    if layers is None:
        return DEFAULT_DEFENSE_LAYERS
    known = {layer.value: layer for layer in DefenseLayer}
    unknown = [str(name) for name in layers if str(name) not in known]
    if unknown:
        raise ValueError(
            f"unknown defense layer(s) {', '.join(repr(u) for u in unknown)}; "
            f"valid layers are: {_valid_layer_names()}"
        )
    selected = {known[str(name)] for name in layers}
    return tuple(layer for layer in DefenseLayer if layer in selected)


def _defense_config(layers: Sequence[DefenseLayer]) -> DefenseConfig:
    """The layer set as the defense module's own config object.

    Written out field by field rather than by ``**kwargs`` so that adding a layer
    to :class:`DefenseConfig` cannot silently become a CLI flag, and so that a
    layer this runner does not expose (L4) stays at its default.
    """
    return DefenseConfig(
        spotlight=DefenseLayer.SPOTLIGHT in layers,
        detect=DefenseLayer.DETECT in layers,
        gate=DefenseLayer.GATE in layers,
    )


# -- pipeline naming: the cache key, so it has to be honest and path-safe -------

# The literal token "local" in this prefix is load-bearing and must never be
# dropped or spelled differently; see _pipeline_name.
_PIPELINE_NAME_PREFIX = "aegis-baseline-local"
# Stands in for "no reasoning effort was in play", so the name never has two
# adjacent separators that could make two different configs collide.
_NO_EFFORT_TOKEN = "noeffort"
# Stands in for "the defense was built with every layer off" - the all-off control
# arm, which is a different pipeline from the undefended baseline (the Aegis
# elements are in the loop, doing L1 bookkeeping) and so needs its own cache entry.
_NO_LAYERS_TOKEN = "nolayers"
# 8 hex chars = 32 bits. This is a cache-busting discriminator, not a security
# boundary, so collision resistance at that width is ample and the name stays
# readable as a directory.
_FINGERPRINT_CHARS = 8
# Everything a POSIX or Windows path is happy with; anything else (notably the "/"
# in a model id like "openai/gpt-oss-120b") becomes an underscore.
_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def _numeric_suffix(task_id: str) -> tuple[int, str]:
    """Sort key placing ``*_0, *_1, *_2, ...`` in numeric (not lexical) order.

    Falls back to the raw id (with a sentinel index) when there is no trailing
    integer, so an unexpected id still sorts deterministically instead of raising.
    """
    match = _TRAILING_INT.search(task_id)
    return (int(match.group(1)) if match else -1, task_id)


def _ordered_ids(ids: Iterable[str]) -> list[str]:
    return sorted((str(i) for i in ids), key=_numeric_suffix)


def select_task_ids(
    suite: Any,
    max_tasks: int,
    max_injection_tasks: int = DEFAULT_MAX_INJECTION_TASKS,
) -> tuple[list[str], list[str]]:
    """Pick the first ``max_tasks`` user tasks and first ``max_injection_tasks``
    injection tasks, both in numeric order.

    The two caps are INDEPENDENT: the injection list is bounded only by
    ``max_injection_tasks``, never additionally clamped to ``max_tasks``. Naming a
    cap is the operator's consent to the resulting spend - which is (tasks x
    injections) + injections task runs, because AgentDojo also runs each injection
    task once standalone as a user task before the injected couples - and a small
    default (:data:`DEFAULT_MAX_INJECTION_TASKS`) is what keeps an un-flagged run
    cheap. Both caps must be >= 1 - a zero cap would run nothing and silently report
    ``None`` metrics, which is worse than refusing.
    """
    if max_tasks < 1:
        raise ValueError(f"--max-tasks must be >= 1 to run anything, got {max_tasks}")
    if max_injection_tasks < 1:
        raise ValueError(
            f"--max-injection-tasks must be >= 1 to measure ASR, got {max_injection_tasks}"
        )
    user_ids = _ordered_ids(suite.user_tasks.keys())[:max_tasks]
    injection_ids = _ordered_ids(suite.injection_tasks.keys())[:max_injection_tasks]
    return user_ids, injection_ids


def _mean(values: Iterable[object]) -> float | None:
    """Mean of a bag of booleans, or ``None`` when the bag is empty.

    A ``None`` metric (nothing ran) is reported honestly rather than as ``0.0``,
    which would read as "ran and scored zero".
    """
    bools = [bool(v) for v in values]
    if not bools:
        return None
    return sum(bools) / len(bools)


def _stringify_keys(results: Any) -> dict[str, bool]:
    """Make a results dict JSON-safe by flattening tuple keys to strings.

    AgentDojo keys utility/security by ``(user_task_id, injection_task_id)``
    tuples, which ``json`` cannot serialise as object keys; join them with ``::``.
    """
    out: dict[str, bool] = {}
    for key, value in dict(results).items():
        if isinstance(key, tuple):
            out["::".join(str(part) for part in key)] = bool(value)
        else:
            out[str(key)] = bool(value)
    return out


def _suite_tool_names(suite: Any) -> tuple[str, ...]:
    """The live suite's tool names, for the L3 detector.

    Tolerant of a suite that does not expose them: the detector's tool list only
    decides whether one advisory pattern can fire, so an empty list costs a flag,
    while raising here would cost the run.
    """
    tools = getattr(suite, "tools", ()) or ()
    names = (getattr(tool, "name", None) for tool in tools)
    return tuple(name for name in names if isinstance(name, str) and name)


def _ledger_summary(aegis: AegisPipeline) -> dict[str, Any]:
    """Aggregate the gate's ledger into something a result file can carry.

    This is the field that separates "the gate refused the call" from "the model
    happened not to take the bait". AgentDojo's ``security_results`` is one boolean
    per couple and cannot tell those apart, so an ASR of 0 with an empty ledger is
    an anecdote about the model, not a measurement of the defense.

    Counts are kept per TOOL and by both axes the ledger records, because the two
    genuinely differ: the gate's ``verdict`` is what the policy said, ``action`` is
    what the executor did about it (a CONFIRM is refused in an unattended run; a
    taint-only DENY on a read-only tool is not). Reporting only one of them would
    make one of those two decisions invisible.

    The three failure counters ride along for the reason both elements count them:
    a layer that silently did not run must not be written up as one that did.
    """
    executor = aegis.executor
    by_tool: dict[str, dict[str, int]] = {}
    for entry in executor.ledger:
        counts = by_tool.setdefault(
            entry.tool_name,
            {verdict.value: 0 for verdict in GateVerdict}
            | {action.value: 0 for action in GateAction},
        )
        counts[entry.verdict.value] += 1
        counts[entry.action.value] += 1
    refusals = len(executor.refusals)
    return {
        "decisions": len(executor.ledger),
        "refusals": refusals,
        "by_tool": {name: by_tool[name] for name in sorted(by_tool)},
        "refused_tools": sorted({entry.tool_name for entry in executor.refusals}),
        # Ungated turns, unguarded turns, and tool results L4 could not judge.
        "gate_failures": int(executor.failures),
        "guard_failures": int(aegis.guard.failures),
        "quarantine_failures": int(aegis.guard.quarantine_failures),
    }


def _config_fingerprint(*parts: object) -> str:
    """A short, stable digest of the settings that decide what a run measures.

    Joined with a separator that cannot occur in any of the parts, so
    ``("ab", "c")`` and ``("a", "bc")`` cannot fingerprint alike. Digesting rather
    than concatenating keeps a URL out of a directory name and keeps the name a
    sane length; ``None`` is distinct from the empty string, because "not
    applicable to this provider" is a different configuration from "empty".
    """
    payload = "\x1f".join("\x00None" if part is None else str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:_FINGERPRINT_CHARS]


def _sanitize_name(name: str) -> str:
    """Make a pipeline name safe to use as a directory component.

    AgentDojo turns ``pipeline.name`` straight into a path segment, and a model id
    like ``openai/gpt-oss-120b`` would otherwise split it across two directory
    levels. Worse, its writer replaces ``/`` with ``_`` while its READER does not,
    so an unsanitized name writes to one path and looks for its cache at another.
    Sanitizing here makes both halves agree - which is exactly why ``force_rerun``
    must default to True (see the module docstring): the cache now actually hits.
    """
    return _UNSAFE_NAME_CHARS.sub("_", name)


def _pipeline_name(
    *,
    provider: Provider,
    model: str,
    base_url: str,
    timeout: float,
    reasoning_effort: str | None,
    benchmark_version: str,
    defense: Defense,
    defense_layers: Sequence[DefenseLayer],
) -> str:
    """Build the pipeline name: an attack-compatible identity plus a config digest.

    Two jobs, both load-bearing.

    (1) MODEL IDENTITY FOR THE ATTACK. The ``important_instructions`` attack derives
    a prose model name by substring-matching this name against AgentDojo's
    ``MODEL_NAMES`` table, and RAISES when nothing matches. Our open models (gpt-oss
    via Groq, llama via NVIDIA, gemini-3.5) are not in that table, so the name
    carries the ``local`` marker, which AgentDojo maps to "Local model" - the honest
    generic identity for a self-hosted-style model. A recognized model id present in
    the name still wins (it is matched earlier). Keeping the literal token ``local``
    intact is therefore a hard requirement, not a stylistic one.

    (2) AN HONEST CACHE KEY. This name is the ONLY discriminator in AgentDojo's
    result-cache path, so anything that can change a score but is missing from the
    name is a way for one run to be served another run's numbers. The digest
    therefore covers the endpoint, the reasoning effort, the request timeout (a
    short one kills a turn and fails a task) and the benchmark version. Values that
    do not apply to the selected provider are fingerprinted as ``None`` - exactly
    the way the result JSON records them - so a gemini run is not fingerprinted
    against an openai-compat endpoint it never used.

    (3) THE DEFENSE IDENTITY. A defended run and the undefended baseline differ in
    nothing AgentDojo keys on - same suite, same task ids, same model - so without
    the defense in this name the defended run would be served the baseline's cached
    results and report them as its own. That is the single most damaging cache hit
    available here, because the numbers it fabricates are exactly the numbers the
    project exists to produce. Both the NAME and the exact LAYER SET are in the
    digest: two arms of the ablation are two different measurements.

    The undefended name is byte-for-byte what it was before the defense seam was
    filled: the defense parts are APPENDED only when a defense is actually on, so
    an interrupted baseline can still be resumed against its own logs and the
    recorded baseline artifact stays reproducible. Any defended run digests a
    strictly longer payload, so it cannot collide with it.

    Deliberately NOT in the digest: ``min_request_interval_s``, which changes only
    the pacing of identical requests, and the task caps, which AgentDojo already
    keys on per task id.
    """
    is_openai_compat = provider is Provider.OPENAI_COMPAT
    effective_base_url = base_url if is_openai_compat else None
    effective_timeout = timeout if is_openai_compat else None
    effort_token = reasoning_effort if reasoning_effort is not None else _NO_EFFORT_TOKEN
    parts: list[object] = [
        str(provider),
        effective_base_url,
        effective_timeout,
        reasoning_effort,
        benchmark_version,
    ]
    tokens = [_PIPELINE_NAME_PREFIX, model, effort_token]
    if defense is not Defense.NONE:
        layer_token = "+".join(layer.value for layer in defense_layers) or _NO_LAYERS_TOKEN
        parts.extend([str(defense), layer_token])
        tokens.extend([str(defense), layer_token])
    tokens.append(_config_fingerprint(*parts))
    return _sanitize_name("-".join(tokens))


# -- seams: wrapped so tests can substitute them without a network or a key -----


def _load_suite(benchmark_version: str, suite_name: str) -> Any:
    return get_suite(benchmark_version, suite_name)


def _build_element(
    *,
    provider: Provider,
    model: str,
    base_url: str,
    api_key_env: str,
    timeout: float,
    reasoning_effort: str | None = None,
    min_request_interval_s: float = DEFAULT_MIN_REQUEST_INTERVAL_S,
) -> Any:
    """Dispatch to the selected provider's builder. That is all this does.

    Each provider module owns reading its own key from the NAMED variable and
    failing, when it is missing, with a message that names that variable and never a
    value - a custom ``--api-key-env`` must never silently fall back to a different
    variable's key, because then the run is not the run the operator asked for.
    Keeping that logic (and its error text) in ONE place per provider is why this
    function is a two-way dispatch and not a pair of inline implementations: two
    copies of a security-relevant message drift apart.

    ``timeout``, ``reasoning_effort`` and ``min_request_interval_s`` are
    openai-compat only (the genai element has no equivalent knobs); the CLI refuses
    them for gemini rather than accepting and ignoring them. Either branch returns
    an element carrying the ``call_count`` / ``request_count`` the runner reports.
    """
    if provider is Provider.OPENAI_COMPAT:
        return build_openai_compat_llm(
            model=model,
            base_url=base_url,
            api_key_env=api_key_env,
            timeout=timeout,
            reasoning_effort=reasoning_effort,
            min_request_interval_s=min_request_interval_s,
        )
    return build_gemini_llm(model=model, api_key_env=api_key_env)


class BuiltPipeline(NamedTuple):
    """What :func:`_build_pipeline` hands back.

    The element rides along so the caller can read its counters after the run, and
    the defense rides along so the result JSON can state what was actually built
    instead of repeating a literal ``None`` that someone would have to remember to
    update when the defense seam is filled.
    """

    pipeline: Any
    element: Any
    defense: str | None
    aegis: AegisPipeline | None = None
    """The defended pipeline's elements, or ``None`` for the undefended baseline.

    Carried for the same reason ``element`` is: the gate ledger lives on an element
    and is only worth anything read AFTER the run. Defaulted so an undefended build
    - and every existing caller of it - stays a three-field construction.
    """


def _build_pipeline(
    *,
    provider: Provider,
    model: str,
    base_url: str,
    api_key_env: str,
    timeout: float,
    benchmark_version: str = DEFAULT_BENCHMARK_VERSION,
    reasoning_effort: str | None = None,
    min_request_interval_s: float = DEFAULT_MIN_REQUEST_INTERVAL_S,
    defense: Defense = DEFAULT_DEFENSE,
    defense_layers: Sequence[DefenseLayer] = (),
    tool_names: Sequence[str] = (),
) -> BuiltPipeline:
    """Build the pipeline, the element, and the defense it was actually built with.

    ``--defense none`` takes AgentDojo's own ``defense=None`` branch through
    ``from_config``; ``--defense aegis`` takes :func:`build_aegis_pipeline`, which
    reproduces that branch element for element and differs only inside the tool
    loop. Either way ``from_config`` accepts an arbitrary ``model_id`` string, so
    the same wiring serves any provider/model pair - the defended path is not
    provider-specific and runs on openai-compat (the measured config) unchanged.

    The defense recorded in the result is READ BACK off what was built - the
    ``PipelineConfig`` for the baseline, ``DefenseConfig.label`` for the defended
    arm - never restated here, so the JSON cannot claim an arm that was not built.

    The explicit ``pipeline.name`` is not cosmetic: without one AgentDojo warns
    "pipeline_name not known" and skips logging, the ``important_instructions``
    attack needs a model identity out of it, and it is the whole of AgentDojo's
    result-cache key. It is assigned AFTER ``build_aegis_pipeline``, overwriting
    the label-based name that function sets: that label carries no ``local`` token,
    and the attack raises on a name it cannot match. See :func:`_pipeline_name`.
    """
    element = _build_element(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key_env=api_key_env,
        timeout=timeout,
        reasoning_effort=reasoning_effort,
        min_request_interval_s=min_request_interval_s,
    )
    config = PipelineConfig(
        llm=element,
        model_id=model,
        defense=None,
        system_message_name=None,
        system_message=None,
    )
    aegis: AegisPipeline | None = None
    if defense is Defense.AEGIS:
        aegis = build_aegis_pipeline(
            element,
            config,
            _defense_config(defense_layers),
            # The LIVE suite's tool names, so L3's tool_invocation_attempt pattern
            # has something to match; falling back to the policy's own list when a
            # caller has none.
            tool_names=tuple(tool_names) if tool_names else None,
        )
        pipeline = aegis.pipeline
        defense_label: str | None = aegis.defense
    else:
        pipeline = AgentPipeline.from_config(config)
        defense_label = config.defense
    pipeline.name = _pipeline_name(
        provider=provider,
        model=model,
        base_url=base_url,
        timeout=timeout,
        reasoning_effort=reasoning_effort,
        benchmark_version=benchmark_version,
        defense=defense,
        defense_layers=defense_layers,
    )
    return BuiltPipeline(pipeline=pipeline, element=element, defense=defense_label, aegis=aegis)


def _load_attack(attack_name: str, suite: Any, pipeline: Any) -> Any:
    return load_attack(attack_name, suite, pipeline)


def run_baseline(
    *,
    provider: str = DEFAULT_PROVIDER,
    suite_name: str = DEFAULT_SUITE,
    model: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
    api_key_env: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    reasoning_effort: str | None = DEFAULT_REASONING_EFFORT,
    min_request_interval_s: float = DEFAULT_MIN_REQUEST_INTERVAL_S,
    attack_name: str = DEFAULT_ATTACK,
    defense: str = DEFAULT_DEFENSE,
    defense_layers: Sequence[str] | None = None,
    max_tasks: int = DEFAULT_MAX_TASKS,
    max_injection_tasks: int = DEFAULT_MAX_INJECTION_TASKS,
    benchmark_version: str = DEFAULT_BENCHMARK_VERSION,
    resume: bool = False,
    logdir: Path = DEFAULT_LOGDIR,
    out_path: Path = DEFAULT_OUT,
) -> dict[str, Any]:
    """Run the baseline and write the result JSON. Returns the same dict written.

    ``model`` and ``api_key_env`` default to ``None`` and are resolved to the
    selected provider's sensible default; passing either overrides it. ``timeout``,
    ``reasoning_effort`` and ``min_request_interval_s`` apply to the openai-compat
    path only, and are recorded as ``None`` for gemini so the JSON never claims a
    setting that had no effect.

    ``resume=False`` (the default) forces every task to actually run.
    ``resume=True`` lets AgentDojo replay any task it already has a log for, which
    is what an interrupted long run wants - and nothing else does; see the module
    docstring for why that is not the default.

    ``defense`` selects the pipeline (``none`` - the undefended default - or
    ``aegis``) and ``defense_layers`` names the Aegis layers to enable; ``None``
    means "every layer this runner exposes" for the Aegis arm, and it must be empty
    for the undefended one. Both the name and the resolved layer set are recorded,
    because two runs are comparable only if a reader can see what was on.

    Runs two phases inside a single ``OutputLogger`` context (AgentDojo needs a real
    logger on the stack or it raises from its NullLogger): a clean utility phase and
    an injected ASR phase, both restricted to the capped task ids.

    Both caps are checked BEFORE anything is built, so a bad cap costs no requests.
    """
    if max_tasks < 1:
        raise ValueError(f"--max-tasks must be >= 1 to run anything, got {max_tasks}")
    if max_injection_tasks < 1:
        raise ValueError(
            f"--max-injection-tasks must be >= 1 to measure ASR, got {max_injection_tasks}"
        )

    provider_enum = Provider(provider)
    defense_enum = Defense(defense)
    resolved_layers = _resolve_defense_layers(defense_enum, defense_layers)
    resolved_model = model if model is not None else _PROVIDER_DEFAULT_MODEL[provider_enum]
    resolved_key_env = (
        api_key_env if api_key_env is not None else _PROVIDER_DEFAULT_KEY_ENV[provider_enum]
    )
    is_openai_compat = provider_enum is Provider.OPENAI_COMPAT
    # The base URL, the reasoning effort, the request timeout and the throttle are
    # only meaningful for the OpenAI-compatible path; genai carries its own endpoint
    # and none of those knobs, so we record None rather than a misleading value for
    # gemini. The timeout in particular CAN change a score - a short one kills a
    # turn and fails a task - so a run that had one must say so.
    recorded_base_url = base_url if is_openai_compat else None
    recorded_reasoning_effort = reasoning_effort if is_openai_compat else None
    recorded_timeout = timeout if is_openai_compat else None
    recorded_min_request_interval = min_request_interval_s if is_openai_compat else None
    force_rerun = not resume

    logdir.mkdir(parents=True, exist_ok=True)

    suite = _load_suite(benchmark_version, suite_name)
    built = _build_pipeline(
        provider=provider_enum,
        model=resolved_model,
        base_url=base_url,
        api_key_env=resolved_key_env,
        timeout=timeout,
        benchmark_version=benchmark_version,
        reasoning_effort=recorded_reasoning_effort,
        min_request_interval_s=min_request_interval_s,
        defense=defense_enum,
        defense_layers=resolved_layers,
        tool_names=_suite_tool_names(suite),
    )
    pipeline, element = built.pipeline, built.element
    attack = _load_attack(attack_name, suite, pipeline)
    user_ids, injection_ids = select_task_ids(suite, max_tasks, max_injection_tasks)

    with OutputLogger(str(logdir)):
        clean = benchmark_suite_without_injections(
            pipeline,
            suite,
            logdir,
            force_rerun,
            user_tasks=user_ids,
            benchmark_version=benchmark_version,
        )
        injected = benchmark_suite_with_injections(
            pipeline,
            suite,
            attack,
            logdir,
            force_rerun,
            user_tasks=user_ids,
            injection_tasks=injection_ids,
            verbose=False,
            benchmark_version=benchmark_version,
        )

    utility = _mean(clean["utility_results"].values())
    asr = _mean(injected["security_results"].values())
    utility_under_attack = _mean(injected["utility_results"].values())
    total_model_calls = int(element.request_count)
    # Zero traffic but non-null metrics can only mean AgentDojo served every task
    # from its on-disk cache. The numbers are then a PREVIOUS run's, sitting next to
    # THIS run's recorded settings, which is the most misleading shape a result file
    # can take - so it is flagged in the file and shouted about on the console.
    replayed = total_model_calls == 0 and any(
        metric is not None for metric in (utility, asr, utility_under_attack)
    )

    result: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "provider": str(provider_enum),
        "base_url": recorded_base_url,
        "model": resolved_model,
        # The NAME of the variable the key came from - never the key itself. A
        # result that cannot say which credential produced it cannot be audited.
        "api_key_env": resolved_key_env,
        "reasoning_effort": recorded_reasoning_effort,
        "timeout": recorded_timeout,
        "min_request_interval_s": recorded_min_request_interval,
        "suite": suite_name,
        "attack": attack_name,
        "benchmark_version": benchmark_version,
        # Three fields, because they answer three different questions: which
        # pipeline was built, what it was built with, and what it called itself.
        "defense": built.defense,
        "defense_name": str(defense_enum),
        "defense_layers": [str(layer) for layer in resolved_layers],
        # The evidence AgentDojo's own booleans cannot carry; null when no gate ran.
        "gate_ledger": _ledger_summary(built.aegis) if built.aegis is not None else None,
        "max_tasks": max_tasks,
        "max_injection_tasks": max_injection_tasks,
        "user_task_ids": user_ids,
        "injection_task_ids": injection_ids,
        "n_user_tasks": len(user_ids),
        "n_injection_tasks": len(injection_ids),
        "force_rerun": force_rerun,
        "replayed": replayed,
        # Attempts (retries included) and turns (agent decisions). Two units, two
        # names, identical meanings on both providers - see the module docstring.
        "total_model_calls": total_model_calls,
        "total_model_turns": int(element.call_count),
        "utility": utility,
        "asr": asr,
        "utility_under_attack": utility_under_attack,
        "raw": {
            "clean": {
                "utility_results": _stringify_keys(clean["utility_results"]),
                "security_results": _stringify_keys(clean["security_results"]),
            },
            "injected": {
                "utility_results": _stringify_keys(injected["utility_results"]),
                "security_results": _stringify_keys(injected["security_results"]),
                "injection_tasks_utility_results": _stringify_keys(
                    injected["injection_tasks_utility_results"]
                ),
            },
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


# The console's version of the ``replayed`` flag. Deliberately unmissable: the
# failure mode it guards against is an operator reading a table of numbers that
# were not measured by the run they just started.
_REPLAY_BANNER = [
    "!" * 52,
    " !! REPLAYED FROM CACHE - THESE ARE NOT NEW MEASUREMENTS !!",
    " Zero model requests were made, so every number above was",
    " read from a previous run's logs in the logdir and is being",
    " reported beside THIS run's settings. Re-run without",
    " --resume (or point --logdir somewhere empty) to measure.",
    "!" * 52,
]


def _format_summary(result: dict[str, Any]) -> str:
    """A short, plain-text summary table (ASCII only) for the console."""

    def pct(value: float | None) -> str:
        return "  n/a" if value is None else f"{value * 100:5.1f}%"

    def shown(value: object) -> str:
        return "n/a" if value is None else str(value)

    # A defended run is not "the baseline", and a table headed as one is how a
    # defended number gets quoted as an undefended one. The undefended heading is
    # left exactly as it was, so the recorded baseline's console output is unchanged.
    title = (
        "Aegis Week-0 baseline"
        if result["defense_name"] == Defense.NONE.value
        else f"Aegis DEFENDED run  [{shown(result['defense'])}]"
    )
    lines = [
        "=" * 52,
        f" {title}  ({result['timestamp']})",
        "=" * 52,
        f" provider           : {result['provider']}",
        f" model              : {result['model']}",
        f" endpoint           : {shown(result['base_url'])}",
        f" key from env var   : {result['api_key_env']}",
        f" reasoning effort   : {shown(result['reasoning_effort'])}",
        f" request timeout    : {shown(result['timeout'])}",
        f" min req interval   : {shown(result['min_request_interval_s'])}",
        f" suite / version    : {result['suite']} / {result['benchmark_version']}",
        f" attack             : {result['attack']}   defense: {shown(result['defense'])}",
        f" defense / layers   : {result['defense_name']}  {result['defense_layers']}",
        f" user tasks         : {result['n_user_tasks']}  {result['user_task_ids']}",
        f" injection tasks    : {result['n_injection_tasks']}  {result['injection_task_ids']}",
        f" force rerun        : {result['force_rerun']}",
        f" model requests     : {result['total_model_calls']}  (attempts, retries included)",
        f" model turns        : {result['total_model_turns']}  (agent decisions)",
        "-" * 52,
        f" utility            : {pct(result['utility'])}",
        f" attack success rate: {pct(result['asr'])}",
        f" utility under atk  : {pct(result['utility_under_attack'])}",
        "=" * 52,
    ]
    ledger = result.get("gate_ledger")
    if ledger is not None:
        # Printed beside the metrics because "ASR fell" and "the gate refused
        # something" are two different claims, and only the pair is a defense.
        lines.extend(
            [
                f" gate decisions     : {ledger['decisions']}  refused: {ledger['refusals']}",
                f" refused tools      : {ledger['refused_tools']}",
                f" defense failures   : gate {ledger['gate_failures']}"
                f"  guard {ledger['guard_failures']}"
                f"  quarantine {ledger['quarantine_failures']}",
                "=" * 52,
            ]
        )
    if result["replayed"]:
        lines.extend(_REPLAY_BANNER)
    return "\n".join(lines)


_EPILOG = f"""\
defaults are the recorded Week-0 baseline's configuration: {DEFAULT_OPENAI_MODEL} on
{DEFAULT_BASE_URL} at reasoning effort '{DEFAULT_REASONING_EFFORT}' - capable enough
to solve the user tasks and measurably hijackable, which is what a security baseline
needs. The default CAPS are smaller than the recorded run's: the quoted utility 100%
/ ASR 8.3% over 12 injected couples was measured at --max-tasks
{RECORDED_BASELINE_MAX_TASKS} --max-injection-tasks {RECORDED_BASELINE_MAX_INJECTION_TASKS},
so reproducing that number means passing both flags.

every run re-runs every task; --resume opts back into AgentDojo's on-disk result
cache, which is for finishing an interrupted run and nothing else.

an un-flagged run is UNDEFENDED. the Aegis arms are asked for explicitly, and each
layer set caches under its own pipeline name, so no defended run can be served the
baseline's numbers:

  python -m evals.agentdojo.runner --defense aegis                       # all layers
  python -m evals.agentdojo.runner --defense aegis --defense-layers gate # one layer

The NVIDIA path stays supported but is NOT a valid security baseline (a small model
scores near-0% ASR out of incapability, not resistance):

  python -m evals.agentdojo.runner \\
      --base-url {NVIDIA_BASE_URL} \\
      --api-key-env {NVIDIA_KEY_ENV} \\
      --model {NVIDIA_MODEL}

keys are never passed as flags: --api-key-env names the environment VARIABLE the
key is read from (it lives in .env, which is gitignored).
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evals.agentdojo.runner",
        description=(
            "Run the AgentDojo benchmark (utility + ASR): the undefended Week-0 "
            "baseline by default, or an Aegis-defended arm with --defense aegis."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--provider",
        choices=[p.value for p in Provider],
        default=DEFAULT_PROVIDER.value,
        help=f"agent-under-test backend (default: {DEFAULT_PROVIDER.value})",
    )
    parser.add_argument(
        "--suite", default=DEFAULT_SUITE, help=f"AgentDojo suite (default: {DEFAULT_SUITE})"
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "model id (default per provider: "
            f"{DEFAULT_OPENAI_MODEL} for openai-compat, {DEFAULT_GEMINI_MODEL} for gemini)"
        ),
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"OpenAI-compatible endpoint, openai-compat only (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--api-key-env",
        default=None,
        help=(
            "NAME of the env var holding the API key (default per provider: "
            f"{DEFAULT_OPENAI_KEY_ENV} / {DEFAULT_GEMINI_KEY_ENV}); "
            "only the name is used - the value is read from the environment, never a flag"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help=(
            "per-request timeout in seconds, openai-compat ONLY - rejected for "
            f"--provider gemini (default: {DEFAULT_TIMEOUT}); it is recorded in the "
            "result because a short timeout can fail a task and change the score"
        ),
    )
    parser.add_argument(
        "--min-request-interval",
        type=float,
        default=None,
        dest="min_request_interval",
        metavar="SECONDS",
        help=(
            "minimum spacing between requests, openai-compat ONLY - rejected for "
            f"--provider gemini (default: {DEFAULT_MIN_REQUEST_INTERVAL_S}); 0 disables "
            "the free-tier rate throttle entirely, which is what a paid tier wants"
        ),
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=list(REASONING_EFFORT_CHOICES),
        default=None,
        help=(
            "reasoning effort for a reasoning model, openai-compat ONLY - rejected for "
            f"--provider gemini (default: {DEFAULT_REASONING_EFFORT}; 'none' asks the "
            "endpoint for no reasoning at all)"
        ),
    )
    parser.add_argument(
        "--attack", default=DEFAULT_ATTACK, help=f"attack name (default: {DEFAULT_ATTACK})"
    )
    parser.add_argument(
        "--defense",
        choices=[d.value for d in Defense],
        default=DEFAULT_DEFENSE.value,
        help=(
            f"defense around the agent (default: {DEFAULT_DEFENSE.value}, the undefended "
            "baseline); 'aegis' builds the Aegis pipeline and records its layer set in "
            "the result and in the cache key, so it can never be served the baseline's "
            "cached numbers"
        ),
    )
    parser.add_argument(
        "--defense-layers",
        default=None,
        metavar="LIST",
        help=(
            "comma-separated Aegis layers, --defense aegis ONLY - rejected for "
            f"--defense none (choices: {_valid_layer_names()}; default: all). L1 "
            "(provenance) is always on and not toggleable; L4 (quarantine) is not "
            "exposed because it costs one extra model call per tool result"
        ),
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=DEFAULT_MAX_TASKS,
        help=f"hard cap on user tasks per phase - spend guard (default: {DEFAULT_MAX_TASKS})",
    )
    parser.add_argument(
        "--max-injection-tasks",
        type=int,
        default=DEFAULT_MAX_INJECTION_TASKS,
        help=(
            "hard cap on injection tasks; the injected phase runs "
            "(user tasks) x (injection tasks) couples PLUS one standalone run per "
            "injection task, so raising this is explicit consent to that spend "
            f"(default: {DEFAULT_MAX_INJECTION_TASKS})"
        ),
    )
    parser.add_argument(
        "--benchmark-version",
        default=DEFAULT_BENCHMARK_VERSION,
        help=f"AgentDojo benchmark version (default: {DEFAULT_BENCHMARK_VERSION})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "reuse AgentDojo's on-disk result for any task already logged instead of "
            "re-running it - for finishing an INTERRUPTED run only. Off by default: a "
            "resumed task makes no model call, so its numbers come from the earlier run"
        ),
    )
    parser.add_argument(
        "--logdir",
        type=Path,
        default=DEFAULT_LOGDIR,
        help=f"raw per-run log dir, gitignored (default: {DEFAULT_LOGDIR})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"committed result JSON (default: {DEFAULT_OUT})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.max_tasks < 1:
        parser.error("--max-tasks must be >= 1 (refusing to run with a non-positive cap)")
    if args.max_injection_tasks < 1:
        parser.error("--max-injection-tasks must be >= 1 (an empty injection set measures no ASR)")
    # The openai-compat-only flags parse with a None sentinel rather than their real
    # defaults, so "not given" stays distinguishable from "given" - which is what
    # lets us REJECT them for gemini (whose element has none of these knobs) instead
    # of accepting flags that would quietly do nothing.
    if args.provider == Provider.GEMINI.value:
        for flag, value in (
            ("--reasoning-effort", args.reasoning_effort),
            ("--timeout", args.timeout),
            ("--min-request-interval", args.min_request_interval),
        ):
            if value is not None:
                parser.error(
                    f"{flag} applies to --provider openai-compat only; "
                    "the gemini element has no such setting"
                )
    # Same rule as the openai-compat-only flags above, for the same reason: a flag
    # that cannot do anything is a lie about the run. --defense-layers with no
    # defense would be recorded in the result JSON and describe layers that were
    # never built.
    if args.defense == Defense.NONE.value and args.defense_layers is not None:
        parser.error(
            "--defense-layers applies to --defense aegis only; the undefended "
            "pipeline has no layers to turn on"
        )
    if args.defense_layers is not None:
        try:
            layers = parse_defense_layers(args.defense_layers)
        except ValueError as exc:
            parser.error(str(exc))
    else:
        layers = _resolve_defense_layers(Defense(args.defense), None)
    reasoning_effort = (
        args.reasoning_effort if args.reasoning_effort is not None else DEFAULT_REASONING_EFFORT
    )
    timeout = args.timeout if args.timeout is not None else DEFAULT_TIMEOUT
    min_request_interval_s = (
        args.min_request_interval
        if args.min_request_interval is not None
        else DEFAULT_MIN_REQUEST_INTERVAL_S
    )

    result = run_baseline(
        provider=args.provider,
        suite_name=args.suite,
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        timeout=timeout,
        reasoning_effort=reasoning_effort,
        min_request_interval_s=min_request_interval_s,
        attack_name=args.attack,
        defense=args.defense,
        defense_layers=[layer.value for layer in layers],
        max_tasks=args.max_tasks,
        max_injection_tasks=args.max_injection_tasks,
        benchmark_version=args.benchmark_version,
        resume=args.resume,
        logdir=args.logdir,
        out_path=args.out,
    )
    print(_format_summary(result))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
