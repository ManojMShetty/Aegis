"""Everything the console computes, as pure functions over plain dictionaries.

No sockets, no HTTP, no globals. :mod:`aegis.console.server` is a thin transport
over this module, which is what lets the whole console be tested without binding
a port - the same reason the security core is testable without a model.

WHY EVERY REQUEST IS A FRESH MIDDLEWARE
---------------------------------------
The console is STATELESS: one request carries the page to guard and the call to
judge, and builds a middleware to run both. There is no session registry, no
cookie, no cross-request taint.

That is a deliberate narrowing, and it removes a whole class of defects rather
than defending against them. :class:`AegisMiddleware` documents that one instance
serves one conversation at a time, and that a missed
:meth:`~aegis.middleware.AegisMiddleware.begin_turn` reset lets one
conversation's evidence refuse another's call - "a defense that never fired being
reported as one that did". A server holding middlewares per browser session has
to get reset ordering, instance reuse and concurrent access all correct, and
every one of those bugs shows up as a WRONG VERDICT on a page whose entire
purpose is to display verdicts. A demo that silently lies about a refusal is
worse than no demo.

Guarding then deciding inside one request is also exactly the sequence the
library models - a tool returned this, now the model wants to call that - so
nothing about the mechanism is lost by refusing to keep state between requests.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from aegis.config.policy import DEFAULT_POLICY_PATH, PolicyError, SecurityPolicy
from aegis.console.scenarios import (
    DEFAULT_TOOLS,
    PAGES,
    SCENARIOS,
    Scenario,
    scenario_by_key,
)
from aegis.domain.trust import TrustTier
from aegis.middleware import (
    CONTEXT_ARG,
    AegisMiddleware,
    Decision,
    DefenseConfig,
    ToolCall,
    ToolOutput,
    appears_in,
    normalise,
)
from aegis.security.capabilities import AuthorizationContext, ToolPolicy
from aegis.security.detector import DetectionResult
from aegis.security.spotlight import (
    DEFAULT_DATAMARK,
    SpotlightStyle,
    guidance_for_style,
    looks_like_marker,
)

__all__ = [
    "MAX_DEPTH",
    "REPO_ROOT",
    "ConsoleError",
    "ablation_arms",
    "boot_payload",
    "load_policy",
    "measured_results",
    "request_for",
    "run_scenario",
    "run_turn",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
"""Only used to find committed benchmark JSON in a source checkout.

Absent from a wheel install, which is why every reader of it degrades to None
rather than raising - see :func:`measured_results`.
"""

_GUARDED_TOOL = "get_web_page"
"""The tool the console attributes a guarded page to. It is in the policy as
read-only, which is what a page fetch is."""


class ConsoleError(ValueError):
    """A request the console can explain rather than a bug it should raise on."""


# ---------------------------------------------------------------------------
# request handling
# ---------------------------------------------------------------------------


def _config_from(raw: Any) -> DefenseConfig:
    """Build a :class:`DefenseConfig` from the page's three checkboxes."""
    if raw is None:
        return DefenseConfig.all_layers()
    if not isinstance(raw, dict):
        raise ConsoleError("'layers' must be an object of layer name -> boolean")
    style_name = str(raw.get("spotlight_style", SpotlightStyle.DATAMARK.value))
    try:
        style = SpotlightStyle(style_name)
    except ValueError:
        raise ConsoleError(
            f"unknown spotlight style {style_name!r}; expected one of "
            f"{', '.join(s.value for s in SpotlightStyle)}"
        ) from None
    return DefenseConfig(
        spotlight=bool(raw.get("spotlight", True)),
        detect=bool(raw.get("detect", True)),
        gate=bool(raw.get("gate", True)),
        # L4 is never offered. It needs a model and a key, so a console that
        # let you tick it would either lie or fail - see boot_payload().
        quarantine=False,
        spotlight_style=style,
    )


def _page_text(request: dict[str, Any]) -> str | None:
    """Resolve the untrusted text this turn, or None to guard nothing."""
    if "page_text" in request and request["page_text"] is not None:
        text = request["page_text"]
        if not isinstance(text, str):
            raise ConsoleError("'page_text' must be a string")
        # An empty string means "no tool returned anything this turn", which is a
        # real and interesting case - it is how the `nopage` scenario shows the
        # same call being allowed with nothing to trace it to. Spelled out rather
        # than left as a falsy coincidence, because every other malformed field
        # here raises and a silent reinterpretation would be the odd one out.
        return text if text else None
    key = request.get("page")
    if key is None:
        return None
    if not isinstance(key, str) or key not in PAGES:
        raise ConsoleError(f"unknown page {key!r}; expected one of {', '.join(PAGES)}")
    return PAGES[key]


MAX_DEPTH = 64
"""How deeply a request may nest before it is refused as malformed.

Nothing the page sends is more than three deep. The cap exists because
``json.loads`` on a few thousand nested brackets raises ``RecursionError``, which
is not a ``JSONDecodeError``, so it escaped the transport's parse handler
entirely and aborted the response mid-flight - leaving the browser with a bare
network failure, the one outcome ``_guarded`` exists to prevent.
"""


def _reject_unusable_text(node: Any, depth: int = 0) -> None:
    """Refuse input that later layers cannot hash, echo, or encode.

    Unpaired UTF-16 surrogates are the case that matters. ``json.loads`` accepts
    an escaped one happily, and then two separate downstream paths reject it: the
    provenance hash in :func:`aegis.domain.trust.sha256_of`, and the encoder that
    writes the response. Both raise ``UnicodeEncodeError``, which reached the
    client as a 500 marked ``"bug": true`` with a traceback on the operator's
    terminal.

    That label was the actual defect. Malformed client input is a 400, and a
    console whose selling point is not overclaiming has no business calling
    somebody else's bad string a bug in itself.
    """
    if depth > MAX_DEPTH:
        raise ConsoleError(f"request nests deeper than {MAX_DEPTH} levels")
    if isinstance(node, str):
        try:
            node.encode("utf-8")
        except UnicodeEncodeError:
            raise ConsoleError(
                "request contains text that is not valid UTF-8 (an unpaired "
                "surrogate); it cannot be hashed for provenance or echoed back"
            ) from None
    elif isinstance(node, dict):
        for key, value in node.items():
            _reject_unusable_text(key, depth + 1)
            _reject_unusable_text(value, depth + 1)
    elif isinstance(node, list):
        for item in node:
            _reject_unusable_text(item, depth + 1)


def _args_from(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConsoleError("'args' must be an object of argument name -> value")
    return {str(k): v for k, v in raw.items()}


def load_policy(policy_path: Path | None) -> SecurityPolicy:
    """Load the shipped policy, or the adopter's own.

    A malformed policy raises :class:`PolicyError`, which is the loader doing
    its job - but on this path it is the operator's YAML rather than a bug in
    the console, so it is re-raised as a :class:`ConsoleError` and the page
    shows the parse error instead of a 500 with a traceback.
    """
    try:
        return SecurityPolicy.load(policy_path)
    except PolicyError as exc:
        raise ConsoleError(f"security policy did not load: {exc}") from exc


def run_turn(request: dict[str, Any], *, policy_path: Path | None = None) -> dict[str, Any]:
    """Guard a page and judge one call, returning the whole trace.

    The request is whatever the browser sent, so everything is validated here and
    a bad field raises :class:`ConsoleError` rather than reaching the middleware.

    ``policy_path`` is the adopter half of this console: point it at your own
    ``trust_tiers.yaml`` and every verdict on the page is your deployment's,
    not this repository's. It is a server-side setting rather than a request
    field on purpose - a browser that could name a path would be the
    file-read primitive :mod:`aegis.console.server` refuses to provide.
    """
    if not isinstance(request, dict):
        raise ConsoleError("request body must be a JSON object")
    _reject_unusable_text(request)

    tool = request.get("tool")
    if not isinstance(tool, str) or not tool:
        raise ConsoleError("'tool' must be a non-empty string")

    config = _config_from(request.get("layers"))
    text = _page_text(request)
    args = _args_from(request.get("args"))
    tools = _tools_from(request.get("tools"))

    # Re-read per request rather than cached at import. It is a small YAML and
    # this is a local console, so the cost is irrelevant next to what it buys:
    # editing trust_tiers.yaml and refreshing the page shows the new posture
    # immediately, which is most of the value for the adopter audience - "what
    # would MY gate do?" is a question you answer by changing the policy and
    # looking, not by restarting a server.
    policy = load_policy(policy_path)
    middleware = AegisMiddleware(
        config,
        policy=policy,
        tool_names=sorted(tools),
        authorization=AuthorizationContext(allow_all=True),
    )
    # Once, on the output side, exactly as the class documents. There is no
    # second turn in a stateless request, so there is no second call to get wrong.
    middleware.begin_turn("console", 1)

    guard_payload: dict[str, Any] | None = None
    if text is not None:
        guarded = middleware.guard(ToolOutput.of(_GUARDED_TOOL, text))
        guard_payload = _guard_payload(text, guarded.spans[0], guarded.record.detection, config)

    decision = middleware.decide([ToolCall(name=tool, args=args)], known_tools=tools)[0]
    tool_policy = policy.policy_for(tool)

    return {
        "arm": config.label,
        "layers": _layers_payload(config),
        "guard": guard_payload,
        "decision": _decision_payload(decision, config),
        "arguments": _argument_trace(args, middleware, tool_policy, decision),
        "policy": _tool_policy_payload(tool, tool_policy, tool in tools),
    }


def _verify(
    scenario: Scenario,
    decision: dict[str, Any],
    guard: dict[str, Any] | None,
    policy_path: Path | None,
) -> dict[str, Any]:
    """Did this run do what the scenario's caption says it does?

    The detector check is here because two captions turn on L3 seeing NOTHING -
    the quiet page is this project's argument for not making a pattern detector
    load-bearing. Verified only at the decision level, a future heuristic that
    happened to fire on that page would leave both captions false while every
    other assertion still passed.
    """
    mismatches: list[str] = []
    if scenario.expect_detection is not None:
        detection = (guard or {}).get("detection")
        actual = detection.get("verdict") if isinstance(detection, dict) else None
        if actual != scenario.expect_detection:
            mismatches.append(
                f"expected L3 to report {scenario.expect_detection!r}, it reported {actual!r}"
            )
    if decision["refused"] != scenario.expect_refused:
        want = "refused" if scenario.expect_refused else "executed"
        mismatches.append(f"expected the call to be {want}, it was {decision['action']}")
    if tuple(decision["codes"]) != scenario.expect_codes:
        mismatches.append(f"expected codes {list(scenario.expect_codes)}, got {decision['codes']}")
    if tuple(decision["tainted_args"]) != scenario.expect_tainted:
        mismatches.append(
            f"expected tainted arguments {list(scenario.expect_tainted)}, "
            f"got {decision['tainted_args']}"
        )
    holds = not mismatches
    if holds:
        note = "this run matches the outcome the caption claims"
    elif policy_path is not None:
        note = (
            "This caption was written against the policy this repository ships, and you "
            "are running a different one - so it is describing a deployment that is not "
            "yours. The verdict above is real; the caption is not about it."
        )
    else:
        note = (
            "The caption no longer matches what the shipped policy produces. That is a "
            "defect in this console, not in your setup - tests/console asserts exactly "
            "this and should have caught it."
        )
    return {
        "holds": holds,
        "mismatches": mismatches,
        "note": note,
        "against_default_policy": policy_path is None,
    }


def _tools_from(raw: Any) -> frozenset[str]:
    if raw is None:
        return frozenset(DEFAULT_TOOLS)
    if not isinstance(raw, list) or not all(isinstance(t, str) for t in raw):
        raise ConsoleError("'tools' must be a list of tool-name strings")
    return frozenset(str(t) for t in raw)


def run_scenario(key: str, *, policy_path: Path | None = None) -> dict[str, Any]:
    """Run one shipped scenario and return its trace plus what it claims.

    The result carries a ``verified`` block saying whether the caption still
    holds, which is not the same question as whether the test suite passes.

    ``tests/console`` checks every caption against the SHIPPED policy. The
    moment someone starts the console with ``--policy their_own.yaml``, those
    expectations describe a different deployment: a scenario captioned "the gate
    refuses this" may legitimately execute because their policy does not list
    that tool as a sink. Shipping the check into the response means the page can
    say "this caption does not hold under the policy you loaded" instead of
    quietly displaying a claim that has stopped being true - which is the exact
    failure the scenarios-as-data design exists to prevent, reappearing one flag
    later.
    """
    scenario = scenario_by_key(key)
    if scenario is None:
        raise ConsoleError(f"unknown scenario {key!r}")
    result = run_turn(request_for(scenario), policy_path=policy_path)
    result["scenario"] = _scenario_payload(scenario)
    result["verified"] = _verify(scenario, result["decision"], result["guard"], policy_path)
    return result


def request_for(scenario: Scenario) -> dict[str, Any]:
    """The console request that reproduces a shipped scenario exactly.

    Shared by the server and by the test that verifies every caption, so the
    tested path and the served path cannot diverge.
    """
    return {
        "page": scenario.page,
        "tool": scenario.tool,
        "args": dict(scenario.args),
        "tools": list(scenario.tools),
        "layers": {
            "spotlight": scenario.config.spotlight,
            "detect": scenario.config.detect,
            "gate": scenario.config.gate,
            "spotlight_style": scenario.config.spotlight_style.value,
        },
    }


# ---------------------------------------------------------------------------
# payload builders
# ---------------------------------------------------------------------------


def _layers_payload(config: DefenseConfig) -> dict[str, Any]:
    return {
        "spotlight": config.spotlight,
        "detect": config.detect,
        "gate": config.gate,
        "quarantine": config.quarantine,
        "spotlight_style": config.spotlight_style.value,
        "enabled": [layer.value for layer in config.enabled_layers],
    }


def _guard_payload(
    raw: str,
    shown: str,
    detection: DetectionResult | None,
    config: DefenseConfig,
) -> dict[str, Any]:
    """What L1-L3 did to one tool result.

    WHY ``fenced`` AND ``datamarked`` ARE NOT SUBSTRING CHECKS
    ---------------------------------------------------------
    They used to be ``"UNTRUSTED" in shown`` and ``DEFAULT_DATAMARK in shown``,
    which reports on the TEXT rather than on what L2 did - and with L2 off,
    ``shown`` IS the attacker's own text. Paste a forged fence, untick
    spotlighting, and the panel claimed a fence L2 had never built. That is not a
    contrived input: pasting a break-out attempt and toggling the layer off is
    precisely the exercise this console exists for.

    Both are now derived from the configuration that actually ran. What the
    substring check was accidentally measuring is a genuinely interesting and
    completely different fact, so it keeps its own field:
    ``contains_forged_fence`` says the untrusted text carries marker-shaped
    content - the break-out attempt L3 flags and the spotlighter neutralises. It
    is a property of the attacker's text, never a claim about ours.
    """
    detect_payload: dict[str, Any] | None = None
    if detection is not None:
        detect_payload = {
            "verdict": detection.verdict.value,
            "score": detection.score,
            "flags": list(detection.flags),
            "signals": [
                {
                    "flag": s.flag,
                    "category": s.category,
                    "severity": s.severity.value,
                    "evidence": s.evidence,
                }
                for s in detection.signals
            ],
        }
    style = config.spotlight_style
    return {
        "raw": raw,
        "shown": shown,
        "rewritten": shown != raw,
        "spotlight_style": style.value if config.spotlight else "",
        "fenced": config.spotlight and style is not SpotlightStyle.ENCODE,
        "datamarked": config.spotlight and style is SpotlightStyle.DATAMARK,
        "encoded": config.spotlight and style is SpotlightStyle.ENCODE,
        "datamark": DEFAULT_DATAMARK,
        "contains_forged_fence": looks_like_marker(raw),
        "tier": TrustTier.UNTRUSTED.label,
        "source": f"tool:{_GUARDED_TOOL}",
        "detection": detect_payload,
        "guidance": (guidance_for_style(style) if config.spotlight else ""),
    }


def _decision_payload(decision: Decision, config: DefenseConfig) -> dict[str, Any]:
    """What L5 did, keeping two different questions apart.

    ``routed_to_gate`` says the tool was in the caller's registry, so the gate
    had a say. ``gate_enabled`` says whether L5 was switched on at all. They come
    apart in the ablation arms: with ``gate=False`` the gate still receives every
    registered call and returns ALLOW unconditionally, so a single "gated: yes"
    would tell a reader watching the L5-off arm that the gate ran and permitted
    the call - which is the opposite of what that arm demonstrates.
    """
    entry = decision.entry
    return {
        "tool": entry.tool_name,
        "action": entry.action.value,
        "refused": entry.refused,
        "verdict": entry.verdict.value,
        "tier": entry.effective_tier.label,
        "codes": [code.value for code in entry.codes],
        "distinct_codes": entry.independent_block_count,
        "tainted_args": list(entry.tainted_args),
        "reason": entry.reason,
        "note": entry.note,
        "refusal_text": decision.refusal_text if entry.refused else "",
        "routed_to_gate": decision.gate is not None,
        "gate_enabled": config.gate,
        # The runtime records an ungated call at the lattice TOP, because the
        # greatest lower bound over no judged arguments is SYSTEM. Rendered next
        # to a trace row showing that same argument as T0_UNTRUSTED it reads as a
        # contradiction, so the tier is labelled as meaningless here rather than
        # silently relayed as a finding.
        "tier_is_meaningful": decision.gate is not None,
        "violations": [
            {"code": v.code.value, "detail": v.detail, "arg": v.arg_name}
            for v in (decision.gate.violations if decision.gate is not None else ())
        ],
    }


def _argument_trace(
    args: dict[str, Any],
    middleware: AegisMiddleware,
    policy: ToolPolicy | None,
    decision: Decision,
) -> list[dict[str, Any]]:
    """Per-argument provenance: WHY each value carries the tier it does.

    Re-derived with the same public helpers the runtime uses
    (:func:`appears_in`, :func:`normalise`) rather than read off a private
    method. ``tests/console`` asserts this agrees with ``entry.tainted_args`` for
    every shipped scenario, so the two cannot drift apart unnoticed.
    """
    records = middleware.state.records
    haystack = [(record, normalise(record.tainted.value)) for record in records]

    # The runtime invents CONTEXT_ARG only for a side-effecting call that supplied
    # no arguments of its own. "Is this row synthetic?" is therefore NOT "is it
    # spelled <conversation>": a caller whose tool genuinely takes an argument of
    # that name passes it in `args`, and the runtime invented nothing.
    #
    # Getting this wrong put four false claims on one row - the value marked
    # untrusted, tiered T0, matched against every record, and captioned "appears
    # verbatim in what a tool returned" - beside a gate that had judged it T3_USER
    # and executed the call. Two derivations of one fact, disagreeing on the page.
    synthetic_row = CONTEXT_ARG in decision.entry.tainted_args and CONTEXT_ARG not in args

    shown: dict[str, Any] = dict(args)
    if synthetic_row:
        shown[CONTEXT_ARG] = "(everything this turn read)"

    out: list[dict[str, Any]] = []
    for name, value in shown.items():
        is_synthetic = synthetic_row and name == CONTEXT_ARG
        if is_synthetic:
            matched = [record for record, _ in haystack]
        else:
            matched = [record for record, text in haystack if appears_in(value, text)]
        flags: list[str] = []
        for record in matched:
            for flag in record.flags:
                if flag not in flags:
                    flags.append(flag)
        tainted = bool(matched)
        out.append(
            {
                "name": name,
                "synthetic": is_synthetic,
                "value": _render(value),
                "tier": (TrustTier.UNTRUSTED if tainted else TrustTier.USER).label,
                "tainted": tainted,
                "matched_tools": [record.tool_name for record in matched],
                "flags": flags,
                "high_risk": bool(policy is not None and policy.is_high_risk(name)),
                "why": _why(tainted, is_synthetic),
            }
        )
    return out


def _why(tainted: bool, synthetic: bool) -> str:
    """The one-line reason a row carries the tier it does."""
    if synthetic:
        return (
            "invented by the runtime: this side-effecting call supplied no arguments, "
            "so it stands for everything the turn read"
        )
    if tainted:
        return "appears verbatim in what a tool returned this turn"
    return "not traceable to any tool output this turn"


def _render(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, default=str)


def _tool_policy_payload(name: str, policy: ToolPolicy | None, registered: bool) -> dict[str, Any]:
    if policy is None:
        return {
            "tool": name,
            "known_to_policy": False,
            "registered": registered,
            "note": (
                "no policy entry: the gate fails closed and denies it"
                if registered
                else "not in the runtime's registry, so the middleware leaves it ungated"
            ),
        }
    return {
        "tool": name,
        "known_to_policy": True,
        "registered": registered,
        "side_effecting": policy.side_effecting,
        "min_arg_tier": policy.min_arg_tier.label,
        "high_risk_args": sorted(policy.high_risk_args),
        "requires_confirmation": policy.requires_confirmation,
        "allowlists": {k: sorted(v) for k, v in policy.allowlists.items()},
    }


def _scenario_payload(scenario: Scenario) -> dict[str, Any]:
    return {
        "key": scenario.key,
        "title": scenario.title,
        "caption": scenario.caption,
        "teaches": scenario.teaches,
        "outcome": scenario.outcome,
        "group": scenario.group,
    }


# ---------------------------------------------------------------------------
# boot
# ---------------------------------------------------------------------------


def _load_result(name: str) -> dict[str, Any] | None:
    """One committed benchmark file, or None if it is not in this install."""
    path = REPO_ROOT / "results" / name
    if not path.is_file():
        return None
    try:
        loaded: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _mcnemar_exact_p(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p from the two discordant counts.

    Under the null each discordant pair is a fair coin, so the number landing on
    one side is ``Binomial(b + c, 1/2)`` and the p-value is twice the smaller
    tail, capped at 1.

    THIS DUPLICATES ``evals/stats/analysis.py`` ON PURPOSE, AND A TEST PINS IT
    ------------------------------------------------------------------------
    ``aegis`` may not import ``evals`` - a test asserts it, because the library
    has to stand without the harness that measures it. Rather than let that
    boundary become the excuse for hardcoding ``0.031`` inside a function whose
    docstring promises it writes nothing down, the arithmetic is re-derived here
    and ``tests/console`` asserts it agrees with the canonical implementation.
    The test may import both; this module may not.
    """
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail: float = sum(math.comb(n, k) for k in range(min(only_a, only_b) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def _paired_security(base: dict[str, Any], deft: dict[str, Any]) -> dict[str, Any] | None:
    """The 2x2 paired table over couples both arms actually measured.

    ``security_results`` is True where the ATTACK SUCCEEDED, so ``only_baseline``
    counts the hijacks the defense removed.
    """
    try:
        a = base["raw"]["injected"]["security_results"]
        b = deft["raw"]["injected"]["security_results"]
    except (KeyError, TypeError):
        return None
    if not isinstance(a, dict) or not isinstance(b, dict):
        return None
    shared = sorted(set(a) & set(b))
    if not shared:
        return None
    both = sum(1 for k in shared if a[k] and b[k])
    only_a = sum(1 for k in shared if a[k] and not b[k])
    only_b = sum(1 for k in shared if b[k] and not a[k])
    return {
        "couples": len(shared),
        "both": both,
        "only_baseline": only_a,
        "only_defended": only_b,
        "neither": len(shared) - both - only_a - only_b,
        "discordant": only_a + only_b,
        "p_value": round(_mcnemar_exact_p(only_a, only_b), 4),
        "p_floor": round(min(1.0, 2 * 0.5 ** (only_a + only_b)), 4),
    }


def measured_results() -> dict[str, Any] | None:
    """The committed AgentDojo numbers, or None when they are not on disk.

    Every figure here is read or COMPUTED from ``results/``, never written down,
    so the page cannot quote something the repository no longer supports. The
    p-value used to be the exception - a literal ``0.031`` sitting inside this
    docstring's own promise - and is now derived from the per-couple booleans in
    the two committed files. That is also what makes ``p_floor`` honest: the
    smallest p that many discordant pairs could possibly reach.

    A wheel install has no ``results/`` directory and gets None, which the page
    renders as an absent section rather than as a zero.
    """
    base = _load_result("week0_baseline_wide.json")
    deft = _load_result("week0_defended_wide.json")
    if base is None or deft is None:
        return None

    # A half-regenerated results/ would otherwise produce a headline pairing two
    # different experiments. Refuse rather than average across them.
    comparable = all(base.get(k) == deft.get(k) for k in ("suite", "attack", "model"))
    paired = _paired_security(base, deft)
    if not comparable or paired is None:
        return None

    return {
        "couples": paired["couples"],
        "suite": base.get("suite", ""),
        "attack": base.get("attack", ""),
        "model": base.get("model", ""),
        "baseline_asr": base.get("asr"),
        "defended_asr": deft.get("asr"),
        "baseline_utility": base.get("utility"),
        "defended_utility": deft.get("utility"),
        "paired": paired,
        "p_value": paired["p_value"],
        "p_note": (
            f"exact McNemar on {paired['discordant']} discordant pairs, "
            f"{paired['only_baseline']} of them hijacks the defense removed - and "
            f"{paired['p_floor']} is the smallest p that many pairs could reach, so this "
            "sits on the floor rather than comfortably below it"
        ),
        "defended_layers": list(deft.get("defense_layers") or []),
    }


_ABLATION_ARMS: tuple[tuple[str, str, str], ...] = (
    ("baseline", "-", "ablation_baseline_screen.json"),
    ("spotlight only", "L1+L2", "ablation_spotlight_screen.json"),
    ("detect only", "L1+L3", "ablation_detect_screen.json"),
    ("gate only", "L1+L5", "ablation_gate_screen.json"),
    ("all layers", "L1+L2+L3+L5", "ablation_alllayers_screen.json"),
)


def ablation_arms() -> list[dict[str, Any]]:
    """The five-arm screen, read from the committed JSON rather than transcribed.

    This used to be a hand-typed table of strings. It happened to be correct, and
    it was the only quoted figure in the console with no drift guard - in the
    repository whose entire thesis is that such a guard is the difference between
    a measurement and an anecdote.

    ``screening_only`` travels with every row because ``results/README.md`` calls
    that caveat "not optional reading": these nine couples were selected to
    contain all six baseline hijacks - they are a 3x3 block, not the hijack set, so
    six of the nine are baseline failures and an arm can introduce a hijack on the
    other three (``spotlight`` did, twice) - and no p-value computed on them is valid.
    """
    out: list[dict[str, Any]] = []
    for arm, layers, filename in _ABLATION_ARMS:
        data = _load_result(filename)
        if data is None:
            continue
        couples = int(data.get("n_user_tasks", 0)) * int(data.get("n_injection_tasks", 0))
        asr = data.get("asr")
        out.append(
            {
                "arm": arm,
                "layers": layers,
                "asr": asr,
                "hijacks": round(asr * couples) if isinstance(asr, int | float) else None,
                "couples": couples,
                "utility": data.get("utility"),
                "screening_only": bool(data.get("screening_only")),
                "replayed": bool(data.get("replayed")),
                "model_calls": data.get("total_model_calls"),
            }
        )
    return out


def boot_payload(*, policy_path: Path | None = None) -> dict[str, Any]:
    """Everything the page needs once, at load."""
    policy = load_policy(policy_path)
    return {
        "policy": {
            "path": str(policy_path or DEFAULT_POLICY_PATH),
            "is_default": policy_path is None,
            "version": policy.version,
            "default_tier": policy.default_tier.label,
            "blocking_flags": sorted(policy.blocking_flags),
            "tool_count": len(policy.tool_policies),
            "sink_count": sum(1 for p in policy.tool_policies if p.side_effecting),
            "sources": [
                {"match": rule.match, "tier": rule.tier.label, "note": rule.note}
                for rule in policy.source_rules
            ],
        },
        "tiers": [
            {
                "label": tier.label,
                "value": int(tier),
                "instruction_authority": tier.is_instruction_authority,
                "attacker_influenced": tier.is_attacker_influenced,
            }
            for tier in sorted(TrustTier, reverse=True)
        ],
        "layers": [
            {
                "id": "L1",
                "name": "Provenance and taint",
                "toggleable": False,
                "state": "always on",
                "note": "Not an arm of the ablation: there is no configuration with it off.",
            },
            {
                "id": "L2",
                "name": "Spotlighting",
                "toggleable": True,
                "state": "on",
                "note": "Fences and datamarks untrusted spans. Costs about 68% more tokens.",
            },
            {
                "id": "L3",
                "name": "Injection detection",
                "toggleable": True,
                "state": "on",
                "note": "Advisory heuristics. Raises flags; the gate is what acts on them.",
            },
            {
                "id": "L4",
                "name": "Quarantined extraction",
                "toggleable": False,
                "state": "unavailable here",
                "note": (
                    "Needs a second model and an API key, so this offline console cannot "
                    "run it. It was also off in every measured benchmark arm, including "
                    "the one labelled 'all layers'."
                ),
            },
            {
                "id": "L5",
                "name": "Capability gate",
                "toggleable": True,
                "state": "on",
                "note": (
                    "Refuses the call. The only layer the measured ablation credits. "
                    "Note that this console runs it with allow_all authorization, so of "
                    "its five rules you are watching four - a deployment also passes the "
                    "capabilities the user actually granted, and that rule has no "
                    "read-only exemption."
                ),
            },
        ],
        "pages": [
            {"key": key, "text": text, "label": _PAGE_LABELS[key]} for key, text in PAGES.items()
        ],
        "scenarios": [_scenario_payload(s) for s in SCENARIOS],
        "tools": list(DEFAULT_TOOLS),
        "spotlight_styles": [style.value for style in SpotlightStyle],
        "measured": measured_results(),
        "ablation": ablation_arms(),
    }


_PAGE_LABELS: dict[str, str] = {
    "poisoned": "Poisoned page - a loud injection",
    "quiet": "Quiet page - the same attack, nothing for a detector to catch",
    "benign": "Benign page - no attack at all",
}
