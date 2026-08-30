"""An offline console over the real defense layers - the project's only UI.

    uv run python scripts/demo_ui.py

WHAT IT IS FOR
--------------
Two audiences, one page. Someone evaluating this project can watch a poisoned
page get fenced, flagged and then refused at the gate, with every tier and reason
code coming from an actual call into :mod:`aegis` rather than from a screenshot.
Someone ADOPTING the library can point it at their own ``trust_tiers.yaml`` and
see what their gate would do to their own tool calls before wiring it into an
agent - which is the cheaper half of an integration to get wrong.

Everything it renders is computed live and offline. No API key, no network, no
model: L1, L2, L3 and L5 are deterministic Python, which is the whole reason a
console like this can exist at all.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It runs no benchmark. The measured numbers it displays - attack success rate
18.8% to 0%, p = 0.031 - are read from committed JSON in ``results/`` and are
labelled as quoted rather than computed, because they came from 32 AgentDojo
couples through an adapter this package does not import. Blurring the two would
be the exact overclaim the rest of the repository works to avoid.

It also cannot run L4. The quarantine extractor needs a second model and a key,
so the layer is shown as unavailable rather than offered as a checkbox that
silently does nothing - and it is worth knowing that L4 was off in every measured
arm too, including the one the results call "all layers".

Three of the shipped scenarios end with the attacker winning. Those are the
residual holes from ``SECURITY.md`` made clickable, and a test asserts that each
one still behaves the way its caption says.

LAYOUT
------
``api.py``       every judgement, as pure functions over dicts. No sockets.
``scenarios.py`` the shipped examples as data, with their expected outcomes.
``server.py``    a thin stdlib HTTP transport. Loopback only, one static asset.
``page.html``    the page itself, self-contained.
"""

from __future__ import annotations

from aegis.console.api import (
    ConsoleError,
    boot_payload,
    measured_results,
    request_for,
    run_scenario,
    run_turn,
)
from aegis.console.scenarios import PAGES, SCENARIOS, Scenario, scenario_by_key

__all__ = [
    "PAGES",
    "SCENARIOS",
    "ConsoleError",
    "Scenario",
    "boot_payload",
    "measured_results",
    "request_for",
    "run_scenario",
    "run_turn",
    "scenario_by_key",
]
