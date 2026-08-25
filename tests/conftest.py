"""Project-wide test guards. The important one: no offline test can reach a
real endpoint, ever.

WHY THIS FILE EXISTS
--------------------
Every test in this suite is supposed to be offline - the only exceptions are
marked ``costly`` and are excluded by the standard ``pytest -m "not costly"``
invocation. "Supposed to be" was the whole problem: the builders here read their
API key from the environment and, if they find one, build a real client and make
a real request. A test that was meant to fail fast in a parser guard would, if
that guard ever regressed, quietly become a live benchmark run on the developer's
own quota - and, worse, one whose results overwrite a committed artifact.

Nothing usually exports those keys (they live in ``.env``, which is not
auto-loaded), so the failure mode is invisible right up until the one moment it
is most likely: an operator who has just exported them to record a live baseline
runs the test suite.

So the environment is emptied of them for every non-costly test. This is a
backstop, not a substitute for a test wiring its own seams: a test that reaches
for a network at all is still a bug. What this guarantees is that the bug costs
an error message instead of quota.

The ``costly`` tests are deliberately exempt - they exist to spend, are opted
into with ``AEGIS_RUN_COSTLY=1``, and gate themselves on the very variables this
fixture would otherwise remove.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from aegis.config.sandbox import MODEL_PROVIDER_ENV_VARS, scan_environment

# Every variable any provider in this repo will read a key from.
#
# Derived from the sandbox guard's own list rather than retyped, because the two
# drifting apart is a silent hole rather than a visible failure: a provider added
# to the runner with a new key variable would still be recognised as model-side by
# the tripwire (so no refusal) while this list, unaware of it, would leave the real
# key in the environment for every offline test. The stripping and the allowing
# have to be answers to the same question.
#
# The names are variable NAMES and carry no secret. Sorted for a stable order.
API_KEY_ENV_VARS: tuple[str, ...] = tuple(sorted(MODEL_PROVIDER_ENV_VARS))


@pytest.fixture(autouse=True)
def _no_real_api_keys(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """Remove every provider API key from the environment for non-costly tests.

    Uses the same ``monkeypatch`` instance the test itself gets, so a test that
    wants a FAKE key can simply ``setenv`` it afterwards, and everything is undone
    at teardown either way.
    """
    if request.node.get_closest_marker("costly") is None:
        for name in API_KEY_ENV_VARS:
            monkeypatch.delenv(name, raising=False)
    yield


@pytest.fixture(autouse=True)
def _no_real_tool_credentials(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """Remove every REAL TOOL credential the startup tripwire would flag.

    WHY, given the tripwire already refuses to run beside one: because the suite
    exercises code paths that CALL the tripwire, and whether they pass would
    otherwise depend on which secrets the developer happened to export in this
    shell. A test that fails on one laptop and passes on another teaches people
    to ignore it.

    Only names the guard already flags are removed, and only their absence is
    asserted anywhere - no value is read, copied or logged here. The tripwire's
    own tests pass explicit mappings rather than the real environment, so they
    are unaffected by this, and a test that wants a fake credential present sets
    one afterwards with the same ``monkeypatch``.
    """
    if request.node.get_closest_marker("costly") is None:
        for finding in scan_environment():
            monkeypatch.delenv(finding.name, raising=False)
    yield
