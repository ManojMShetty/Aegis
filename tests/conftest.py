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

# Every variable any provider in this repo will read a key from. Adding a
# provider means adding its variable here; the names are variable NAMES and
# carry no secret.
API_KEY_ENV_VARS = (
    "GROQ_API_KEY",
    "GEMINI_API_KEY",
    "NVIDIA_API_KEY",
)


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
