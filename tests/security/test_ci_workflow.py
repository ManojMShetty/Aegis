"""The CI workflow is a security control, so it is tested like one.

WHY THIS FILE EXISTS
--------------------
``.github/workflows/ci.yml`` makes three promises that nothing else in the
repository can keep on its own:

1. **No credential reaches CI.** There is no ``secrets:`` block, and a step
   asserts - from the repository's own list of provider variables - that none of
   them is set. A ``costly`` test therefore cannot spend quota on a runner even
   if somebody deletes the ``-m "not costly"`` filter.
2. **The gates actually run.** A workflow that installs the project and then
   forgets to invoke mypy is worse than no workflow: it is a green badge over an
   unchecked tree.
3. **The install does not prune.** ``uv sync`` removes anything outside the
   lock's resolution, and a bare ``uv run`` implies a sync. Either one would
   silently uninstall agentdojo, and the eval tests would stop collecting - which
   pytest reports as *fewer tests passing*, not as a failure.

Every one of those is a single edited line away from being false, and none of
them fails loudly when broken - a pruned environment and a missing gate both look
like a green build. So they are asserted here, against the file itself.

The parsing is deliberately structural rather than a substring search over the
raw text: a check that greps for ``"not costly"`` anywhere in the file passes
happily when the phrase survives only in a comment.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

from aegis.config.sandbox import TOOL_MODE_ENV, ToolMode

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "ci.yml"
README_PATH = REPO_ROOT / "README.md"

# The workflow file may legitimately be absent from a clone. Pushing anything under
# .github/workflows/ needs the `workflow` OAuth scope, which the account that
# committed the rest of this work does not hold, so ci.yml can be present locally
# and missing on the remote until someone with that scope pushes it. These tests
# describe a file that is not part of the package, so they skip rather than fail:
# a red suite on a fresh clone would say "this project is broken" when what is
# actually true is "one file needs a wider token".
pytestmark = [
    pytest.mark.security,
    pytest.mark.skipif(
        not WORKFLOW_PATH.is_file(),
        reason=".github/workflows/ci.yml is not in this checkout (needs the `workflow` push scope)",
    ),
]

# Packages whose presence in the install line would mean CI is paying for a GPU
# wheel or a multi-gigabyte download that no test imports.
HEAVY_PACKAGES = ("torch", "sentence-transformers", "ranx", "scipy", "statsmodels", "numpy")


def _load_workflow() -> dict[str, Any]:
    """Parse the workflow. Raw text is available separately for the few text checks."""
    loaded = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _triggers(workflow: dict[str, Any]) -> dict[str, Any]:
    """Return the ``on:`` mapping.

    YAML 1.1 - which PyYAML implements - resolves the bare word ``on`` to the
    boolean ``True``, so the key of the trigger block is not the string somebody
    reading the file sees. GitHub reads it correctly; a test that looks up
    ``workflow["on"]`` and gets a KeyError is the test being wrong, not the file.
    """
    block = workflow.get("on", workflow.get(True))
    assert isinstance(block, dict), "the workflow declares no trigger mapping"
    return block


def _steps() -> list[dict[str, Any]]:
    jobs = _load_workflow()["jobs"]
    steps: list[dict[str, Any]] = []
    for job in jobs.values():
        steps.extend(job["steps"])
    return steps


def _run_commands() -> list[str]:
    return [str(step["run"]) for step in _steps() if "run" in step]


def _gate_commands() -> list[str]:
    """Run blocks other than the install line.

    The install line names ``pytest`` and ``mypy`` as PACKAGES. Searching every
    command for the word ``mypy`` would therefore pass whether or not any step
    ever invokes it - which is precisely the failure this file exists to catch.
    """
    return [command for command in _run_commands() if "uv pip install" not in command]


def _step_index(needle: str) -> int:
    """Index of the first step whose ``run`` block contains ``needle``."""
    for index, step in enumerate(_steps()):
        if needle in str(step.get("run", "")):
            return index
    pytest.fail(f"no CI step runs {needle!r}")


# --------------------------------------------------------------------------
# 1. The workflow exists and fires on the events that matter.
# --------------------------------------------------------------------------


def test_ci_workflow_exists_and_parses() -> None:
    """A workflow GitHub cannot parse is a workflow that never runs, silently."""
    assert WORKFLOW_PATH.is_file()
    assert _load_workflow()["jobs"], "the workflow declares no job"


def test_ci_runs_on_push_and_pull_request() -> None:
    """Pull request only would leave the default branch unchecked; push only, every PR."""
    triggers = _triggers(_load_workflow())
    assert "push" in triggers
    assert "pull_request" in triggers


def test_ci_sets_up_python_313() -> None:
    """The tree is developed on 3.13; a gate run on 3.11 checks a different language."""
    commands = " ".join(_run_commands())
    assert "3.13" in commands, "no step pins Python 3.13"


# --------------------------------------------------------------------------
# 2. No credential can reach the runner.
# --------------------------------------------------------------------------


def test_ci_references_no_repository_secret() -> None:
    """The strongest form of the guarantee: there is nothing to leak in the first place.

    ``secrets.GITHUB_TOKEN`` would be harmless, but allowing the pattern at all
    invites the next line, which will be a provider key.
    """
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "secrets." not in text, "the workflow reads a repository secret"


def test_ci_asserts_no_api_key_is_present() -> None:
    """A step must actively check, not merely decline to set one.

    Declining to set a key protects against this repository's own mistakes. The
    assertion protects against an organisation-wide secret, an inherited runner
    environment, and a fork that adds one.
    """
    scripts = [command for command in _run_commands() if "MODEL_PROVIDER_ENV_VARS" in command]
    assert scripts, "no step checks the environment for provider keys"
    script = scripts[0]
    assert "sys.exit(1)" in script, "the check reports but does not fail the build"


def test_the_key_check_derives_its_names_from_the_source() -> None:
    """A hand-copied list of variable names rots the moment a provider is added.

    Importing :data:`MODEL_PROVIDER_ENV_VARS` means adding a provider to the
    sandbox module extends the CI check for free.
    """
    script = next(command for command in _run_commands() if "MODEL_PROVIDER_ENV_VARS" in command)
    assert "from aegis.config.sandbox import" in script


def test_the_key_check_runs_before_the_tests() -> None:
    """Order is the whole point: a check that runs after pytest has already spent the quota."""
    assert _step_index("MODEL_PROVIDER_ENV_VARS") < _step_index('pytest -m "not costly"')


def test_ci_declares_mock_tools() -> None:
    """The tripwire's default, made explicit, so a runner cannot inherit ``real``."""
    env = _load_workflow()["env"]
    assert env[TOOL_MODE_ENV] == ToolMode.MOCK.value


# --------------------------------------------------------------------------
# 3. The four gates actually run.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gate",
    ["ruff format --check", "ruff check", "mypy", 'pytest -m "not costly"'],
)
def test_every_gate_is_invoked(gate: str) -> None:
    """Each gate the README claims is kept green must have a step that runs it."""
    assert any(gate in command for command in _gate_commands()), f"no CI step runs {gate!r}"


def test_pytest_excludes_the_costly_tests() -> None:
    """Second belt to the no-key check: even with a key, the paid tests are deselected."""
    pytest_commands = [command for command in _gate_commands() if "pytest" in command]
    assert pytest_commands
    assert any('-m "not costly"' in command for command in pytest_commands)


# --------------------------------------------------------------------------
# 4. The install neither prunes nor pulls weight no test imports.
# --------------------------------------------------------------------------


def test_ci_never_runs_uv_sync() -> None:
    """``uv sync`` prunes to the lock, deleting the packages the eval tests import.

    (An earlier version of this docstring added that a sync would try to build ranx,
    which does not build on 3.13. Neither half holds now: ranx ships a pure-Python
    wheel, and it is in no extra and no lock entry, so a sync could never reach it.
    Pruning is the whole reason, and it is enough.)
    """
    for command in _run_commands():
        assert "uv sync" not in command, f"a CI step runs uv sync: {command!r}"


def test_every_uv_run_disables_the_implicit_sync() -> None:
    """The subtle version of the same trap: a bare ``uv run`` syncs before it runs.

    That would prune the environment mid-workflow, and the symptom is a *smaller*
    passing test count rather than a red build - which nobody reads as breakage.
    """
    for command in _run_commands():
        for line in command.splitlines():
            if "uv run" in line:
                assert "--no-sync" in line, f"uv run without --no-sync: {line.strip()!r}"


def test_the_install_pulls_nothing_heavy() -> None:
    """No GPU wheel, no multi-gigabyte model, no numerical stack this project does not use.

    Nothing under ``tests/`` imports any of these; a test that needs one needs an
    ``importorskip``, not a bigger runner.
    """
    install = next(command for command in _run_commands() if "uv pip install" in command)
    for package in HEAVY_PACKAGES:
        assert package not in install, f"the CI install pulls {package!r}"


def test_the_install_covers_what_the_eval_tests_import() -> None:
    """These are imported at module scope under ``tests/evals/``.

    Without them those modules do not collect, and a suite that collects fewer
    tests still reports success - the exact failure this workflow must not have.

    The check is on COVERAGE, not on spelling. CI once hand-pinned each package on
    the install line because the `evals` extra did not declare them; now the extra
    does, and pinning them twice would be a second list to keep in step with the
    first. So a package counts as covered if the line names it OR names an extra
    that declares it, and the extra's contents are read from pyproject.toml rather
    than restated here - otherwise this test becomes the third copy of the same
    list.
    """
    install = next(command for command in _run_commands() if "uv pip install" in command)

    extras = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    optional = extras["project"]["optional-dependencies"]
    named_extras = set(re.findall(r"\[([a-z,]+)\]", install))
    covered = {
        requirement.split(">")[0].split("=")[0].split("[")[0].strip()
        for extra in named_extras
        for name in extra.split(",")
        for requirement in optional.get(name, ())
    }

    for package in ("agentdojo", "google-genai", "openai", "tenacity"):
        assert package in install or package in covered, (
            f"the CI install neither names {package!r} nor pulls it via an extra it "
            f"installs; tests/evals imports it at module scope, so collection would fail"
        )

    # langgraph fails DIFFERENTLY, and worse. tests/adapters/ guards its imports
    # with `pytest.importorskip`, so losing the framework does not break
    # collection - it silently skips twelve tests and CI stays green. That is the
    # exact shape this whole test exists to prevent, one step quieter, so it is
    # asserted here rather than left to arrive as an agentdojo transitive.
    for package in ("langgraph",):
        assert package in install or package in covered, (
            f"the CI install neither names {package!r} nor pulls it via an extra it "
            f"installs; tests/adapters would then SKIP rather than fail, and a suite "
            f"that quietly collects fewer tests still reports success"
        )


# --------------------------------------------------------------------------
# 5. A badge may only point at a workflow that exists.
# --------------------------------------------------------------------------


def test_any_readme_badge_points_at_this_workflow() -> None:
    """A badge for a renamed or deleted workflow renders red forever, and lies either way.

    The README is allowed to carry no badge at all - that is the honest state
    until the workflow has actually run once on a real push.
    """
    readme = README_PATH.read_text(encoding="utf-8")
    for line in readme.splitlines():
        if "workflows/" in line and "badge.svg" in line:
            assert "workflows/ci.yml" in line, f"badge points at a missing workflow: {line.strip()}"
