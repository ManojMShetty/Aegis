"""The lab-isolation invariants - each asserting what is ACTUALLY enforced.

WHY THIS FILE IS WRITTEN THE WAY IT IS
--------------------------------------
An earlier version of ``SECURITY.md`` cited this exact path as proof that "every
attack runs inside a Docker network with no internet egress". The file did not
exist. That is the failure this suite is built against: not a bug in a defense,
but a document making a security claim with nothing underneath it, which spends a
reader's trust on air.

So every test here is deliberately narrow, and asserts only the mechanism it
names:

* the offline suite cannot reach a provider, because ``tests/conftest.py`` empties
  the environment of provider keys;
* the startup tripwire fires on a REAL TOOL credential;
* and does NOT fire on the model provider's key, which the harness legitimately
  needs to reach the agent under test;
* the import guard refuses a real tool module;
* the compose service declares an internal network.

What is NOT asserted here matters just as much. Nothing in this file claims the
recorded benchmark runs were network-isolated. They were not: the agent under
test is a hosted model, so every recorded run made outbound requests, and an
``internal: true`` network cannot reach a hosted endpoint either. The compose
assertions below are about the file being real and coherent, not about where any
published number came from.

DOCKER IS OPTIONAL, ALWAYS
--------------------------
The compose file is a text artifact, so its structure is checked by parsing YAML
- no daemon, no image, no network access. The one test that shells out to the
Docker CLI is marked ``integration`` and skips when the CLI is absent. A missing
Docker installation must never fail this suite; a suite that fails without Docker
would simply be run less often, and an isolation test nobody runs isolates
nothing.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from tests.conftest import API_KEY_ENV_VARS

from aegis.config.sandbox import (
    ALLOW_REAL_CREDENTIALS_ENV,
    MODEL_PROVIDER_ENV_VARS,
    TOOL_MODE_ENV,
    CredentialFinding,
    FindingKind,
    SandboxViolation,
    ToolMode,
    enforce_sandbox,
    resolve_tool_mode,
    scan_environment,
)
from aegis.tools.guard import RealToolImportError, require_real_tools_enabled

pytestmark = pytest.mark.security

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_PATH = REPO_ROOT / "docker" / "docker-compose.yml"
DOCKERFILE_PATH = REPO_ROOT / "docker" / "Dockerfile"

REAL_TOOL_MODULE = "aegis.tools.real_example"

# Fake secrets, shaped like the real thing and valid nowhere. Every value here is
# invented; the prefixes are public issuer constants, not credential material.
FAKE_GITHUB = "ghp_0000000000000000000000000000000000"
FAKE_SLACK = "xoxb-0000-0000-notarealslacktoken"
FAKE_GROQ = "gsk_0000000000000000000000000000notreal"


# --------------------------------------------------------------------------
# 1. The offline suite cannot reach a provider.
# --------------------------------------------------------------------------


def test_offline_suite_has_no_provider_key_in_the_environment() -> None:
    """The conftest guarantee, asserted rather than assumed.

    This is the one isolation control that was already enforced before any of the
    rest of this file existed, and everything else here is layered on top of it.
    """
    present = [name for name in API_KEY_ENV_VARS if os.environ.get(name)]
    assert present == [], f"conftest should have stripped these for a non-costly test: {present}"


def test_provider_client_refuses_to_build_without_a_key() -> None:
    """With the key stripped, a client cannot even be constructed - so it cannot call out.

    Proves the stripping has teeth: the failure is a loud error at construction,
    not a request that silently goes out on somebody's quota.
    """
    httpx = pytest.importorskip("httpx", reason="the llm extra is optional")
    assert httpx is not None
    from aegis.llm.base import LLMError
    from aegis.llm.providers.gemini import GeminiProvider

    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        GeminiProvider(model="gemini-2.0-flash")


# --------------------------------------------------------------------------
# 2. The tripwire fires on a real TOOL credential.
# --------------------------------------------------------------------------


def test_tripwire_fires_on_tool_credential_by_name_shape() -> None:
    """A name nobody enumerated in advance is still caught - shape, not allowlist."""
    env = {TOOL_MODE_ENV: "mock", "GITHUB_TOKEN": "not-a-real-token-but-shaped-like-one"}
    with pytest.raises(SandboxViolation) as exc:
        enforce_sandbox(env)
    assert exc.value.findings == (CredentialFinding("GITHUB_TOKEN", FindingKind.NAME_SHAPE),)


def test_tripwire_fires_on_tool_credential_by_value_prefix() -> None:
    """An innocuous NAME cannot launder a credential: the value's issuer prefix fires."""
    env = {TOOL_MODE_ENV: "mock", "HELPER_SETTING": FAKE_SLACK}
    with pytest.raises(SandboxViolation) as exc:
        enforce_sandbox(env)
    assert exc.value.findings == (
        CredentialFinding("HELPER_SETTING", FindingKind.VALUE_PREFIX, "Slack"),
    )


def test_tripwire_fires_even_inside_a_model_provider_variable() -> None:
    """A GitHub token pasted into GROQ_API_KEY is a GitHub token, allowlist or not."""
    env = {TOOL_MODE_ENV: "mock", "GROQ_API_KEY": FAKE_GITHUB}
    with pytest.raises(SandboxViolation) as exc:
        enforce_sandbox(env)
    assert [finding.name for finding in exc.value.findings] == ["GROQ_API_KEY"]


def test_tripwire_is_the_default_with_no_tool_mode_set() -> None:
    """Mock is the default, so the guard protects a shell that set nothing at all."""
    with pytest.raises(SandboxViolation):
        enforce_sandbox({"AWS_SECRET_ACCESS_KEY": "notarealawssecret"})


def test_error_names_the_variable_and_never_echoes_the_value() -> None:
    """The single most important property here: the tripwire must not become the leak.

    A guard that prints what it found puts the credential into CI logs, terminal
    scrollback and bug reports - a wider exposure than the one it prevents.
    """
    env = {
        TOOL_MODE_ENV: "mock",
        "GITHUB_TOKEN": FAKE_GITHUB,
        "SLACK_BOT_TOKEN": FAKE_SLACK,
    }
    with pytest.raises(SandboxViolation) as exc:
        enforce_sandbox(env)
    message = str(exc.value)
    assert "GITHUB_TOKEN" in message
    assert "SLACK_BOT_TOKEN" in message
    for secret in (FAKE_GITHUB, FAKE_SLACK):
        assert secret not in message
        # Not even a fragment: a "helpfully redacted" tail is still key material.
        assert secret[8:] not in message
    # The finding objects cannot leak either - the value is not a field.
    for finding in exc.value.findings:
        assert FAKE_GITHUB not in repr(finding)
        assert FAKE_SLACK not in repr(finding)
        assert FAKE_GITHUB not in finding.describe()


def test_error_points_at_the_escape_hatch() -> None:
    """An undocumented bypass gets bypassed by deleting the guard instead."""
    with pytest.raises(SandboxViolation, match=ALLOW_REAL_CREDENTIALS_ENV):
        enforce_sandbox({TOOL_MODE_ENV: "mock", "GITHUB_TOKEN": FAKE_GITHUB})


# --------------------------------------------------------------------------
# 3. ...and does NOT fire on the model provider's key.
# --------------------------------------------------------------------------


def test_model_provider_key_is_allowed_under_mock_tools() -> None:
    """The distinction that makes this guard useful rather than theatre.

    The harness must reach the agent under test, which is a hosted model by
    design. A guard that refused GROQ_API_KEY would be removed within a day.
    """
    env = {TOOL_MODE_ENV: "mock", "GROQ_API_KEY": FAKE_GROQ}
    assert scan_environment(env) == ()
    assert enforce_sandbox(env) is ToolMode.MOCK


@pytest.mark.parametrize("name", sorted(MODEL_PROVIDER_ENV_VARS))
def test_every_model_provider_variable_is_allowed(name: str) -> None:
    """Each allowlisted name individually, so one bad entry cannot hide in the set."""
    assert scan_environment({TOOL_MODE_ENV: "mock", name: "some-model-key-value"}) == ()


def test_runs_own_key_variable_can_be_declared() -> None:
    """The runner's ``--api-key-env`` seam: a run may name its own model variable."""
    env = {TOOL_MODE_ENV: "mock", "MY_LOCAL_LLM_API_KEY": "value"}
    assert scan_environment(env) != ()
    assert (
        scan_environment(env, extra_model_provider_vars=frozenset({"MY_LOCAL_LLM_API_KEY"})) == ()
    )


def test_declaring_a_model_variable_cannot_smuggle_a_tool_credential() -> None:
    """Widening the model allowlist must not widen it to actual tool credentials."""
    env = {TOOL_MODE_ENV: "mock", "MY_LOCAL_LLM_API_KEY": FAKE_GITHUB}
    findings = scan_environment(env, extra_model_provider_vars=frozenset({"MY_LOCAL_LLM_API_KEY"}))
    assert [finding.issuer for finding in findings] == ["GitHub"]


def test_ordinary_environment_variables_do_not_trip_the_guard() -> None:
    """False positives are not free: a noisy guard is a disabled guard."""
    env = {
        TOOL_MODE_ENV: "mock",
        "PATH": "/usr/bin:/bin",
        "MONKEY": "not a key despite ending in KEY",
        "AEGIS_API_KEY_ENV": "GROQ_API_KEY",  # names a variable, holds no secret
        "EMPTY_TOKEN": "   ",  # exported but empty: nothing to leak
    }
    assert scan_environment(env) == ()


# --------------------------------------------------------------------------
# 4. Modes and the escape hatch.
# --------------------------------------------------------------------------


def test_escape_hatch_lets_a_deliberate_operator_through() -> None:
    """It must work, or people route around the guard permanently."""
    env = {
        TOOL_MODE_ENV: "mock",
        ALLOW_REAL_CREDENTIALS_ENV: "true",
        "GITHUB_TOKEN": FAKE_GITHUB,
    }
    assert enforce_sandbox(env) is ToolMode.MOCK


def test_real_tools_require_the_escape_hatch_even_with_a_clean_environment() -> None:
    """A clean environment says no credential is lying around - not that side effects are OK."""
    with pytest.raises(SandboxViolation, match=ALLOW_REAL_CREDENTIALS_ENV):
        enforce_sandbox({TOOL_MODE_ENV: "real"})


def test_unknown_tool_mode_fails_closed() -> None:
    """``AEGIS_TOOLS=mocks`` must never be read as "real tools, guard off"."""
    with pytest.raises(SandboxViolation, match="not a tool mode"):
        resolve_tool_mode({TOOL_MODE_ENV: "mocks"})


def test_escape_hatch_is_not_set_by_accident() -> None:
    """Anything short of a deliberate true leaves the guard on."""
    for value in ("", "false", "no", "0", "TRUE-ish", "yes please"):
        with pytest.raises(SandboxViolation):
            enforce_sandbox(
                {
                    TOOL_MODE_ENV: "mock",
                    ALLOW_REAL_CREDENTIALS_ENV: value,
                    "GITHUB_TOKEN": FAKE_GITHUB,
                }
            )


# --------------------------------------------------------------------------
# 5. The import guard.
# --------------------------------------------------------------------------


def test_real_tool_module_refuses_to_import_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """The dangerous callable never enters the process, so nothing can reach it."""
    monkeypatch.delenv(ALLOW_REAL_CREDENTIALS_ENV, raising=False)
    monkeypatch.delitem(sys.modules, REAL_TOOL_MODULE, raising=False)
    with pytest.raises(RealToolImportError, match=ALLOW_REAL_CREDENTIALS_ENV):
        importlib.import_module(REAL_TOOL_MODULE)
    assert REAL_TOOL_MODULE not in sys.modules


def test_refused_import_is_an_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reads correctly to any caller doing ``except ImportError``."""
    monkeypatch.delenv(ALLOW_REAL_CREDENTIALS_ENV, raising=False)
    monkeypatch.delitem(sys.modules, REAL_TOOL_MODULE, raising=False)
    with pytest.raises(ImportError):
        importlib.import_module(REAL_TOOL_MODULE)


def test_real_tool_module_imports_with_the_deliberate_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """And the example stays inert even then - it is a demonstration, not a tool."""
    monkeypatch.setenv(ALLOW_REAL_CREDENTIALS_ENV, "true")
    monkeypatch.delitem(sys.modules, REAL_TOOL_MODULE, raising=False)
    module = importlib.import_module(REAL_TOOL_MODULE)
    monkeypatch.delitem(sys.modules, REAL_TOOL_MODULE, raising=False)
    with pytest.raises(NotImplementedError):
        module.send_email("nobody@example.test", "subject", "body")


def test_import_guard_reads_the_same_switch_as_the_tripwire() -> None:
    """One auditable variable governs every path from this lab to a real side effect."""
    require_real_tools_enabled("aegis.tools.pretend", env={ALLOW_REAL_CREDENTIALS_ENV: "true"})
    with pytest.raises(RealToolImportError):
        require_real_tools_enabled("aegis.tools.pretend", env={})


def test_eval_runner_refuses_to_start_beside_a_tool_credential(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A guard nobody calls is theatre - so assert the runner actually calls it.

    ``main()`` runs the tripwire after argument validation and before it builds a
    pipeline, so the refusal costs nothing: no suite is loaded, no client is
    constructed, no request is made, and no result artifact is written.
    """
    runner = pytest.importorskip("evals.agentdojo.runner", reason="the evals extra is optional")
    monkeypatch.setenv("GITHUB_TOKEN", FAKE_GITHUB)
    out_path = tmp_path / "out.json"

    exit_code = runner.main(["--logdir", str(tmp_path / "logs"), "--out", str(out_path)])

    assert exit_code == 2
    assert not out_path.exists()
    stderr = capsys.readouterr().err
    assert "GITHUB_TOKEN" in stderr
    assert FAKE_GITHUB not in stderr


# --------------------------------------------------------------------------
# 6. The compose service declares an internal network. No daemon required.
# --------------------------------------------------------------------------


def _load_compose() -> dict[str, Any]:
    text = COMPOSE_PATH.read_text(encoding="utf-8")
    loaded = yaml.safe_load(text)
    assert isinstance(loaded, dict)
    return loaded


def test_compose_and_dockerfile_exist() -> None:
    """SECURITY.md cites these paths. Citing a file that does not exist is how this started."""
    assert COMPOSE_PATH.is_file()
    assert DOCKERFILE_PATH.is_file()


def test_eval_service_network_is_internal() -> None:
    """``internal: true`` is the guarantee: Docker attaches no gateway, so there is no route out."""
    compose = _load_compose()
    networks = compose["networks"]
    assert networks, "a compose file with no declared network gets the default bridge - routable"
    for name, spec in networks.items():
        assert spec.get("internal") is True, f"network {name!r} is not internal"


def test_every_service_is_attached_only_to_internal_networks() -> None:
    """A second service on the default bridge would quietly restore egress for the whole file."""
    compose = _load_compose()
    internal = {name for name, spec in compose["networks"].items() if spec.get("internal") is True}
    for name, service in compose["services"].items():
        attached = service.get("networks")
        assert attached, f"service {name!r} declares no network, so it lands on the default bridge"
        assert set(attached) <= internal, f"service {name!r} is attached to a routable network"


def test_compose_mounts_no_credential_file() -> None:
    """`.env` holds the model key; mounting it into an attack container is the mistake."""
    compose = _load_compose()
    for name, service in compose["services"].items():
        assert "env_file" not in service, f"service {name!r} declares an env_file"
        for volume in service.get("volumes", []):
            assert ".env" not in str(volume), f"service {name!r} mounts a credential file"


def test_compose_declares_mock_tools_and_the_hatch_closed() -> None:
    """The container states its posture, so a reader need not know the defaults."""
    service = _load_compose()["services"]["eval"]
    assert service["environment"][TOOL_MODE_ENV] == ToolMode.MOCK.value
    assert service["environment"][ALLOW_REAL_CREDENTIALS_ENV] == "false"


def test_dockerfile_copies_no_credential_file() -> None:
    """Baking `.env` into an image is worse than mounting it: it ships."""
    text = DOCKERFILE_PATH.read_text(encoding="utf-8")
    copy_lines = [line for line in text.splitlines() if line.strip().upper().startswith("COPY")]
    assert copy_lines
    for line in copy_lines:
        assert ".env" not in line, f"Dockerfile copies a credential file: {line.strip()}"


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("docker") is None, reason="docker CLI not installed")
def test_compose_file_validates_against_the_docker_cli() -> None:
    """Proof the file is coherent to Docker itself, not merely valid YAML.

    Skipped, never failed, when Docker is absent - the structural assertions above
    already run everywhere, and a suite that demands a daemon gets run less often.
    """
    try:
        completed = subprocess.run(
            ["docker", "compose", "-f", str(COMPOSE_PATH), "config"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - env dependent
        pytest.skip(f"docker compose unavailable: {exc}")
    if completed.returncode != 0:  # pragma: no cover - env dependent
        stderr = completed.stderr.lower()
        if "compose" in stderr and ("not a docker command" in stderr or "unknown" in stderr):
            pytest.skip("docker CLI lacks the compose plugin")
        pytest.fail(f"docker compose config rejected the file:\n{completed.stderr}")
    rendered = yaml.safe_load(completed.stdout)
    assert rendered["networks"]["lab"]["internal"] is True


# ---------------------------------------------------------------------------
# The guard's coverage, not just its behaviour
# ---------------------------------------------------------------------------


def test_every_module_in_aegis_tools_calls_the_guard() -> None:
    """A guard nobody is obliged to call protects only the authors who remember.

    `require_real_tools_enabled` genuinely blocks the import of a module that
    invokes it - there are tests above for that. What it cannot do by itself is
    stop the NEXT real tool from simply not calling it: an unguarded module
    dropped into this package imports cleanly with the escape hatch off, and
    every other test in this file still passes, because they all exercise the
    one module that opts in.

    That gap is what makes SECURITY.md's "no real tools exist" bullet the only
    enforced claim with nothing under it. This closes it structurally: every
    module in the package must invoke the guard at import scope, so the way to
    add an unguarded tool is to delete a test that says you may not.

    Kept as source inspection rather than an import: importing each module to
    find out whether it is guarded would run the very code the guard exists to
    keep out of the process.
    """
    package = REPO_ROOT / "src" / "aegis" / "tools"
    exempt = {"__init__.py", "guard.py"}

    unguarded = [
        path.name
        for path in sorted(package.glob("*.py"))
        if path.name not in exempt
        and "require_real_tools_enabled(" not in path.read_text(encoding="utf-8")
    ]

    assert not unguarded, (
        "every module in aegis.tools must call require_real_tools_enabled() at import "
        f"scope; these do not: {unguarded}. A real tool that skips the guard is a real "
        "tool the escape hatch does not gate."
    )


def test_the_guard_coverage_check_would_catch_an_unguarded_module(tmp_path: Path) -> None:
    """The paired negative: prove the check above is not vacuously true.

    It currently passes because the package holds one module and that module is
    guarded. A check that would also pass over an unguarded module would be
    worthless, so run the same rule against a package that has one.
    """
    package = tmp_path / "tools"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "guard.py").write_text("def require_real_tools_enabled(name): ...", encoding="utf-8")
    (package / "real_example.py").write_text(
        "from aegis.tools.guard import require_real_tools_enabled\n"
        "require_real_tools_enabled(__name__)\n",
        encoding="utf-8",
    )
    (package / "real_smtp.py").write_text("import socket\n", encoding="utf-8")

    exempt = {"__init__.py", "guard.py"}
    unguarded = [
        path.name
        for path in sorted(package.glob("*.py"))
        if path.name not in exempt
        and "require_real_tools_enabled(" not in path.read_text(encoding="utf-8")
    ]

    assert unguarded == ["real_smtp.py"], (
        "the coverage rule must flag a module that does not call the guard, or it is not a rule"
    )
