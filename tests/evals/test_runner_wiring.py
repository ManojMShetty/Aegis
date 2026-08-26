"""Runner wiring against fake benchmarks - no network, no key, no model calls.

The runner's own logic is small but load-bearing: pick the first N tasks in
NUMERIC order, honour the two spend caps, aggregate the booleans into
utility/asr, and write a JSON-safe file. Each of those is asserted here with the
two AgentDojo ``benchmark_suite_*`` entry points and every I/O seam replaced by a
stub, so the test exercises the runner and nothing behind it.

Three groups of tests exist because three classes of mistake are expensive here:

* the CLI tests pin ``--reasoning-effort``, ``--timeout``,
  ``--min-request-interval`` and ``--max-injection-tasks`` (including that they
  reach the result JSON, so a recorded run always states the configuration that
  produced it), and pin that the openai-compat-only flags are REFUSED for gemini
  rather than accepted and ignored;
* the defaults tests pin the Groq/gpt-oss-120b configuration by literal value - a
  silent default change would make two runs incomparable while both still looked
  valid;
* the cache-honesty tests pin the machinery that keeps a REPLAY from being written
  up as a measurement: ``force_rerun`` true by default at both benchmark call
  sites, ``--resume`` as the only way back into the cache, a ``pipeline.name`` that
  fingerprints everything able to change a score, and the ``replayed`` flag that
  fires when a run reports numbers having made no requests.

Every test that calls ``main`` passes ``--logdir``/``--out`` into ``tmp_path``,
including the ones that expect the parser to refuse the arguments outright: a test
must never be one regression away from being a live benchmark run that overwrites
the committed result artifact.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from evals.agentdojo import runner
from evals.agentdojo.openai_llm import DEFAULT_MIN_REQUEST_INTERVAL_S


class _FakeLogger:
    """Stand-in for OutputLogger - just a no-op context manager."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __enter__(self) -> _FakeLogger:
        return self

    def __exit__(self, *args: Any) -> bool:
        return False


def _fake_suite() -> Any:
    # Keys deliberately scrambled so numeric ordering (not dict order) is tested.
    user_ids = ["user_task_2", "user_task_0", "user_task_10", "user_task_1", "user_task_3"]
    injection_ids = ["injection_task_1", "injection_task_0", "injection_task_2"]
    return SimpleNamespace(
        name="workspace",
        user_tasks=dict.fromkeys(user_ids, object()),
        injection_tasks=dict.fromkeys(injection_ids, object()),
    )


CLEAN_RESULTS: dict[str, Any] = {
    "utility_results": {
        ("user_task_0", ""): True,
        ("user_task_1", ""): True,
        ("user_task_2", ""): False,
        ("user_task_3", ""): False,
    },
    "security_results": {
        ("user_task_0", ""): False,
        ("user_task_1", ""): False,
        ("user_task_2", ""): False,
        ("user_task_3", ""): False,
    },
    "injection_tasks_utility_results": {},
}

INJECTED_RESULTS: dict[str, Any] = {
    "utility_results": {
        ("user_task_0", "injection_task_0"): True,
        ("user_task_0", "injection_task_1"): True,
        ("user_task_1", "injection_task_0"): False,
        ("user_task_1", "injection_task_1"): False,
    },
    "security_results": {
        ("user_task_0", "injection_task_0"): True,
        ("user_task_0", "injection_task_1"): False,
        ("user_task_1", "injection_task_0"): False,
        ("user_task_1", "injection_task_1"): False,
    },
    "injection_tasks_utility_results": {
        "injection_task_0": True,
        "injection_task_1": True,
    },
}


class _Recorder:
    """Records the args each benchmark call was made with; returns a canned result."""

    def __init__(self, return_value: dict[str, Any]) -> None:
        self.return_value = return_value
        self.calls: list[dict[str, Any]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"args": args, "kwargs": kwargs})
        return self.return_value


class _PipelineFactory:
    """Stand-in for ``_build_pipeline``: records its kwargs, returns a BuiltPipeline.

    It hands back the real :class:`runner.BuiltPipeline` triple rather than a bare
    pair, because the defense the runner records comes off that triple - the whole
    point of the seam being single. ``defense`` is settable so a test can prove the
    JSON reports what was actually built instead of a hardcoded ``None`` that
    happens to match today.
    """

    def __init__(self, pipeline: Any, element: Any) -> None:
        self.pipeline = pipeline
        self.element = element
        self.defense: str | None = None
        # The defended elements, when a test wants the ledger to reach the JSON.
        # None is the undefended build, which is what every other test expects.
        self.aegis: Any = None
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return runner.BuiltPipeline(
            pipeline=self.pipeline,
            element=self.element,
            defense=self.defense,
            aegis=self.aegis,
        )


@pytest.fixture
def wired(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Patch every seam so run_baseline exercises only the runner's own logic.

    The element carries BOTH counters with DIFFERENT values, so a test cannot pass
    by reading the wrong one: ``request_count`` (attempts, what quota is spent on)
    is what ``total_model_calls`` must report, and ``call_count`` (turns) is what
    ``total_model_turns`` must report. A non-zero attempt count also keeps an
    ordinary wired run out of the "replayed from cache" branch, which the replay
    tests enter deliberately by zeroing it.
    """
    element = SimpleNamespace(call_count=7, request_count=11)
    pipeline = SimpleNamespace(name=None)
    suite = _fake_suite()
    attack = SimpleNamespace(name="important_instructions")

    clean = _Recorder(CLEAN_RESULTS)
    injected = _Recorder(INJECTED_RESULTS)
    build_pipeline = _PipelineFactory(pipeline, element)

    monkeypatch.setattr(runner, "OutputLogger", _FakeLogger)
    monkeypatch.setattr(runner, "_load_suite", lambda *a, **k: suite)
    monkeypatch.setattr(runner, "_build_pipeline", build_pipeline)
    monkeypatch.setattr(runner, "_load_attack", lambda *a, **k: attack)
    monkeypatch.setattr(runner, "benchmark_suite_without_injections", clean)
    monkeypatch.setattr(runner, "benchmark_suite_with_injections", injected)

    return {
        "clean": clean,
        "injected": injected,
        "element": element,
        "suite": suite,
        "build_pipeline": build_pipeline,
        "built": build_pipeline.calls,
    }


def _sink_flags(tmp_path: Path) -> list[str]:
    """``--logdir``/``--out`` pointed inside ``tmp_path``.

    Passed by EVERY main() test, including those that expect the parser to refuse
    the run before it starts. If such a guard regressed, an un-flagged test would
    run the benchmark for real and write its result over the committed
    :data:`runner.DEFAULT_OUT` artifact - a file that costs real quota to
    reproduce. The flags make that regression cost a failing assertion instead.
    """
    return ["--logdir", str(tmp_path / "logs"), "--out", str(tmp_path / "out.json")]


def test_metric_aggregation(wired: dict[str, Any], tmp_path: Path) -> None:
    result = runner.run_baseline(
        max_tasks=4,
        logdir=tmp_path / "logs",
        out_path=tmp_path / "out.json",
    )
    assert result["utility"] == 0.5  # 2 of 4 clean tasks solved
    assert result["asr"] == 0.25  # 1 of 4 injected couples hijacked
    assert result["utility_under_attack"] == 0.5  # 2 of 4 still did the user's job
    # Two units, two fields, and each must read the counter it names: attempts
    # (what quota was spent on) and turns (how many decisions the agent made).
    assert result["total_model_calls"] == 11  # element.request_count - attempts
    assert result["total_model_turns"] == 7  # element.call_count - turns
    assert result["replayed"] is False  # requests were made, so nothing was replayed


def test_max_tasks_cap_and_numeric_order(wired: dict[str, Any], tmp_path: Path) -> None:
    runner.run_baseline(
        max_tasks=2,
        logdir=tmp_path / "logs",
        out_path=tmp_path / "out.json",
    )
    # Clean phase gets the first 2 user tasks in NUMERIC order (0, 1) - not the
    # scrambled dict order, and not user_task_10.
    clean_kwargs = wired["clean"].calls[0]["kwargs"]
    assert clean_kwargs["user_tasks"] == ["user_task_0", "user_task_1"]

    # Injected phase: same user cap, injection set capped to the small default.
    injected_kwargs = wired["injected"].calls[0]["kwargs"]
    assert injected_kwargs["user_tasks"] == ["user_task_0", "user_task_1"]
    assert injected_kwargs["injection_tasks"] == ["injection_task_0", "injection_task_1"]


def test_output_json_shape_is_serializable(wired: dict[str, Any], tmp_path: Path) -> None:
    out_path = tmp_path / "week0.json"
    result = runner.run_baseline(
        suite_name="workspace",
        model="custom/model-x",
        max_tasks=4,
        logdir=tmp_path / "logs",
        out_path=out_path,
    )
    assert out_path.exists()
    on_disk = json.loads(out_path.read_text(encoding="utf-8"))

    # File and return value agree, and the file round-trips through JSON.
    assert on_disk == result

    # Required top-level fields are present.
    for field in (
        "timestamp",
        "provider",
        "base_url",
        "model",
        "api_key_env",
        "reasoning_effort",
        "timeout",
        "min_request_interval_s",
        "suite",
        "attack",
        "benchmark_version",
        "defense",
        "max_tasks",
        "max_injection_tasks",
        "user_task_ids",
        "injection_task_ids",
        "n_user_tasks",
        "n_injection_tasks",
        "force_rerun",
        "replayed",
        "total_model_calls",
        "total_model_turns",
        "utility",
        "asr",
        "utility_under_attack",
        "raw",
    ):
        assert field in on_disk, field

    assert on_disk["defense"] is None
    # A result must be able to say which credential produced it - by VARIABLE NAME,
    # never by value.
    assert on_disk["api_key_env"] == runner.DEFAULT_OPENAI_KEY_ENV
    assert on_disk["model"] == "custom/model-x"
    assert on_disk["n_user_tasks"] == 4

    # Tuple keys must have been flattened to strings for JSON.
    clean_util = on_disk["raw"]["clean"]["utility_results"]
    assert "user_task_0::" in clean_util
    injected_sec = on_disk["raw"]["injected"]["security_results"]
    assert "user_task_0::injection_task_0" in injected_sec


def test_select_task_ids_numeric_and_capped() -> None:
    suite = _fake_suite()
    user_ids, injection_ids = runner.select_task_ids(suite, max_tasks=3)
    assert user_ids == ["user_task_0", "user_task_1", "user_task_2"]
    # The default injection cap, not the user cap.
    assert injection_ids == ["injection_task_0", "injection_task_1"]


def test_injection_cap_is_independent_of_max_tasks() -> None:
    """An explicit injection cap is the operator's consent and is NOT clamped.

    The old behaviour silently took min(max_tasks, cap), which made
    ``--max-tasks 1 --max-injection-tasks 3`` quietly measure ASR over one injection
    - a number that did not match what was asked for.
    """
    suite = _fake_suite()
    user_ids, injection_ids = runner.select_task_ids(suite, max_tasks=1, max_injection_tasks=3)
    assert user_ids == ["user_task_0"]
    assert injection_ids == ["injection_task_0", "injection_task_1", "injection_task_2"]


def test_non_positive_max_tasks_refused(tmp_path: Path) -> None:
    """Both doors are locked, not just the outer one.

    ``run_baseline`` checks the cap before building anything, and
    ``select_task_ids`` checks it again for callers that reach it directly - the
    sibling injection guard has always been pinned in both places, and an unpinned
    guard is one refactor away from being deleted.
    """
    with pytest.raises(ValueError, match="max-tasks"):
        runner.run_baseline(max_tasks=0, logdir=tmp_path, out_path=tmp_path / "o.json")
    with pytest.raises(ValueError, match="max-tasks"):
        runner.select_task_ids(_fake_suite(), max_tasks=0)


def test_non_positive_max_injection_tasks_refused(tmp_path: Path) -> None:
    """A zero injection cap would run the injected phase over nothing and report a
    null ASR that looks like a measurement; refuse it the same way as --max-tasks."""
    with pytest.raises(ValueError, match="max-injection-tasks"):
        runner.run_baseline(max_injection_tasks=0, logdir=tmp_path, out_path=tmp_path / "o.json")
    with pytest.raises(ValueError, match="max-injection-tasks"):
        runner.select_task_ids(_fake_suite(), max_tasks=1, max_injection_tasks=-1)


def test_main_rejects_non_positive_max_tasks(wired: dict[str, Any], tmp_path: Path) -> None:
    """The parser refuses the cap, and nothing downstream is reached.

    Wired and sunk into tmp_path on purpose: this test exists to prove a guard
    fires, so it must stay harmless in the world where the guard does NOT fire.
    """
    with pytest.raises(SystemExit):
        runner.main(["--max-tasks", "0", *_sink_flags(tmp_path)])
    assert wired["clean"].calls == []  # the guard fired before any benchmark ran


@pytest.mark.parametrize("bad", ["0", "-1"])
def test_main_rejects_non_positive_max_injection_tasks(
    bad: str, wired: dict[str, Any], tmp_path: Path
) -> None:
    with pytest.raises(SystemExit):
        runner.main(["--max-injection-tasks", bad, *_sink_flags(tmp_path)])
    assert wired["injected"].calls == []


def test_json_carries_provider_base_url_model_for_default_openai(
    wired: dict[str, Any], tmp_path: Path
) -> None:
    """The default provider records itself, its endpoint, and its default model."""
    result = runner.run_baseline(
        max_tasks=2,
        logdir=tmp_path / "logs",
        out_path=tmp_path / "out.json",
    )
    assert result["provider"] == "openai-compat"
    assert result["base_url"] == runner.DEFAULT_BASE_URL
    assert result["model"] == runner.DEFAULT_OPENAI_MODEL
    assert result["reasoning_effort"] == runner.DEFAULT_REASONING_EFFORT


def test_gemini_provider_switches_defaults_and_nulls_base_url(
    wired: dict[str, Any], tmp_path: Path
) -> None:
    """Selecting gemini switches the default model and records no base_url (genai
    carries its own endpoint - a recorded URL would be misleading). The reasoning
    effort is nulled for the same reason: the genai element has no such knob, so
    recording one would claim a setting that did nothing."""
    result = runner.run_baseline(
        provider="gemini",
        max_tasks=2,
        logdir=tmp_path / "logs",
        out_path=tmp_path / "out.json",
    )
    assert result["provider"] == "gemini"
    assert result["base_url"] is None
    assert result["model"] == runner.DEFAULT_GEMINI_MODEL
    assert result["reasoning_effort"] is None
    assert wired["built"][0]["reasoning_effort"] is None


def test_explicit_model_is_honored_for_selected_provider(
    wired: dict[str, Any], tmp_path: Path
) -> None:
    """An explicit --model overrides the provider default, whichever provider it is."""
    result = runner.run_baseline(
        provider="openai-compat",
        model="custom/model-x",
        max_tasks=1,
        logdir=tmp_path / "logs",
        out_path=tmp_path / "out.json",
    )
    assert result["model"] == "custom/model-x"


# -- the canonical defaults ----------------------------------------------------


def test_parser_defaults_are_the_recorded_groq_baseline() -> None:
    """The un-flagged run uses the recorded baseline's PROVIDER, ENDPOINT, MODEL and
    REASONING EFFORT - not its task caps.

    Asserted by literal value, not by referring to the constants, so a change to the
    baseline configuration has to be made deliberately in two places instead of
    silently making two runs incomparable.

    The default caps are deliberately SMALLER than the recorded run's (the default
    spend profile is cheap), so an un-flagged run does not reproduce the quoted
    utility 100% / ASR 8.3% over 12 injected couples; that needs
    ``--max-tasks 3 --max-injection-tasks 4``. Both numbers are pinned here so the
    prose in the module docstring and the epilog cannot drift from the constants
    they quote.
    """
    args = runner._build_parser().parse_args([])
    assert args.provider == "openai-compat"
    assert args.base_url == "https://api.groq.com/openai/v1"
    assert args.model is None  # resolved per provider
    assert args.api_key_env is None  # resolved per provider
    assert args.max_injection_tasks == 2
    # --reasoning-effort parses as a None SENTINEL so "not given" stays
    # distinguishable from "given" (that is what lets gemini reject it); the real
    # default is applied afterwards.
    assert args.reasoning_effort is None
    assert runner.DEFAULT_OPENAI_MODEL == "openai/gpt-oss-120b"
    assert runner.DEFAULT_OPENAI_KEY_ENV == "GROQ_API_KEY"
    assert runner.DEFAULT_REASONING_EFFORT == "low"
    # 3 x 4 = the 12 injected couples the quoted ASR is a fraction of - and NOT
    # what an un-flagged run does.
    assert runner.RECORDED_BASELINE_MAX_TASKS == 3
    assert runner.RECORDED_BASELINE_MAX_INJECTION_TASKS == 4
    assert (args.max_tasks, args.max_injection_tasks) != (
        runner.RECORDED_BASELINE_MAX_TASKS,
        runner.RECORDED_BASELINE_MAX_INJECTION_TASKS,
    )


def test_cli_flags_reach_the_result_json(
    wired: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run records the flags it ran under, so a result file is self-describing."""
    out_path = tmp_path / "out.json"
    exit_code = runner.main(
        [
            "--max-tasks",
            "2",
            "--max-injection-tasks",
            "3",
            "--reasoning-effort",
            "medium",
            "--logdir",
            str(tmp_path / "logs"),
            "--out",
            str(out_path),
        ]
    )
    capsys.readouterr()  # the summary table is not under test here
    assert exit_code == 0

    on_disk = json.loads(out_path.read_text(encoding="utf-8"))
    assert on_disk["reasoning_effort"] == "medium"
    assert on_disk["max_injection_tasks"] == 3
    assert on_disk["max_tasks"] == 2
    # The independent cap really did widen the injected phase past --max-tasks.
    assert on_disk["injection_task_ids"] == [
        "injection_task_0",
        "injection_task_1",
        "injection_task_2",
    ]
    assert wired["built"][0]["reasoning_effort"] == "medium"


def test_cli_default_reasoning_effort_is_applied(
    wired: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Omitting the flag runs at the baseline's effort, and says so in the JSON."""
    out_path = tmp_path / "out.json"
    runner.main(
        [
            "--max-tasks",
            "1",
            "--logdir",
            str(tmp_path / "logs"),
            "--out",
            str(out_path),
        ]
    )
    capsys.readouterr()
    on_disk = json.loads(out_path.read_text(encoding="utf-8"))
    assert on_disk["reasoning_effort"] == runner.DEFAULT_REASONING_EFFORT


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--reasoning-effort", "low"),
        ("--timeout", "5"),
        ("--min-request-interval", "0"),
    ],
)
def test_openai_only_flags_are_rejected_for_gemini(
    flag: str, value: str, wired: dict[str, Any], tmp_path: Path
) -> None:
    """The gemini element has none of these knobs, so each flag is refused rather
    than accepted and silently ignored - a flag that does nothing is a lie, and
    ``--timeout`` in particular can change a SCORE (a short one kills a turn and
    fails the task), so two runs that differed in it must not look identical."""
    with pytest.raises(SystemExit):
        runner.main(["--provider", "gemini", flag, value, *_sink_flags(tmp_path)])
    assert wired["clean"].calls == []


def test_invalid_reasoning_effort_is_rejected_by_the_parser(
    wired: dict[str, Any], tmp_path: Path
) -> None:
    """argparse choices reject a bad level before any run starts (zero requests)."""
    with pytest.raises(SystemExit):
        runner.main(["--reasoning-effort", "lowish", *_sink_flags(tmp_path)])
    assert wired["clean"].calls == []


# -- cache honesty: a replay must never be written up as a measurement ---------


def _force_rerun_args(wired: dict[str, Any]) -> tuple[Any, Any]:
    """The ``force_rerun`` argument each benchmark call actually received.

    Both AgentDojo entry points take it POSITIONALLY - 4th for the clean phase, 5th
    for the injected one (which also takes the attack) - so it is read back out of
    the recorded positional args. That is exactly where the original bug hid: an
    unnamed ``False`` in the middle of a long call, which made every re-run replay
    the previous run's numbers off disk instead of measuring anything.
    """
    return wired["clean"].calls[0]["args"][3], wired["injected"].calls[0]["args"][4]


def test_force_rerun_is_on_by_default_at_both_call_sites(
    wired: dict[str, Any], tmp_path: Path
) -> None:
    """An ordinary run measures; it never replays.

    AgentDojo's result cache is keyed only by pipeline name / suite / task, so a
    cache hit returns the OLD run's numbers and they get written beside the NEW
    run's settings. Defaulting to force_rerun is what makes the default outcome of
    running the benchmark an actual benchmark.
    """
    result = runner.run_baseline(
        max_tasks=2,
        logdir=tmp_path / "logs",
        out_path=tmp_path / "out.json",
    )
    assert _force_rerun_args(wired) == (True, True)
    # ... and the run states which mode it was in, so the file is self-describing.
    assert result["force_rerun"] is True


def test_resume_opts_back_into_the_cache(wired: dict[str, Any], tmp_path: Path) -> None:
    """``resume=True`` is the ONLY way back into the cache - for an interrupted run.

    It is recorded, because a resumed run's numbers are partly an earlier run's and
    a reader of the file has to be able to tell.
    """
    result = runner.run_baseline(
        max_tasks=2,
        resume=True,
        logdir=tmp_path / "logs",
        out_path=tmp_path / "out.json",
    )
    assert _force_rerun_args(wired) == (False, False)
    assert result["force_rerun"] is False


def test_cli_resume_flag_is_off_unless_asked_for(
    wired: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The flag exists, defaults to off, and reaches both benchmark calls."""
    assert runner._build_parser().parse_args([]).resume is False

    out_path = tmp_path / "out.json"
    runner.main(
        [
            "--max-tasks",
            "1",
            "--resume",
            "--logdir",
            str(tmp_path / "logs"),
            "--out",
            str(out_path),
        ]
    )
    capsys.readouterr()

    assert _force_rerun_args(wired) == (False, False)
    assert json.loads(out_path.read_text(encoding="utf-8"))["force_rerun"] is False


def test_zero_requests_with_real_metrics_is_flagged_as_a_replay(
    wired: dict[str, Any], tmp_path: Path
) -> None:
    """Numbers without traffic can only have come off disk - so say so, loudly.

    This is the belt-and-braces check behind force_rerun: whatever the flags
    claimed, a run reporting metrics while having made zero requests did not measure
    them. The flag goes in the file and a banner goes on the console, because the
    dangerous shape is a table of plausible numbers sitting next to settings that
    never produced them.
    """
    wired["element"].request_count = 0
    result = runner.run_baseline(
        max_tasks=2,
        resume=True,
        logdir=tmp_path / "logs",
        out_path=tmp_path / "out.json",
    )

    assert result["total_model_calls"] == 0
    assert result["utility"] is not None  # metrics ARE present - that is the trap
    assert result["replayed"] is True

    summary = runner._format_summary(result)
    assert "REPLAYED FROM CACHE" in summary
    assert "NOT NEW MEASUREMENTS" in summary


def test_a_run_that_made_requests_is_not_flagged_as_a_replay(
    wired: dict[str, Any], tmp_path: Path
) -> None:
    """The other half of the guard: a real run must not cry wolf."""
    result = runner.run_baseline(
        max_tasks=2,
        logdir=tmp_path / "logs",
        out_path=tmp_path / "out.json",
    )
    assert result["replayed"] is False
    assert "REPLAYED FROM CACHE" not in runner._format_summary(result)


# The configuration a pipeline name is built from; each test below changes one key.
# The defense keys are REQUIRED arguments, not defaulted ones: the defense is the
# discriminator that keeps a defended run out of the undefended run's cache, so a
# caller that forgets it should not silently get the baseline's name.
_NAME_BASE: dict[str, Any] = {
    "provider": runner.Provider.OPENAI_COMPAT,
    "model": "openai/gpt-oss-120b",
    "base_url": runner.DEFAULT_BASE_URL,
    "timeout": 90.0,
    "reasoning_effort": "low",
    "benchmark_version": "v1.2",
    "defense": runner.Defense.NONE,
    "defense_layers": (),
}


def test_pipeline_name_keeps_the_local_token_and_is_path_safe() -> None:
    """Two hard requirements on a name that is also a directory and a cache key.

    The literal ``local`` token must survive: the important_instructions attack
    substring-matches this name against AgentDojo's MODEL_NAMES table and RAISES
    when nothing matches, and ``local`` is the entry that maps to "Local model".
    And the name must be path-safe, because AgentDojo turns it straight into a path
    segment - the ``/`` in a model id like ``openai/gpt-oss-120b`` would otherwise
    split one run's cache across two directory levels.
    """
    name = runner._pipeline_name(**_NAME_BASE)
    assert "local" in name
    assert re.fullmatch(r"[A-Za-z0-9._-]+", name), name


def test_agentdojo_can_derive_a_model_name_from_the_pipeline_name() -> None:
    """The reason the ``local`` token matters, asserted against the real table.

    ``important_instructions`` calls ``get_model_name_from_pipeline`` while building
    its jailbreak text, and that function RAISES when no MODEL_NAMES key is a
    substring of the pipeline name. Our open models (gpt-oss via Groq, llama via
    NVIDIA, gemini-3.5) are in no version of that table, so dropping or respelling
    the token would kill the run at attack-construction time - before a single task
    - and the fix would look like a naming nit. Asserting the substring is a proxy;
    this calls the function AgentDojo actually calls.
    """
    from agentdojo.attacks.base_attacks import get_model_name_from_pipeline

    configurations = [
        _NAME_BASE,
        {**_NAME_BASE, "model": "meta/llama-3.1-8b-instruct"},
        {
            **_NAME_BASE,
            "provider": runner.Provider.GEMINI,
            "model": runner.DEFAULT_GEMINI_MODEL,
            "reasoning_effort": None,
        },
    ]
    for config in configurations:
        pipeline = SimpleNamespace(name=runner._pipeline_name(**config))
        assert get_model_name_from_pipeline(pipeline) == "Local model", pipeline.name


def test_pipeline_name_is_stable_for_an_identical_configuration() -> None:
    """The fingerprint is a digest, not a nonce - re-running the SAME configuration
    under ``--resume`` has to find its own logs again."""
    assert runner._pipeline_name(**_NAME_BASE) == runner._pipeline_name(**_NAME_BASE)


@pytest.mark.parametrize(
    "field,value",
    [
        ("reasoning_effort", "high"),
        ("reasoning_effort", None),
        ("base_url", "https://integrate.api.nvidia.com/v1"),
        ("timeout", 5.0),
        ("benchmark_version", "v1.1"),
        ("model", "meta/llama-3.1-8b-instruct"),
        ("provider", runner.Provider.GEMINI),
        # The defense is a score-changing setting like any other - more so, since a
        # defended run served the baseline's cache would report the baseline's ASR
        # as the defense's own.
        ("defense", runner.Defense.AEGIS),
    ],
)
def test_pipeline_name_changes_with_anything_that_can_change_a_score(
    field: str, value: Any
) -> None:
    """Every setting able to change a result is in the cache key.

    ``pipeline.name`` is the ONLY discriminator in AgentDojo's cache path, so a
    setting missing from it is a way for one run to be served another run's numbers
    - which is exactly what a name carrying the model and nothing else allowed.
    """
    changed = {**_NAME_BASE, field: value}
    assert runner._pipeline_name(**changed) != runner._pipeline_name(**_NAME_BASE)
    assert "local" in runner._pipeline_name(**changed)


def test_pipeline_name_ignores_settings_that_did_not_apply() -> None:
    """A gemini run is not fingerprinted against an endpoint it never called.

    The openai-compat-only settings are fingerprinted as ``None`` for gemini -
    exactly the way the result JSON records them - so two gemini runs differing only
    in a flag neither of them could use share a cache entry, which is correct.
    """
    gemini = {**_NAME_BASE, "provider": runner.Provider.GEMINI, "reasoning_effort": None}
    other_endpoint = {**gemini, "base_url": "https://example.invalid/v1", "timeout": 5.0}
    assert runner._pipeline_name(**gemini) == runner._pipeline_name(**other_endpoint)


# -- the honesty helpers -------------------------------------------------------


def test_mean_reports_none_for_an_empty_bag() -> None:
    """ "Nothing ran" is None, never 0.0.

    0.0 is a real score - "ran, and every single one failed" - so an empty phase
    reported as 0.0 is indistinguishable from a perfectly defended one. This is the
    assertion that makes ``return None`` a decision instead of an accident.
    """
    assert runner._mean([]) is None


def test_mean_averages_booleans() -> None:
    assert runner._mean([True, True, False]) == pytest.approx(2 / 3)
    assert runner._mean([False, False]) == 0.0
    assert runner._mean([True]) == 1.0


def test_format_summary_renders_na_for_a_metric_that_did_not_run(
    wired: dict[str, Any], tmp_path: Path
) -> None:
    """A missing metric prints ``n/a``, not ``0.0%``.

    The console is where a number gets read and quoted, so the distinction _mean
    preserves has to survive rendering too - a phase that never ran must not print
    as a flawless defensive score.
    """
    result = runner.run_baseline(
        max_tasks=2,
        logdir=tmp_path / "logs",
        out_path=tmp_path / "out.json",
    )
    result["utility"] = None
    result["timeout"] = None

    lines = runner._format_summary(result).splitlines()
    utility_line = next(line for line in lines if line.startswith(" utility  "))
    assert "n/a" in utility_line
    assert "0.0%" not in utility_line
    # A non-metric field that legitimately does not apply renders the same way.
    assert "n/a" in next(line for line in lines if line.startswith(" request timeout"))
    # ... while a metric that DID run still prints as a percentage.
    assert "%" in next(line for line in lines if line.startswith(" attack success rate"))


# -- the flags that can change a score must be recorded ------------------------


def test_timeout_and_throttle_reach_the_element_and_the_result_json(
    wired: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--timeout`` and ``--min-request-interval`` are plumbed AND recorded.

    Both were previously invisible: the timeout was help-texted but written
    nowhere, and the throttle interval had no CLI entry path at all, so every real
    run silently used the built-in default while the docs promised a paid tier could
    pass 0. A timeout can fail a task and move the score, so two runs that differed
    in it must not look byte-for-byte comparable.
    """
    out_path = tmp_path / "out.json"
    runner.main(
        [
            "--max-tasks",
            "1",
            "--timeout",
            "12.5",
            "--min-request-interval",
            "0",
            "--logdir",
            str(tmp_path / "logs"),
            "--out",
            str(out_path),
        ]
    )
    capsys.readouterr()

    assert wired["built"][0]["timeout"] == 12.5
    assert wired["built"][0]["min_request_interval_s"] == 0.0

    on_disk = json.loads(out_path.read_text(encoding="utf-8"))
    assert on_disk["timeout"] == 12.5
    assert on_disk["min_request_interval_s"] == 0.0


def test_omitted_openai_flags_fall_back_to_the_documented_defaults(
    wired: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The None sentinels that let gemini reject these flags must not leak through
    as the value an openai-compat run actually runs with."""
    out_path = tmp_path / "out.json"
    runner.main(["--max-tasks", "1", "--logdir", str(tmp_path / "logs"), "--out", str(out_path)])
    capsys.readouterr()

    on_disk = json.loads(out_path.read_text(encoding="utf-8"))
    assert on_disk["timeout"] == runner.DEFAULT_TIMEOUT
    assert on_disk["min_request_interval_s"] == DEFAULT_MIN_REQUEST_INTERVAL_S


def test_gemini_records_null_for_every_setting_it_cannot_use(
    wired: dict[str, Any], tmp_path: Path
) -> None:
    """Recording a value the run could not use would be a claim about a knob that
    did nothing; null is the honest entry."""
    result = runner.run_baseline(
        provider="gemini",
        max_tasks=1,
        logdir=tmp_path / "logs",
        out_path=tmp_path / "out.json",
    )
    assert result["timeout"] is None
    assert result["min_request_interval_s"] is None
    assert result["base_url"] is None
    assert result["reasoning_effort"] is None
    # The key VARIABLE is still recorded - that one applies to every provider.
    assert result["api_key_env"] == runner.DEFAULT_GEMINI_KEY_ENV


def test_api_key_env_records_the_variable_name_that_was_asked_for(
    wired: dict[str, Any], tmp_path: Path
) -> None:
    """A custom key variable is recorded by NAME, so a result can be audited back to
    the credential that produced it. A name is not a secret; a value would be."""
    result = runner.run_baseline(
        api_key_env="MY_CUSTOM_KEY_VAR",
        max_tasks=1,
        logdir=tmp_path / "logs",
        out_path=tmp_path / "out.json",
    )
    assert result["api_key_env"] == "MY_CUSTOM_KEY_VAR"


def test_defense_is_read_off_the_built_pipeline_not_hardcoded(
    wired: dict[str, Any], tmp_path: Path
) -> None:
    """When the defense seam is filled, the JSON has to follow it automatically.

    The field used to be a literal ``None`` in the result dict, which would keep
    reporting "undefended" for a defended run until someone remembered the second
    place. Here the builder reports a defense and the JSON must agree.
    """
    wired["build_pipeline"].defense = "aegis-defense-v1"
    result = runner.run_baseline(
        max_tasks=1,
        logdir=tmp_path / "logs",
        out_path=tmp_path / "out.json",
    )
    assert result["defense"] == "aegis-defense-v1"


# -- the real element builders -------------------------------------------------


def test_build_pipeline_openai_is_undefended_and_wraps_the_element(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real _build_pipeline builds the counted OpenAI element with the requested
    reasoning effort and hands it to from_config with defense=None (the baseline,
    and the future defense seam), and reports that defense back rather than a
    literal."""
    from evals.agentdojo.openai_llm import CountingOpenAILLM

    captured: dict[str, Any] = {}

    class _FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            # build_openai_compat_llm patches chat.completions.create (endpoint guards).
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: None))

    def _fake_from_config(config: Any) -> Any:
        captured["config"] = config
        return SimpleNamespace(name=None)

    monkeypatch.setattr("openai.OpenAI", _FakeOpenAI)
    monkeypatch.setenv("GROQ_API_KEY", "gsk-fake-not-real")
    monkeypatch.setattr(runner.AgentPipeline, "from_config", _fake_from_config)

    built = runner._build_pipeline(
        provider=runner.Provider.OPENAI_COMPAT,
        model=runner.DEFAULT_OPENAI_MODEL,
        base_url=runner.DEFAULT_BASE_URL,
        api_key_env=runner.DEFAULT_OPENAI_KEY_ENV,
        timeout=90.0,
        reasoning_effort="low",
    )

    element = built.element
    assert isinstance(element, CountingOpenAILLM)
    assert element.reasoning_effort == "low"
    assert captured["config"].defense is None
    assert captured["config"].model_id == runner.DEFAULT_OPENAI_MODEL
    assert captured["config"].llm is element
    # The defense reported is READ OFF the config that was built, so the result JSON
    # cannot keep claiming "none" once the defense seam is filled.
    assert built.defense is captured["config"].defense
    # The "local" token is required: the important_instructions attack derives a
    # model name by substring-matching pipeline.name against AgentDojo's MODEL_NAMES
    # table and raises when nothing matches; "local" maps to "Local model".
    assert "local" in built.pipeline.name
    assert built.pipeline.name.startswith(runner._PIPELINE_NAME_PREFIX)


def test_gemini_element_fails_on_the_named_key_var_not_a_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A custom --api-key-env must not silently fall back to GEMINI_API_KEY.

    If it did, a run asked for with one key would quietly execute with another and
    the operator would never learn which. The error names the variable that was
    asked for - and, as everywhere, never a value.
    """
    from evals.agentdojo.gemini_llm import Gemini3LLMError

    leaked = "AIza-SHOULD-NEVER-BE-USED"
    monkeypatch.setenv("GEMINI_API_KEY", leaked)
    monkeypatch.delenv("MY_OTHER_GEMINI_VAR", raising=False)

    with pytest.raises(Gemini3LLMError) as exc:
        runner._build_element(
            provider=runner.Provider.GEMINI,
            model=runner.DEFAULT_GEMINI_MODEL,
            base_url="",
            api_key_env="MY_OTHER_GEMINI_VAR",
            timeout=90.0,
        )

    message = str(exc.value)
    assert "MY_OTHER_GEMINI_VAR" in message
    assert "GEMINI_API_KEY" not in message
    assert leaked not in message


# -- the defense seam ----------------------------------------------------------
#
# Four classes of mistake are expensive here, and each has tests below:
#
# * a layer spec that parses to the WRONG arm (a typo accepted as "off") - the run
#   is then labelled with the arm that was asked for and measures another one;
# * a defense that does not reach the result JSON - two runs are comparable only if
#   a reader can see what was on;
# * a defended run sharing the undefended run's pipeline name - AgentDojo would
#   serve it the baseline's cached numbers and they would be published as the
#   defense's own, which is the single most damaging cache hit available here;
# * --defense none drifting away from the recorded baseline's behaviour.


def test_parse_defense_layers_accepts_the_layer_names_in_canonical_order() -> None:
    """Order and duplicates in the spec do not change the arm it names.

    ``gate,spotlight`` and ``spotlight,gate`` are the same run, so they must
    produce the same layer tuple - which is what the result JSON records and what
    the cache key digests.
    """
    layers = runner.parse_defense_layers("gate,spotlight")
    assert layers == (runner.DefenseLayer.SPOTLIGHT, runner.DefenseLayer.GATE)
    assert runner.parse_defense_layers("spotlight,gate") == layers
    assert runner.parse_defense_layers(" GATE , spotlight,gate ") == layers


def test_parse_defense_layers_aliases() -> None:
    """``all`` is every exposed layer; ``none`` is the all-off control arm."""
    assert runner.parse_defense_layers("all") == runner.DEFAULT_DEFENSE_LAYERS
    assert runner.parse_defense_layers("all") == tuple(runner.DefenseLayer)
    assert runner.parse_defense_layers("none") == ()


@pytest.mark.parametrize("spec", ["l2", "spotlight,quarantine", "gate,typo", ""])
def test_parse_defense_layers_refuses_unknown_names_and_names_the_valid_ones(
    spec: str,
) -> None:
    """An unrecognised layer is refused, and the error says what is valid.

    Silently dropping it would run an ablation arm nobody asked for and record the
    arm they did ask for. The message has to carry the vocabulary, because the
    operator's next action is to retype the flag.
    """
    with pytest.raises(ValueError) as exc:
        runner.parse_defense_layers(spec)
    message = str(exc.value)
    for layer in runner.DefenseLayer:
        assert layer.value in message


@pytest.mark.parametrize("spec", ["all", "none", "gate", "spotlight,gate", "GATE , gate"])
def test_run_baseline_and_the_cli_parse_layer_names_the_same_way(spec: str) -> None:
    """One vocabulary, two doors - and the programmatic one is not the poor relation.

    ``_resolve_defense_layers`` used to re-implement the parse and reject ``all``
    and ``none`` in an error message that advertised them as valid: it told the
    caller a value was valid while refusing it. That door is the one an ablation
    sweep looping over layer sets goes through, so it delegates rather than
    duplicating - a rule that exists twice is a rule that drifts.
    """
    resolved = runner._resolve_defense_layers(runner.Defense.AEGIS, spec.split(","))
    assert resolved == runner.parse_defense_layers(spec)


def test_the_programmatic_path_accepts_the_aliases_its_error_advertises() -> None:
    """The exact defect: ``run_baseline(defense_layers=["all"])`` used to raise."""
    assert (
        runner._resolve_defense_layers(runner.Defense.AEGIS, ["all"])
        == runner.DEFAULT_DEFENSE_LAYERS
    )
    assert runner._resolve_defense_layers(runner.Defense.AEGIS, ["none"]) == ()


def test_an_already_resolved_layer_tuple_survives_the_round_trip() -> None:
    """The CLI hands back a resolved tuple, and an EMPTY one is the all-off arm.

    ``()`` is what ``--defense-layers none`` produces and it must stay the all-off
    control arm, not be re-parsed as the empty spec ``""`` (which is an error).
    """
    assert runner._resolve_defense_layers(runner.Defense.AEGIS, ()) == ()
    assert (
        runner._resolve_defense_layers(runner.Defense.AEGIS, runner.DEFAULT_DEFENSE_LAYERS)
        == runner.DEFAULT_DEFENSE_LAYERS
    )
    assert (
        runner._resolve_defense_layers(runner.Defense.AEGIS, None) == runner.DEFAULT_DEFENSE_LAYERS
    )


@pytest.mark.parametrize("spec", ["all,gate", "none,gate", "gate,none"])
def test_parse_defense_layers_refuses_an_alias_mixed_with_layers(spec: str) -> None:
    """``none,gate`` is a contradiction and ``all,gate`` is a guess about precedence.

    Both are refused rather than resolved, because either resolution silently runs
    an arm the operator did not name.
    """
    with pytest.raises(ValueError, match="whole set"):
        runner.parse_defense_layers(spec)


def test_defense_layers_are_rejected_without_a_defense(
    wired: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A flag that cannot do anything is a lie - the same rule the gemini-only flags
    get. Recording layers for a pipeline that has none would describe a defense that
    was never built."""
    with pytest.raises(SystemExit):
        runner.main(["--defense-layers", "gate", *_sink_flags(tmp_path)])
    assert wired["clean"].calls == []  # refused before anything ran
    assert "--defense-layers" in capsys.readouterr().err


def test_unknown_defense_layer_is_refused_by_the_cli_before_any_run(
    wired: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """parser.error, not a traceback, and the message names every valid layer."""
    with pytest.raises(SystemExit):
        runner.main(["--defense", "aegis", "--defense-layers", "l5", *_sink_flags(tmp_path)])
    assert wired["clean"].calls == []
    stderr = capsys.readouterr().err
    for layer in runner.DefenseLayer:
        assert layer.value in stderr


def test_run_baseline_refuses_defense_layers_for_the_undefended_pipeline(
    wired: dict[str, Any], tmp_path: Path
) -> None:
    """The second door: the CLI guard is repeated for direct callers of run_baseline,
    exactly as the --max-tasks guard is."""
    with pytest.raises(ValueError, match="defense-layers"):
        runner.run_baseline(
            defense="none",
            defense_layers=["gate"],
            logdir=tmp_path / "logs",
            out_path=tmp_path / "out.json",
        )
    with pytest.raises(ValueError, match="unknown defense layer"):
        runner.run_baseline(
            defense="aegis",
            defense_layers=["l5"],
            logdir=tmp_path / "logs",
            out_path=tmp_path / "out.json",
        )


def test_defense_name_and_layers_reach_the_result_json(
    wired: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The exact arm is written to the file, not just "a defense was on"."""
    out_path = tmp_path / "out.json"
    wired["build_pipeline"].defense = "aegis-l1+l3+l5"
    runner.main(
        [
            "--max-tasks",
            "1",
            "--defense",
            "aegis",
            "--defense-layers",
            "detect,gate",
            "--logdir",
            str(tmp_path / "logs"),
            "--out",
            str(out_path),
        ]
    )
    capsys.readouterr()

    on_disk = json.loads(out_path.read_text(encoding="utf-8"))
    assert on_disk["defense_name"] == "aegis"
    assert on_disk["defense_layers"] == ["detect", "gate"]
    # The label is READ OFF what was built, so it cannot disagree with the arm.
    assert on_disk["defense"] == "aegis-l1+l3+l5"
    # ... and the builder really was asked for that arm.
    built = wired["built"][0]
    assert built["defense"] is runner.Defense.AEGIS
    assert built["defense_layers"] == (runner.DefenseLayer.DETECT, runner.DefenseLayer.GATE)


def test_the_result_json_records_agentdojos_own_cache_key(
    wired: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The name is the log directory AND the cache key, so the file has to carry it.

    AgentDojo writes every transcript to ``logdir/<pipeline_name>/suite/...`` and
    keys its result cache on the same string. A result file that records the
    settings but not the name cannot be tied back to the transcripts that produced
    it, so the ``replayed`` flag next to it is auditable only from the console of a
    run that is already over.
    """
    out_path = tmp_path / "out.json"
    wired["build_pipeline"].pipeline.name = "aegis-baseline-local-fake-name-deadbeef"
    runner.main(["--max-tasks", "1", "--logdir", str(tmp_path / "logs"), "--out", str(out_path)])
    capsys.readouterr()

    on_disk = json.loads(out_path.read_text(encoding="utf-8"))
    assert on_disk["pipeline_name"] == "aegis-baseline-local-fake-name-deadbeef"
    # Read off the pipeline that was built, never restated - a name the runner
    # recomputed for the file could differ from the one AgentDojo actually keyed on.
    assert on_disk["pipeline_name"] == wired["build_pipeline"].pipeline.name


def test_run_baseline_takes_the_aliases_the_cli_takes(
    wired: dict[str, Any], tmp_path: Path
) -> None:
    """The programmatic door, end to end: an ablation sweep passes layer sets here.

    ``defense_layers=["all"]`` used to raise a ValueError whose own message listed
    ``all`` as valid.
    """
    everything = runner.run_baseline(
        defense="aegis",
        defense_layers=["all"],
        max_tasks=1,
        logdir=tmp_path / "logs",
        out_path=tmp_path / "all.json",
    )
    nothing = runner.run_baseline(
        defense="aegis",
        defense_layers=["none"],
        max_tasks=1,
        logdir=tmp_path / "logs",
        out_path=tmp_path / "none.json",
    )

    assert everything["defense_layers"] == [layer.value for layer in runner.DEFAULT_DEFENSE_LAYERS]
    assert nothing["defense_layers"] == []
    assert wired["built"][0]["defense_layers"] == runner.DEFAULT_DEFENSE_LAYERS
    assert wired["built"][1]["defense_layers"] == ()


def test_defense_aegis_defaults_to_every_exposed_layer(
    wired: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--defense aegis`` with no layer list is the full-defense arm, and says so."""
    out_path = tmp_path / "out.json"
    runner.main(
        [
            "--max-tasks",
            "1",
            "--defense",
            "aegis",
            "--logdir",
            str(tmp_path / "logs"),
            "--out",
            str(out_path),
        ]
    )
    capsys.readouterr()
    on_disk = json.loads(out_path.read_text(encoding="utf-8"))
    assert on_disk["defense_layers"] == [layer.value for layer in runner.DEFAULT_DEFENSE_LAYERS]


def test_defense_none_is_the_unchanged_baseline(
    wired: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The recorded baseline command still measures exactly what it always did.

    ``--defense none`` is also the default, so this pins the un-flagged run too: an
    undefended build, no layers, no ledger, and a ``defense`` field that stays null.
    """
    out_path = tmp_path / "out.json"
    runner.main(["--max-tasks", "1", "--logdir", str(tmp_path / "logs"), "--out", str(out_path)])
    capsys.readouterr()
    default_run = json.loads(out_path.read_text(encoding="utf-8"))

    runner.main(
        [
            "--max-tasks",
            "1",
            "--defense",
            "none",
            "--logdir",
            str(tmp_path / "logs"),
            "--out",
            str(out_path),
        ]
    )
    capsys.readouterr()
    explicit_run = json.loads(out_path.read_text(encoding="utf-8"))

    for result in (default_run, explicit_run):
        assert result["defense"] is None
        assert result["defense_name"] == "none"
        assert result["defense_layers"] == []
        assert result["gate_ledger"] is None
    # Two runs of the same undefended configuration differ only in their timestamp.
    assert {k: v for k, v in default_run.items() if k != "timestamp"} == {
        k: v for k, v in explicit_run.items() if k != "timestamp"
    }
    # The builder was asked for the undefended pipeline, with no layers.
    for built in wired["built"]:
        assert built["defense"] is runner.Defense.NONE
        assert built["defense_layers"] == ()


def test_the_undefended_name_is_the_documented_formula_and_nothing_else() -> None:
    """The baseline's cache key, recomputed by hand from what is documented.

    The undefended name used to be byte-for-byte the pre-defense five-part digest,
    so an interrupted baseline could resume against its own logs. Adding the MODEL
    moved it, deliberately: the model appeared in the name only as a token, and
    ``_sanitize_name`` folds every path-hostile character into ``_``, so
    ``openai/gpt-oss-120b``, ``openai_gpt-oss-120b`` and ``openai:gpt-oss-120b``
    produced ONE name and ONE digest - three models, one cache entry. A stale name
    only makes a re-run actually re-run, which is the safe direction; a colliding
    one publishes one model's numbers under another's.

    The defense parts are still APPENDED only when a defense is on, so no defended
    arm can collide with this. If this name changes again, that is a decision to
    make deliberately, not one to discover after a --resume silently re-ran a paid
    benchmark.
    """
    import hashlib

    payload = "\x1f".join(
        [
            "openai-compat",
            "openai/gpt-oss-120b",
            runner.DEFAULT_BASE_URL,
            "90.0",
            "low",
            "v1.2",
        ]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[: runner._FINGERPRINT_CHARS]
    expected = runner._sanitize_name(
        f"{runner._PIPELINE_NAME_PREFIX}-openai/gpt-oss-120b-low-{digest}"
    )
    assert runner._pipeline_name(**_NAME_BASE) == expected


@pytest.mark.parametrize("model", ["openai_gpt-oss-120b", "openai:gpt-oss-120b"])
def test_two_model_ids_the_sanitizer_folds_together_are_still_two_runs(model: str) -> None:
    """The collision the digest closes, spelled out.

    These ids sanitize to the same readable token as ``openai/gpt-oss-120b``, so
    while the model was only NAMED they produced an identical directory AND an
    identical digest - and AgentDojo would have served one model's cached results
    to another. The token can still collide; the digest may not.
    """
    canonical = runner._pipeline_name(**_NAME_BASE)
    variant = runner._pipeline_name(**{**_NAME_BASE, "model": model})

    assert variant != canonical, model
    assert "local" in variant and re.fullmatch(r"[A-Za-z0-9._-]+", variant)


def test_the_defense_label_is_part_of_the_cache_key() -> None:
    """Two arms differing only in something the LAYER SET cannot say are two runs.

    ``DefenseConfig.label`` carries the spotlight STYLE, which changes what the
    model is shown and therefore what the run measures, while the layer token says
    only "l2 was on". The style is fixed today, so this drives it through the seam
    the runner builds its config with - the point is that the digest reads the
    label, not that a flag exists yet.
    """
    from evals.agentdojo.defense import DefenseConfig, SpotlightStyle

    arm: dict[str, Any] = {
        **_NAME_BASE,
        "defense": runner.Defense.AEGIS,
        "defense_layers": runner.DEFAULT_DEFENSE_LAYERS,
    }
    datamark = runner._pipeline_name(**arm)

    original = runner._defense_config
    try:
        runner._defense_config = lambda layers: DefenseConfig(  # type: ignore[assignment]
            spotlight=True, detect=True, gate=True, spotlight_style=SpotlightStyle.DELIMIT
        )
        delimited = runner._pipeline_name(**arm)
    finally:
        runner._defense_config = original  # type: ignore[assignment]

    assert delimited != datamark, "the style is in the label, so it is in the digest"


def test_a_defended_run_can_never_be_served_the_undefended_cache() -> None:
    """The fingerprint is the ONLY mechanism separating these two runs.

    Same suite, same task ids, same model, same endpoint: nothing else in
    AgentDojo's cache path differs between a defended run and the baseline. Without
    the defense in the name, the defended run is served the baseline's results and
    reports the baseline's ASR as the defense's own.
    """
    baseline = runner._pipeline_name(**_NAME_BASE)
    names = {baseline}
    for layers in (
        (),
        (runner.DefenseLayer.GATE,),
        (runner.DefenseLayer.SPOTLIGHT, runner.DefenseLayer.GATE),
        runner.DEFAULT_DEFENSE_LAYERS,
    ):
        name = runner._pipeline_name(
            **{**_NAME_BASE, "defense": runner.Defense.AEGIS, "defense_layers": layers}
        )
        # Distinct from the baseline AND from every other arm: two arms of the
        # ablation are two different measurements.
        assert name not in names, name
        names.add(name)
        # The two hard requirements on any name still hold - the attack derives a
        # model identity from it, and AgentDojo makes it a directory.
        assert "local" in name
        assert re.fullmatch(r"[A-Za-z0-9._-]+", name), name


def test_defended_pipeline_name_is_stable_for_an_identical_arm() -> None:
    """A digest, not a nonce: re-running one arm under --resume finds its own logs."""
    arm: dict[str, Any] = {
        **_NAME_BASE,
        "defense": runner.Defense.AEGIS,
        "defense_layers": (runner.DefenseLayer.GATE,),
    }
    assert runner._pipeline_name(**arm) == runner._pipeline_name(**arm)


def test_agentdojo_can_derive_a_model_name_from_a_defended_pipeline_name() -> None:
    """The ``local`` token has to survive the defense suffix too.

    ``build_aegis_pipeline`` names its pipeline after the DefenseConfig label, which
    carries no model identity at all; the runner overwrites that name for exactly
    this reason. If it did not, the important_instructions attack would raise while
    being constructed - before a single task ran.
    """
    from agentdojo.attacks.base_attacks import get_model_name_from_pipeline

    pipeline = SimpleNamespace(
        name=runner._pipeline_name(
            **{
                **_NAME_BASE,
                "defense": runner.Defense.AEGIS,
                "defense_layers": runner.DEFAULT_DEFENSE_LAYERS,
            }
        )
    )
    assert get_model_name_from_pipeline(pipeline) == "Local model"


def _gate_entry(tool: str, verdict: Any, action: Any) -> Any:
    from evals.agentdojo.defense import GateEntry

    from aegis.domain.trust import TrustTier

    return GateEntry(
        conversation_key="task-a",
        tool_name=tool,
        verdict=verdict,
        action=action,
        effective_tier=TrustTier.UNTRUSTED,
    )


def _fake_aegis(entries: list[Any]) -> Any:
    """A stand-in for the built defense carrying real ledger entries.

    The entries are real :class:`GateEntry` values, so the summary aggregates the
    same enum members the executor writes; only the elements holding them are
    stubbed, because building the real ones needs an LLM.
    """
    from evals.agentdojo.defense import GateAction

    executor = SimpleNamespace(
        ledger=tuple(entries),
        refusals=tuple(e for e in entries if e.action is GateAction.REFUSED),
        failures=0,
    )
    guard = SimpleNamespace(failures=0, quarantine_failures=2)
    return SimpleNamespace(executor=executor, guard=guard)


def test_gate_ledger_summary_lands_in_the_result_json(
    wired: dict[str, Any], tmp_path: Path
) -> None:
    """The evidence AgentDojo's own booleans cannot carry.

    ``security_results`` is one boolean per couple: it cannot distinguish "the gate
    refused the call" from "the model happened not to take the bait", and only the
    first is a defense. So the ledger is summarised into the file - per tool, and on
    both axes, because a CONFIRM refused for want of a human and a taint-only DENY
    overridden on a read-only tool are decisions the verdict and the action disagree
    about by design.
    """
    from evals.agentdojo.defense import GateAction

    from aegis.security.capabilities import Verdict

    wired["build_pipeline"].defense = "aegis-l1+l2-datamark+l3+l5"
    wired["build_pipeline"].aegis = _fake_aegis(
        [
            _gate_entry("send_email", Verdict.DENY, GateAction.REFUSED),
            _gate_entry("send_email", Verdict.ALLOW, GateAction.EXECUTED),
            _gate_entry("send_email", Verdict.CONFIRM, GateAction.REFUSED),
            _gate_entry("read_inbox", Verdict.DENY, GateAction.EXECUTED),
        ]
    )
    out_path = tmp_path / "out.json"
    result = runner.run_baseline(
        defense="aegis",
        max_tasks=1,
        logdir=tmp_path / "logs",
        out_path=out_path,
    )

    ledger = json.loads(out_path.read_text(encoding="utf-8"))["gate_ledger"]
    assert ledger == result["gate_ledger"]
    assert ledger["decisions"] == 4
    assert ledger["refusals"] == 2
    assert ledger["refused_tools"] == ["send_email"]
    assert ledger["by_tool"]["send_email"] == {
        "allow": 1,
        "confirm": 1,
        "deny": 1,
        "executed": 1,
        "refused": 2,
    }
    # A read-only DENY that was executed anyway shows up on BOTH axes, which is the
    # whole reason both are recorded.
    assert ledger["by_tool"]["read_inbox"]["deny"] == 1
    assert ledger["by_tool"]["read_inbox"]["executed"] == 1
    # A layer that silently did not run must not be reported as one that did.
    assert ledger["gate_failures"] == 0
    assert ledger["guard_failures"] == 0
    assert ledger["quarantine_failures"] == 2

    summary = runner._format_summary(result)
    assert "gate decisions" in summary
    assert "send_email" in summary


def test_suite_tool_names_reach_the_builder_and_tolerate_a_suite_without_tools(
    wired: dict[str, Any], tmp_path: Path
) -> None:
    """L3's tool_invocation_attempt pattern needs the LIVE suite's tool names.

    Without them that flag can never fire, and the detect arm would quietly measure
    a weaker detector than the one it names. A suite that does not expose tools
    costs a flag, not the run.
    """
    assert runner._suite_tool_names(SimpleNamespace()) == ()
    assert runner._suite_tool_names(SimpleNamespace(tools=None)) == ()
    assert runner._suite_tool_names(
        SimpleNamespace(tools=[SimpleNamespace(name="send_email"), SimpleNamespace(other=1)])
    ) == ("send_email",)

    wired["suite"].tools = [SimpleNamespace(name="send_email")]
    runner.run_baseline(max_tasks=1, logdir=tmp_path / "logs", out_path=tmp_path / "out.json")
    assert wired["built"][0]["tool_names"] == ("send_email",)


def test_build_pipeline_aegis_on_openai_compat_wires_both_elements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The measured configuration, defended: Groq/openai-compat + the Aegis elements.

    Built for real (no from_config monkeypatch - ``build_aegis_pipeline`` composes
    the element list itself) and entirely offline: the OpenAI client is faked and no
    request is made. What is asserted is the contract the defense module documents -
    the gate takes ToolsExecutor's seat, the guard sits immediately after it, and the
    agent-under-test element is the one that actually runs.
    """
    from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop
    from evals.agentdojo.defense import AegisGatedToolsExecutor, AegisToolOutputGuard
    from evals.agentdojo.openai_llm import CountingOpenAILLM

    class _FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: None))

    monkeypatch.setattr("openai.OpenAI", _FakeOpenAI)
    monkeypatch.setenv("GROQ_API_KEY", "gsk-fake-not-real")

    built = runner._build_pipeline(
        provider=runner.Provider.OPENAI_COMPAT,
        model=runner.DEFAULT_OPENAI_MODEL,
        base_url=runner.DEFAULT_BASE_URL,
        api_key_env=runner.DEFAULT_OPENAI_KEY_ENV,
        timeout=90.0,
        reasoning_effort="low",
        defense=runner.Defense.AEGIS,
        defense_layers=runner.DEFAULT_DEFENSE_LAYERS,
        tool_names=("send_email",),
    )

    assert isinstance(built.element, CountingOpenAILLM)
    assert built.aegis is not None
    # The label of the arm that was built, not a literal restated here.
    assert built.defense == built.aegis.defense
    assert built.defense is not None and built.defense.startswith("aegis-")
    # One state object shared by both elements: a gate with its own state can never
    # refuse anything, which looks exactly like a working pipeline.
    assert built.aegis.executor.state is built.aegis.guard.state

    loop = list(built.pipeline.elements)[-1]
    assert isinstance(loop, ToolsExecutionLoop)
    gate, guard, llm = list(loop.elements)
    assert isinstance(gate, AegisGatedToolsExecutor)
    assert isinstance(guard, AegisToolOutputGuard)
    assert llm is built.element

    # The runner's name wins over build_aegis_pipeline's label-based one: the attack
    # substring-matches MODEL_NAMES against it and raises when nothing matches.
    assert "local" in built.pipeline.name
    assert built.pipeline.name.startswith(runner._PIPELINE_NAME_PREFIX)
    assert re.fullmatch(r"[A-Za-z0-9._-]+", built.pipeline.name)
    assert built.pipeline.name != runner._pipeline_name(
        provider=runner.Provider.OPENAI_COMPAT,
        model=runner.DEFAULT_OPENAI_MODEL,
        base_url=runner.DEFAULT_BASE_URL,
        timeout=90.0,
        reasoning_effort="low",
        benchmark_version=runner.DEFAULT_BENCHMARK_VERSION,
        defense=runner.Defense.NONE,
        defense_layers=(),
    )


def test_build_pipeline_aegis_layer_set_reaches_the_defense_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One CLI layer name, one DefenseConfig field - and L4 stays off.

    Quarantine is not exposed by the runner (it costs an extra model call per tool
    result), so no layer spec may turn it on; asking for it would also raise at
    construction for want of an extractor.
    """
    from evals.agentdojo.defense import DefenseConfig

    class _FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: None))

    monkeypatch.setattr("openai.OpenAI", _FakeOpenAI)
    monkeypatch.setenv("GROQ_API_KEY", "gsk-fake-not-real")

    built = runner._build_pipeline(
        provider=runner.Provider.OPENAI_COMPAT,
        model=runner.DEFAULT_OPENAI_MODEL,
        base_url=runner.DEFAULT_BASE_URL,
        api_key_env=runner.DEFAULT_OPENAI_KEY_ENV,
        timeout=90.0,
        reasoning_effort="low",
        defense=runner.Defense.AEGIS,
        defense_layers=(runner.DefenseLayer.GATE,),
    )
    assert built.aegis is not None
    config: DefenseConfig = built.aegis.executor.config
    assert (config.spotlight, config.detect, config.gate) == (False, False, True)
    assert config.quarantine is False
    assert built.aegis.guard.config == config


@pytest.mark.costly
def test_real_api_baseline_smoke(tmp_path: Path) -> None:
    """Real end-to-end run - skipped unless explicitly opted in. Do NOT run casually.

    Exercises the DEFAULT configuration (openai-compat / Groq / gpt-oss-120b at
    reasoning effort "low") - the one the recorded Week-0 baseline used, so a smoke
    run and a real run fail the same way. Gated on AEGIS_RUN_COSTLY=1 and a real
    GROQ_API_KEY so a normal ``pytest -m "not costly"`` never spends a cent of quota.

    Its output goes to ``tmp_path``, never to the defaults. A 1x1 smoke result
    written to :data:`runner.DEFAULT_OUT` would overwrite the recorded baseline
    with a two-task run that looks exactly like one - and that artifact costs real
    quota to reproduce. The logdir is separated for the same reason: a smoke run's
    task logs must not become cache entries a later real run could be served from.
    """
    import os

    if os.environ.get("AEGIS_RUN_COSTLY") != "1" or not os.environ.get("GROQ_API_KEY"):
        pytest.skip("costly: set AEGIS_RUN_COSTLY=1 and GROQ_API_KEY to run")
    result = runner.run_baseline(
        max_tasks=1,
        max_injection_tasks=1,
        out_path=tmp_path / "smoke.json",
        logdir=tmp_path / "logs",
    )
    assert 0.0 <= (result["asr"] or 0.0) <= 1.0


# --------------------------------------------------------------------------
# Explicit task selection, and the caveat it must carry
# --------------------------------------------------------------------------


def test_explicit_task_ids_replace_the_first_n_selection(
    wired: dict[str, Any], tmp_path: Path
) -> None:
    """Naming tasks runs those tasks, in numeric order, regardless of the caps.

    This exists to make a cheap re-measurement possible: the couples that carry
    the signal are usually a small subset, and re-running the whole grid to reach
    them is most of the token bill.
    """
    result = runner.run_baseline(
        max_tasks=8,
        max_injection_tasks=4,
        only_user_tasks=["user_task_3", "user_task_1"],
        only_injection_tasks=["injection_task_2"],
        logdir=tmp_path / "logs",
        out_path=tmp_path / "out.json",
    )

    assert result["user_task_ids"] == ["user_task_1", "user_task_3"], (
        "numeric order, not given order"
    )
    assert result["injection_task_ids"] == ["injection_task_2"]
    clean_kwargs = wired["clean"].calls[0]["kwargs"]
    assert clean_kwargs["user_tasks"] == ["user_task_1", "user_task_3"]


def test_a_hand_picked_run_declares_itself_screening_only(
    wired: dict[str, Any], tmp_path: Path
) -> None:
    """The number from a chosen subset must not be quotable as a rate.

    The specific hazard is not smallness, it is SELECTION: choosing the couples a
    previous arm failed on means the run can only observe failures the new arm
    fixes, never failures it introduces. ASR is biased downward and a paired
    p-value computed from it is invalid. The artifact says so about itself, so the
    caveat cannot be lost between the run and the write-up.
    """
    picked = runner.run_baseline(
        max_tasks=8,
        max_injection_tasks=4,
        only_user_tasks=["user_task_3"],
        logdir=tmp_path / "logs",
        out_path=tmp_path / "picked.json",
    )
    assert picked["screening_only"] is True
    assert picked["task_selection"] == "explicit"

    whole = runner.run_baseline(
        max_tasks=4,
        max_injection_tasks=2,
        logdir=tmp_path / "logs2",
        out_path=tmp_path / "whole.json",
    )
    assert whole["screening_only"] is False
    assert whole["task_selection"] == "first-n"


def test_an_unknown_task_id_is_refused_rather_than_dropped(
    wired: dict[str, Any], tmp_path: Path
) -> None:
    """A typo that silently selected fewer tasks would under-report a rate."""
    with pytest.raises(ValueError, match="does not have"):
        runner.run_baseline(
            only_user_tasks=["user_task_1", "user_task_nope"],
            logdir=tmp_path / "logs",
            out_path=tmp_path / "out.json",
        )
