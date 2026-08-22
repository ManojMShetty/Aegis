"""LLMRouter - config-driven role resolution, a shared budget, and fail-closed.

The router is exercised entirely through the offline ``fake`` provider kind, so
no key or network is involved; the ``gemini`` kind is validated against the
shipped config without ever building a live client.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from aegis.llm.base import Message, Role
from aegis.llm.budget import Budget, BudgetExceeded
from aegis.llm.providers.fake import FakeProvider
from aegis.llm.router import DEFAULT_MODELS_PATH, LLMRouter, RouterError

FAKE_CONFIG = dedent(
    """
    roles:
      quarantine:
        provider: fake
        model: fake-lite
        echo_json: true
      judge:
        provider: fake
        model: fake-lite
      privileged_agent:
        provider: fake
        model: fake-pro
      attacker:
        provider: fake
        model: fake-pro
    budget:
      max_requests_per_day: 3
      max_requests_per_run: 2
    """
)


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "models.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _router(tmp_path: Path, text: str = FAKE_CONFIG) -> LLMRouter:
    # Point the shared budget at a temp file so we never touch the repo's state.
    budget = Budget(
        path=tmp_path / "budget.json",
        max_requests_per_day=3,
        max_requests_per_run=2,
    )
    return LLMRouter.load(_write(tmp_path, text), budget=budget)


# --------------------------------------------------------------------------
# Basic routing
# --------------------------------------------------------------------------


def test_returns_a_fake_provider_for_a_fake_kind_role(tmp_path: Path) -> None:
    router = _router(tmp_path)
    provider = router.provider_for(Role.QUARANTINE)
    assert isinstance(provider, FakeProvider)


def test_provider_is_cached_per_role(tmp_path: Path) -> None:
    router = _router(tmp_path)
    first = router.provider_for(Role.JUDGE)
    second = router.provider_for(Role.JUDGE)
    assert first is second  # lazy build, then cached


def test_echo_json_flag_flows_into_the_fake_provider(tmp_path: Path) -> None:
    router = _router(tmp_path)
    provider = router.provider_for(Role.QUARANTINE)
    resp = provider.complete(
        [Message(role="user", content="hello")],
        json_schema={"type": "object"},
    )
    assert resp.text == '{"echo": "hello"}'


# --------------------------------------------------------------------------
# Shared budget
# --------------------------------------------------------------------------


def test_budget_is_shared_across_roles(tmp_path: Path) -> None:
    router = _router(tmp_path)
    quarantine = router.provider_for(Role.QUARANTINE)
    judge = router.provider_for(Role.JUDGE)

    quarantine.complete([Message(role="user", content="a")])
    judge.complete([Message(role="user", content="b")])

    # Both calls landed on the one budget the router owns.
    assert router.budget.request_count == 2


def test_shared_budget_cap_spans_roles(tmp_path: Path) -> None:
    router = _router(tmp_path)  # per-run cap is 2
    router.provider_for(Role.QUARANTINE).complete([Message(role="user", content="a")])
    router.provider_for(Role.JUDGE).complete([Message(role="user", content="b")])

    # The third call across any role trips the shared per-run cap.
    with pytest.raises(BudgetExceeded):
        router.provider_for(Role.ATTACKER).complete([Message(role="user", content="c")])


def test_budget_built_from_config_when_not_injected(tmp_path: Path) -> None:
    router = LLMRouter.load(_write(tmp_path, FAKE_CONFIG))
    assert router.budget.max_requests_per_day == 3
    assert router.budget.max_requests_per_run == 2


# --------------------------------------------------------------------------
# Fail closed
# --------------------------------------------------------------------------


def test_unknown_role_fails_closed(tmp_path: Path) -> None:
    text = dedent(
        """
        roles:
          judge:
            provider: fake
            model: fake-lite
        """
    )
    router = _router(tmp_path, text)
    with pytest.raises(RouterError, match="no config for role 'quarantine'"):
        router.provider_for(Role.QUARANTINE)


def test_unknown_provider_kind_fails_closed(tmp_path: Path) -> None:
    text = dedent(
        """
        roles:
          judge:
            provider: wizardry
            model: crystal-ball
        """
    )
    router = _router(tmp_path, text)
    with pytest.raises(RouterError, match="unknown provider kind"):
        router.provider_for(Role.JUDGE)


def test_role_missing_model_fails_closed(tmp_path: Path) -> None:
    text = dedent(
        """
        roles:
          judge:
            provider: fake
        """
    )
    router = _router(tmp_path, text)
    with pytest.raises(RouterError, match="missing a 'model'"):
        router.provider_for(Role.JUDGE)


def test_missing_config_file_raises(tmp_path: Path) -> None:
    with pytest.raises(RouterError, match="not found"):
        LLMRouter.load(tmp_path / "nope.yaml")


@pytest.mark.parametrize(
    "bad_value",
    ['"200"', "true", "-5", "0"],  # quoted int, bool, negative, zero
)
def test_malformed_budget_cap_fails_closed(tmp_path: Path, bad_value: str) -> None:
    """A malformed cap must raise, not silently resolve to unlimited.

    Fail-open here would disable the one component whose entire job is to cap
    spend - the worst possible place for a silent default.
    """
    text = dedent(
        f"""
        roles:
          judge:
            provider: fake
            model: fake-lite
        budget:
          max_requests_per_day: {bad_value}
        """
    )
    with pytest.raises(RouterError, match="must be a positive integer"):
        LLMRouter.load(_write(tmp_path, text))


def test_config_without_roles_mapping_raises(tmp_path: Path) -> None:
    with pytest.raises(RouterError, match="'roles' mapping"):
        LLMRouter.load(_write(tmp_path, "budget: {}\n"))


# --------------------------------------------------------------------------
# The shipped config must be valid
# --------------------------------------------------------------------------


def test_shipped_models_config_is_present_and_defines_every_role() -> None:
    assert DEFAULT_MODELS_PATH.is_file(), f"expected models config at {DEFAULT_MODELS_PATH}"
    # Build with an explicit key so no environment is required, and never call out.
    router = LLMRouter.load(api_key="unused-in-this-test")
    try:
        for role in Role:
            provider = router.provider_for(role)
            assert provider is not None
    finally:
        router.close()  # release the httpx clients the gemini providers own


def test_shipped_budget_is_below_free_tier(tmp_path: Path) -> None:
    router = LLMRouter.load(api_key="unused-in-this-test")
    assert router.budget.max_requests_per_day is not None
    assert router.budget.max_requests_per_day <= 200
    assert router.budget.max_requests_per_run is not None
