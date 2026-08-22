"""The frugality guard - counters, caps, day rollover, and persistence.

These are the tests that make "we will not blow the free-tier quota" a checked
property rather than a hope, so they lean on the fail-closed edges: a cap is
reached *before* the offending call, and a stale on-disk day resets to zero.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from aegis.llm.base import LLMResponse
from aegis.llm.budget import Budget, BudgetExceeded


def _resp(*, input_tokens: int = 10, output_tokens: int = 5) -> LLMResponse:
    return LLMResponse(
        text="ok",
        model="fake",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        finish_reason="STOP",
        raw={},
    )


def _budget(tmp_path: Path, **kwargs: int | None) -> Budget:
    return Budget(path=tmp_path / "budget.json", **kwargs)


# --------------------------------------------------------------------------
# Counters
# --------------------------------------------------------------------------


def test_counters_start_at_zero(tmp_path: Path) -> None:
    b = _budget(tmp_path)
    assert b.request_count == 0
    assert b.input_tokens == 0
    assert b.output_tokens == 0
    assert b.run_request_count == 0


def test_record_accumulates_requests_and_tokens(tmp_path: Path) -> None:
    b = _budget(tmp_path)
    b.record(_resp(input_tokens=10, output_tokens=5))
    b.record(_resp(input_tokens=3, output_tokens=7))
    assert b.request_count == 2
    assert b.run_request_count == 2
    assert b.input_tokens == 13
    assert b.output_tokens == 12


def test_record_request_counts_a_zero_token_request(tmp_path: Path) -> None:
    """A refusal or a retried 429 spent quota but returned no usable tokens.

    The provider calls this per server-reaching attempt so those still count.
    """
    b = _budget(tmp_path)
    b.record_request()
    b.record_request(input_tokens=4)
    assert b.request_count == 2
    assert b.input_tokens == 4
    assert b.output_tokens == 0


def test_check_does_not_count(tmp_path: Path) -> None:
    """Only record() moves counters; check() is a pure guard."""
    b = _budget(tmp_path, max_requests_per_day=5)
    b.check()
    b.check()
    assert b.request_count == 0


# --------------------------------------------------------------------------
# Caps fail closed
# --------------------------------------------------------------------------


def test_per_run_cap_raises_before_the_offending_call(tmp_path: Path) -> None:
    b = _budget(tmp_path, max_requests_per_run=2)
    b.check()
    b.record(_resp())
    b.check()
    b.record(_resp())
    # The third call would breach the run cap - refuse before spending it.
    with pytest.raises(BudgetExceeded, match="per-run"):
        b.check()
    assert b.run_request_count == 2


def test_per_day_cap_raises_before_the_offending_call(tmp_path: Path) -> None:
    b = _budget(tmp_path, max_requests_per_day=1)
    b.check()
    b.record(_resp())
    with pytest.raises(BudgetExceeded, match="per-day"):
        b.check()
    assert b.request_count == 1


def test_no_caps_never_raises(tmp_path: Path) -> None:
    b = _budget(tmp_path)
    for _ in range(50):
        b.check()
        b.record(_resp())
    assert b.request_count == 50


def test_per_day_cap_is_enforced_across_processes(tmp_path: Path) -> None:
    """A second Budget over the same file inherits today's spent count."""
    path = tmp_path / "budget.json"
    first = Budget(path=path, max_requests_per_day=2)
    first.check()
    first.record(_resp())

    second = Budget(path=path, max_requests_per_day=2)
    assert second.request_count == 1
    second.check()
    second.record(_resp())

    third = Budget(path=path, max_requests_per_day=2)
    with pytest.raises(BudgetExceeded, match="per-day"):
        third.check()


def test_per_run_cap_is_fresh_each_process(tmp_path: Path) -> None:
    """The run counter is process-local even though the day counter persists."""
    path = tmp_path / "budget.json"
    first = Budget(path=path, max_requests_per_run=1)
    first.check()
    first.record(_resp())
    with pytest.raises(BudgetExceeded):
        first.check()

    # A new process starts the run counter over, so one more call is allowed.
    second = Budget(path=path, max_requests_per_run=1)
    assert second.run_request_count == 0
    second.check()  # does not raise


# --------------------------------------------------------------------------
# Day rollover
# --------------------------------------------------------------------------


def test_stale_day_resets_daily_counters_on_load(tmp_path: Path) -> None:
    path = tmp_path / "budget.json"
    yesterday = (datetime.now(UTC).date() - timedelta(days=1)).isoformat()
    path.write_text(
        json.dumps(
            {
                "date": yesterday,
                "request_count": 199,
                "input_tokens": 5000,
                "output_tokens": 4000,
            }
        ),
        encoding="utf-8",
    )

    b = Budget(path=path, max_requests_per_day=200)
    assert b.date == datetime.now(UTC).date()
    assert b.request_count == 0
    assert b.input_tokens == 0
    assert b.output_tokens == 0
    b.check()  # yesterday's near-exhausted count must not block today


def test_same_day_counters_are_preserved_on_load(tmp_path: Path) -> None:
    path = tmp_path / "budget.json"
    today = datetime.now(UTC).date().isoformat()
    path.write_text(
        json.dumps({"date": today, "request_count": 3, "input_tokens": 30, "output_tokens": 12}),
        encoding="utf-8",
    )
    b = Budget(path=path)
    assert b.request_count == 3
    assert b.input_tokens == 30
    assert b.output_tokens == 12


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def test_persistence_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "budget.json"  # parent created on write
    b = Budget(path=path)
    b.record(_resp(input_tokens=11, output_tokens=22))

    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["date"] == datetime.now(UTC).date().isoformat()
    assert on_disk["request_count"] == 1
    assert on_disk["input_tokens"] == 11
    assert on_disk["output_tokens"] == 22

    reloaded = Budget(path=path)
    assert reloaded.request_count == 1
    assert reloaded.input_tokens == 11
    assert reloaded.output_tokens == 22


def test_corrupt_state_file_is_treated_as_a_fresh_day(tmp_path: Path) -> None:
    path = tmp_path / "budget.json"
    path.write_text("{ not valid json", encoding="utf-8")
    b = Budget(path=path)
    assert b.request_count == 0
    b.check()  # must not raise on recovery
