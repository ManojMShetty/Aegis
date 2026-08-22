"""The frugality guard - a hard, fail-closed cap on LLM spend.

WHY THIS EXISTS AT ALL
----------------------
Aegis is built to run on a *free* Gemini tier: no credit card, a fixed number of
requests per day. That constraint is a feature (anyone can reproduce the eval),
but it has a sharp edge - the daily quota is shared across every process on the
key, and a runaway eval loop can burn a whole day's allowance in seconds. Worse,
blowing the cap does not fail cleanly: you get 429s halfway through a run and a
half-written results file.

So the budget is a *pre-flight* check, not a post-mortem meter. :meth:`check`
runs BEFORE a call and refuses when the next request would exceed a cap. That is
deliberately fail-closed: the cost of a false stop is "run it again tomorrow",
while the cost of a false go is a blown quota and a broken run.

WHY IT PERSISTS TO DISK
-----------------------
An eval is many short-lived processes over a day. An in-memory counter would
reset every time and enforce nothing. The daily counters therefore live in a
small JSON file keyed by UTC date; on load, a stale date rolls the daily
counters back to zero. The per-*run* counter, by contrast, is intentionally
process-local (a fresh cap for each invocation) and is never persisted.

The file is pure runtime state and must not be committed; the default path lives
under ``results/raw/`` (already gitignored). No network, no imports beyond the
standard library - this stays trivially unit-testable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from aegis.llm.base import LLMResponse

__all__ = ["DEFAULT_BUDGET_PATH", "Budget", "BudgetExceeded"]

# src/aegis/llm/budget.py -> parents[3] is the repo root, matching the pattern in
# aegis.config.policy. results/raw/ is gitignored, so this stays uncommitted.
DEFAULT_BUDGET_PATH = Path(__file__).resolve().parents[3] / "results" / "raw" / "llm_budget.json"


class BudgetExceeded(Exception):
    """Raised by :meth:`Budget.check` when the next call would breach a cap.

    Fail-closed by design: it is thrown *before* the request is issued, so the
    underlying quota is never actually spent past the limit.
    """


@dataclass(slots=True)
class _DailyCounters:
    """The persisted, per-UTC-day tally."""

    day: date
    request_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class Budget:
    """Tracks LLM usage and enforces per-day and per-run request caps.

    Instances are cheap and safe to share: the router hands one ``Budget`` to
    every provider it builds, so the caps span all roles on the key.
    """

    def __init__(
        self,
        *,
        max_requests_per_day: int | None = None,
        max_requests_per_run: int | None = None,
        path: Path | str | None = None,
    ) -> None:
        self._max_per_day = max_requests_per_day
        self._max_per_run = max_requests_per_run
        self._path = Path(path) if path is not None else DEFAULT_BUDGET_PATH
        self._run_count = 0
        self._daily = self._load()

    # -- queries ---------------------------------------------------------

    @property
    def path(self) -> Path:
        return self._path

    @property
    def date(self) -> date:
        """The UTC day the persisted counters currently belong to."""
        return self._daily.day

    @property
    def request_count(self) -> int:
        """Requests recorded today (UTC), across every process on this file."""
        return self._daily.request_count

    @property
    def input_tokens(self) -> int:
        return self._daily.input_tokens

    @property
    def output_tokens(self) -> int:
        return self._daily.output_tokens

    @property
    def run_request_count(self) -> int:
        """Requests recorded by *this* process since construction."""
        return self._run_count

    @property
    def max_requests_per_day(self) -> int | None:
        return self._max_per_day

    @property
    def max_requests_per_run(self) -> int | None:
        return self._max_per_run

    # -- enforcement -----------------------------------------------------

    def check(self) -> None:
        """Raise :class:`BudgetExceeded` if the next request would breach a cap.

        Rolls the day over first, so a process that crosses UTC midnight sees the
        fresh allowance rather than yesterday's exhausted one.
        """
        self._roll_if_new_day()
        if self._max_per_run is not None and self._run_count >= self._max_per_run:
            raise BudgetExceeded(
                f"per-run request cap reached ({self._run_count}/{self._max_per_run}); "
                "not issuing another LLM call this run"
            )
        if self._max_per_day is not None and self._daily.request_count >= self._max_per_day:
            raise BudgetExceeded(
                f"per-day request cap reached ({self._daily.request_count}/{self._max_per_day}) "
                f"for {self._daily.day.isoformat()} UTC; try again after the daily reset"
            )

    def record(self, resp: LLMResponse) -> None:
        """Account for one successful call and persist the new daily totals."""
        self.record_request(input_tokens=resp.input_tokens, output_tokens=resp.output_tokens)

    def record_request(self, *, input_tokens: int = 0, output_tokens: int = 0) -> None:
        """Account for one request that reached the server, and persist.

        Counts the request itself even when it yielded no usable output: a safety
        refusal or a retried 429 is a real HTTP round-trip that still drew on the
        daily quota this guard exists to protect. Providers call this for *every*
        attempt that reached the server; :meth:`record` is the success-path
        convenience that also adds the returned token counts.

        NOTE: the read-modify-write is not cross-process locked. The atomic
        write-then-replace prevents a corrupt file, but two *concurrent*
        processes can still lose an update (both read N, both write N+1). The
        design is sequential, short-lived processes; add file locking before
        running concurrent eval workers against one key.
        """
        self._roll_if_new_day()
        self._run_count += 1
        self._daily.request_count += 1
        self._daily.input_tokens += input_tokens
        self._daily.output_tokens += output_tokens
        self._persist()

    # -- persistence -----------------------------------------------------

    def _load(self) -> _DailyCounters:
        today = datetime.now(UTC).date()
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # Missing or corrupt state file: treat as a fresh day. The file is
            # our own runtime scratch, so this is recovery, not a security event.
            return _DailyCounters(day=today)

        stored = _parse_date(raw.get("date") if isinstance(raw, dict) else None)
        if stored != today:
            # Date rollover (or unreadable date): start the day clean.
            return _DailyCounters(day=today)
        return _DailyCounters(
            day=today,
            request_count=_as_int(raw.get("request_count")),
            input_tokens=_as_int(raw.get("input_tokens")),
            output_tokens=_as_int(raw.get("output_tokens")),
        )

    def _roll_if_new_day(self) -> None:
        today = datetime.now(UTC).date()
        if self._daily.day != today:
            self._daily = _DailyCounters(day=today)

    def _persist(self) -> None:
        payload = {
            "date": self._daily.day.isoformat(),
            "request_count": self._daily.request_count,
            "input_tokens": self._daily.input_tokens,
            "output_tokens": self._daily.output_tokens,
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-replace so a crash mid-write cannot corrupt the counters.
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self._path)


def _parse_date(value: object) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _as_int(value: object) -> int:
    # bool is an int subclass; exclude it so a stray "true" cannot become 1.
    return value if isinstance(value, int) and not isinstance(value, bool) else 0
