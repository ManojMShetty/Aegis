"""Extend the baseline and defended arms to a wider couple set, across quota resets.

WHY THIS EXISTS
---------------
The paired comparison at 16 couples produced 2 discordant pairs, and the exact
McNemar test cannot return below p = 0.0625 until there are 6 - so the result is
directionally right and statistically silent. The fix is more couples, and the
constraint is a hard daily token cap that a wider run cannot fit inside.

Those two facts together mean the measurement has to survive being interrupted:
run until the endpoint refuses, wait for the budget to refill, resume from the
task results already on disk, repeat. That is what this does. It is deliberately
dumb - it drives the real runner as a subprocess and owns no measurement logic of
its own, so nothing here can influence a number.

BOTH ARMS MOVE TOGETHER, AND WHY THAT MATTERS
---------------------------------------------
A pair needs the same couple measured under both conditions. Extending only the
arm that happens to have budget left would grow one side of the comparison and
silently drop the extra couples on the floor at analysis time. So each pass tops
up whichever arm is behind, and the useful sample is always the intersection.

    uv run python scripts/extend_arms.py --user-tasks 8 --injection-tasks 4
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from aegis.config.sandbox import scan_environment

# How long to wait after the endpoint reports the daily budget is gone. The cap
# refills on a daily boundary, so a short retry only burns wall-clock; 20 minutes
# is frequent enough to pick the budget up promptly without hammering.
_BACKOFF_SECONDS = 20 * 60

# A run that fails for a reason that is NOT the budget (a bug, a bad flag) will
# fail again immediately, so stop rather than spin.
_MAX_CONSECUTIVE_OTHER_FAILURES = 2

_RATE_LIMIT_MARKERS = ("rate_limit_exceeded", "RateLimitError", "tokens per day")


@dataclass(frozen=True, slots=True)
class Arm:
    """One side of the comparison."""

    label: str
    defense: str
    layers: str | None
    out: Path

    def command(self, user_tasks: int, injection_tasks: int) -> list[str]:
        cmd = [
            sys.executable,
            "-m",
            "evals.agentdojo.runner",
            "--max-tasks",
            str(user_tasks),
            "--max-injection-tasks",
            str(injection_tasks),
            "--resume",
            "--out",
            str(self.out),
        ]
        if self.defense != "none":
            cmd += ["--defense", self.defense]
            if self.layers:
                cmd += ["--defense-layers", self.layers]
        return cmd


def _couples_recorded(out: Path) -> int:
    """How many injected couples the arm's result file currently covers."""
    try:
        payload = json.loads(out.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    security = payload.get("raw", {}).get("injected", {}).get("security_results", {})
    return len(security) if isinstance(security, dict) else 0


def _sandboxed_env() -> dict[str, str]:
    """This process's environment minus every credential the eval must not inherit.

    The runner refuses to start beside a variable shaped like a real tool
    credential (aegis.config.sandbox), and it is right to: a benchmark run drives
    an agent through attack payloads, so any unrelated secret sitting in its
    environment is one tool call away from an exfiltration target. In practice the
    ambient shell carries such variables for reasons that have nothing to do with
    this project - the harness that launched it, an unrelated CLI, a developer's
    own tokens.

    The wrong response is AEGIS_ALLOW_REAL_CREDENTIALS=true, which switches the
    guard off wholesale and would let a genuine mistake through for the rest of the
    run. The right one is to hand the subprocess an environment that has nothing to
    find, which is strictly safer than the guard passing.

    The decision of what counts is delegated to `scan_environment`, so this cannot
    drift from what the runner enforces: whatever the guard would refuse, we remove.
    """
    env = dict(os.environ)
    for finding in scan_environment(env):
        env.pop(finding.name, None)
    return env


def _run(arm: Arm, user_tasks: int, injection_tasks: int) -> tuple[bool, str]:
    """Drive one arm once. Returns (budget_exhausted, tail_of_output)."""
    # Fixed argv, no shell: nothing here is built from anything the model produced.
    proc = subprocess.run(
        arm.command(user_tasks, injection_tasks),
        capture_output=True,
        text=True,
        check=False,
        env=_sandboxed_env(),
    )
    combined = f"{proc.stdout}\n{proc.stderr}"
    exhausted = any(marker in combined for marker in _RATE_LIMIT_MARKERS)
    if proc.returncode == 0:
        return False, "ok"
    tail = "\n".join(line for line in combined.splitlines() if line.strip())[-400:]
    return exhausted, tail


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-tasks", type=int, default=8)
    parser.add_argument("--injection-tasks", type=int, default=4)
    parser.add_argument(
        "--deadline-hours",
        type=float,
        default=6.0,
        help="stop starting new passes after this many hours",
    )
    args = parser.parse_args(argv)

    target = args.user_tasks * args.injection_tasks
    arms = (
        Arm("baseline", "none", None, Path("results/week0_baseline_wide.json")),
        Arm("defended", "aegis", "all", Path("results/week0_defended_wide.json")),
    )

    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY is not set in the environment")
        return 1

    deadline = time.monotonic() + args.deadline_hours * 3600
    other_failures = 0

    def log(message: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

    log(f"target {target} couples per arm ({args.user_tasks} x {args.injection_tasks})")

    while time.monotonic() < deadline:
        behind = [a for a in arms if _couples_recorded(a.out) < target]
        if not behind:
            log("both arms complete")
            break

        # Top up whichever arm is furthest behind, so the two never drift apart by
        # more than one pass and the pairable intersection keeps growing.
        arm = min(behind, key=lambda a: _couples_recorded(a.out))
        have = _couples_recorded(arm.out)
        log(f"{arm.label}: {have}/{target} couples - running")

        exhausted, detail = _run(arm, args.user_tasks, args.injection_tasks)
        now = _couples_recorded(arm.out)

        if detail == "ok":
            other_failures = 0
            log(f"{arm.label}: now {now}/{target} couples")
            continue

        if exhausted:
            other_failures = 0
            log(f"{arm.label}: daily budget gone at {now}/{target}; waiting for the reset")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(_BACKOFF_SECONDS, remaining))
            continue

        other_failures += 1
        log(f"{arm.label}: failed for a non-budget reason ({other_failures}):\n{detail}")
        if other_failures >= _MAX_CONSECUTIVE_OTHER_FAILURES:
            log("stopping: this is not going to fix itself by waiting")
            return 1

    for arm in arms:
        log(f"final {arm.label}: {_couples_recorded(arm.out)}/{target} couples -> {arm.out}")
    print(
        "\nCompare with:\n  uv run python -m evals.stats.analysis "
        f"--baseline {arms[0].out} --defended {arms[1].out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
