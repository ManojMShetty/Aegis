"""Load `.env` from the repository root, without a dependency and without surprises.

WHY THIS EXISTS
---------------
Two error messages in this project tell an operator their key "lives in .env,
which is gitignored", and `.env.example` says "copy to `.env` and fill in". None
of that was true: nothing read the file. A newcomer who did exactly what the
repository told them to got a missing-key error naming the file they had just
filled in, which is a worse first impression than having no `.env.example` at all.

Rather than delete the instructions, this makes them true.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
* **Never overrides an exported variable.** An explicit `export` in the shell is
  a stronger statement of intent than a file checked out days ago, and a run that
  silently used the file's value while the operator was looking at their shell
  would be the kind of provenance confusion this project spends most of its
  effort eliminating.
* **Never loads from anywhere but the repository root.** Not the current working
  directory, not a parent walk. A `.env` found by searching upward is a file the
  operator did not necessarily mean to give to this process.
* **Never logs a value.** Only the count and the names, and only when asked.
* **Is not automatic.** It is called from the CLI entry points, not from module
  import, so importing `aegis` never mutates the environment of a host process
  that embeds it.

Parsing is deliberately minimal: `KEY=value`, `#` comments, optional `export`
prefix, and surrounding quotes stripped. It is not a shell, and a line it cannot
parse is skipped rather than guessed at.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["DEFAULT_ENV_PATH", "load_dotenv"]

# The repository root: src/aegis/config/dotenv.py -> up four.
DEFAULT_ENV_PATH = Path(__file__).resolve().parents[3] / ".env"


def _strip_quotes(value: str) -> str:
    """Remove one matching pair of surrounding quotes, if present."""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def load_dotenv(path: Path | None = None) -> tuple[str, ...]:
    """Set variables from ``path`` that are not already in the environment.

    Returns the NAMES that were set, in file order, so a caller can report what it
    picked up. Returns an empty tuple when the file is absent - a missing `.env`
    is the normal case for someone who exports in their shell, and is not an error.
    """
    env_path = DEFAULT_ENV_PATH if path is None else path
    try:
        text = env_path.read_text(encoding="utf-8")
    except OSError:
        return ()

    applied: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        name, separator, value = line.partition("=")
        if not separator:
            continue
        name = name.strip()
        if not name or name in os.environ:
            # Already exported: the shell wins, always.
            continue
        os.environ[name] = _strip_quotes(value.strip())
        applied.append(name)
    return tuple(applied)
