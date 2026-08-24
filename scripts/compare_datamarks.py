"""Which datamark character does the same job for the fewest tokens?

WHY THIS MATTERS
----------------
Datamarking (Microsoft's spotlighting) replaces whitespace inside untrusted text
with a marker character, so the model can see exactly where the untrusted span
starts and stops even if the content tries to fake a boundary. The DEFAULT marker
here is U+E000, a Unicode private-use codepoint - chosen because no legitimate
document contains it, which is a real security property.

But a codepoint no tokenizer has ever seen is a codepoint the tokenizer has no
merge rule for, so it fragments into several tokens where the single space it
replaced cost one. Measured on this suite's own tool outputs that is a +68% token
bill on every tool result, re-paid on every later turn of the conversation - which
turns a one-day defended run into a two-day one on a capped free tier.

This measures real candidates against the real recorded tool outputs so the choice
is made on data, not on which character looks most exotic.

    uv run --with tiktoken python scripts/compare_datamarks.py
"""

from __future__ import annotations

import json
import pathlib
import sys

from aegis.domain.trust import Tainted
from aegis.security.spotlight import Spotlighter, SpotlightStyle

LOGDIR = pathlib.Path("results/raw/agentdojo_logs")

# Candidates, with the tradeoff each one makes.
CANDIDATES: dict[str, str] = {
    "\ue000": "U+E000 private use (current default) - unforgeable, unknown to the tokenizer",
    "^": "caret - the marker used in the spotlighting paper; one common token",
    "|": "pipe - one common token, but occurs in ordinary text and tables",
    "\u00b7": "U+00B7 middle dot - obvious separator, known codepoint",
    "\u2588": "U+2588 full block - rare in prose, known codepoint",
}


def _tool_outputs(messages: object) -> list[str]:
    out: list[str] = []
    if not isinstance(messages, list):
        return out
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            out.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    for key in ("content", "text"):
                        if isinstance(block.get(key), str):
                            out.append(block[key])
                            break
    return out


def main() -> int:
    try:
        import tiktoken

        enc = tiktoken.get_encoding("o200k_base")
    except Exception:
        print("needs tiktoken: uv run --with tiktoken python scripts/compare_datamarks.py")
        return 1

    samples: list[str] = []
    for path in LOGDIR.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        messages = payload.get("messages") if isinstance(payload, dict) else payload
        samples.extend(_tool_outputs(messages))

    if not samples:
        print(f"no tool outputs found under {LOGDIR}")
        return 1

    raw_tokens = sum(len(enc.encode(s, disallowed_special=())) for s in samples)
    print(f"tool outputs: {len(samples)}   baseline tokens: {raw_tokens:,}")
    print()
    print(f"{'marker':<10} {'tokens':>10} {'vs raw':>9}   note")
    print("-" * 92)

    for mark, note in CANDIDATES.items():
        spot = Spotlighter(style=SpotlightStyle.DATAMARK, datamark=mark)
        total = 0
        for s in samples:
            tainted = Tainted.untrusted(s, source_uri="agentdojo://tool-output")
            total += len(enc.encode(spot.wrap(tainted).text, disallowed_special=()))
        ratio = total / raw_tokens
        label = ascii(mark)
        print(f"{label:<10} {total:>10,} {ratio:>8.3f}x   {note}")

    print()
    print(
        "A cheaper marker is only an improvement if it keeps the security property:\n"
        "the model must still be able to tell marked text from unmarked, and content\n"
        "must not be able to forge the boundary. A marker that occurs naturally in tool\n"
        "output (pipe, caret in code) weakens that; the security tests are the arbiter."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
