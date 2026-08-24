"""Estimate what the DEFENDED arm will cost, from the undefended run's own logs.

WHY THIS EXISTS
---------------
The defended arm has to cover exactly the same (user task x injection task)
couples as the baseline, or the paired McNemar comparison has nothing to pair.
But it will not cost the same: spotlighting rewrites every tool output into an
explicitly-marked data block, which adds characters to the model's INPUT on every
subsequent turn of that conversation. With a hard daily token cap, "does the
defended arm fit in one day" is a budgeting question that decides whether the
comparison lands tomorrow or in three days - and guessing it wrong wastes a full
day of quota discovering the answer.

Rather than guess a multiplier, this measures the real tool outputs AgentDojo
already recorded in the undefended run and re-spotlights them, so the inflation
figure comes from this suite's actual data.

    uv run python scripts/estimate_defended_cost.py
"""

from __future__ import annotations

import json
import pathlib
import sys

from aegis.domain.trust import Tainted
from aegis.security.spotlight import Spotlighter, SpotlightStyle, guidance_for_style

# The undefended run's logs. Each file is one task run: a list of chat messages.
LOGDIR = pathlib.Path("results/raw/agentdojo_logs")

# Rough characters-per-token for English + JSON-ish tool payloads. Only used to
# turn a character ratio into a token estimate; the RATIO is the measured part.
CHARS_PER_TOKEN = 3.6

# How a defended run's pipeline directory announces itself. The estimate is of
# what defending WOULD cost, so it has to be taken from undefended transcripts.
_DEFENDED_MARKER = "-aegis-"


def _tool_outputs(messages: object) -> list[str]:
    """Pull the text of every tool-result message out of one recorded task run."""
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
                if isinstance(block, dict) and isinstance(block.get("content"), str):
                    out.append(block["content"])
                elif isinstance(block, dict) and isinstance(block.get("text"), str):
                    out.append(block["text"])
    return out


def main() -> int:
    if not LOGDIR.is_dir():
        print(f"no logs at {LOGDIR} - run the baseline first")
        return 1

    spotlighter = Spotlighter(style=SpotlightStyle.DATAMARK)

    runs = 0
    raw_chars = 0
    marked_chars = 0
    outputs = 0
    samples: list[str] = []
    # One logical task run can exist under SEVERAL pipeline directories: every time
    # a fingerprint input changes, the previous directory is orphaned and its logs
    # are copied to the new name rather than re-measured (see results/README.md).
    # Walking the whole tree would then count the same measurement two or three
    # times and quietly bias the ratio, so key on the path BELOW the pipeline
    # directory - suite/user_task/attack/injection.json - which is the identity of
    # the task run itself.
    seen: set[str] = set()
    for path in sorted(LOGDIR.rglob("*.json")):
        try:
            relative = path.relative_to(LOGDIR).as_posix()
        except ValueError:
            continue
        pipeline, _, rest = relative.partition("/")
        # Measure UNDEFENDED runs only. A defended run's logs are already
        # spotlighted, so re-spotlighting them would mark marked text and report a
        # cost nobody pays. The pipeline name carries the defense identity, which
        # is what makes the two separable at all.
        if _DEFENDED_MARKER in pipeline:
            continue
        identity = rest or relative
        if identity in seen:
            continue
        seen.add(identity)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        messages = payload.get("messages") if isinstance(payload, dict) else payload
        texts = _tool_outputs(messages)
        if not texts:
            continue
        runs += 1
        for text in texts:
            outputs += 1
            samples.append(text)
            raw_chars += len(text)
            # The guard spotlights a TAINTED tool output, so measure the same
            # thing it will: wrap() takes the Tainted value, not a bare string.
            tainted = Tainted.untrusted(text, source_uri="agentdojo://tool-output")
            marked_chars += len(spotlighter.wrap(tainted).text)

    if outputs == 0:
        print("found no tool-result messages in the logs - nothing to measure")
        return 1

    ratio = marked_chars / raw_chars
    print(f"task runs inspected        : {runs}")
    print(f"tool outputs measured      : {outputs}")
    print(f"raw tool-output chars      : {raw_chars:,}")
    print(f"spotlighted chars          : {marked_chars:,}")
    print(f"char inflation             : {ratio:.3f}x  ({(ratio - 1) * 100:+.1f}%)")
    print()

    # Characters are the WRONG unit to bill in. Datamarking replaces whitespace
    # runs with U+E000, a private-use codepoint: it collapses several characters
    # into one, but a rare codepoint can cost MORE tokens than the whitespace it
    # replaced. Measure tokens directly rather than trusting the char proxy.
    try:
        import tiktoken

        enc = tiktoken.get_encoding("o200k_base")
    except Exception:
        print("tiktoken unavailable - rerun with: uv run --with tiktoken python <this>")
        return 0

    raw_tokens = 0
    marked_tokens = 0
    for text in samples:
        raw_tokens += len(enc.encode(text, disallowed_special=()))
        tainted = Tainted.untrusted(text, source_uri="agentdojo://tool-output")
        marked_tokens += len(enc.encode(spotlighter.wrap(tainted).text, disallowed_special=()))

    token_ratio = marked_tokens / raw_tokens if raw_tokens else 0.0
    print(f"raw tool-output tokens     : {raw_tokens:,}")
    print(f"spotlighted tokens         : {marked_tokens:,}")
    print(f"TOKEN inflation            : {token_ratio:.3f}x  ({(token_ratio - 1) * 100:+.1f}%)")
    print()

    # The other half of the defended arm's bill, and easy to forget because it is
    # not in the tool output at all: L2 also appends the marker convention to the
    # SYSTEM message, which is re-sent on every turn of every defended
    # conversation. Small per turn, but the only question this script exists to
    # answer is whether the arm fits inside a day of quota.
    guidance_tokens = len(
        enc.encode(guidance_for_style(SpotlightStyle.DATAMARK), disallowed_special=())
    )
    print(f"L2 system-message guidance : {guidance_tokens:,} tokens, per TURN")
    print()
    print(
        "A tool output is re-sent on every LATER turn of its conversation, so the token\n"
        "delta above is paid once per remaining turn, not once per run. Treat it as a\n"
        "per-turn surcharge on the defended arm's input."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
