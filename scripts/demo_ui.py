"""Open the Aegis console - the interactive version of the other two demos.

    uv run python scripts/demo_ui.py

WHERE THIS SITS AMONG THE DEMOS
-------------------------------
``demo_attack.py``     the capability gate deciding one call, in a terminal.
``demo_middleware.py`` the whole runtime driving a hand-written tool loop.
``demo_ui.py``         the same layers, but you supply the poisoned text and the
                       tool call, and watch every tier and reason code change.

The interesting thing the first two cannot do is let you get it WRONG on purpose:
reword the attack until the detector goes quiet, reformat the recipient until the
taint match misses, rename an argument until the policy stops applying to it. The
console ships those three as buttons, because a defense demo that only shows the
defense winning teaches the wrong lesson.

Offline. No API key, no network, no model - L1, L2, L3 and L5 are deterministic
Python. The server is loopback-only and standard library only; see
``src/aegis/console/server.py`` for why both of those are constants rather than
flags.
"""

from __future__ import annotations

import sys

from aegis.console.server import main

if __name__ == "__main__":
    sys.exit(main())
