"""External benchmarks - where Aegis is graded by someone else's rubric.

WHY THIS LIVES OUTSIDE ``src/aegis``
------------------------------------
Everything under ``src/aegis`` is the system under test; everything here is the
test bench that scores it. Keeping the two apart is not tidiness for its own
sake - it is the same principle the security core is built on. A benchmark that
imported freely from the product, or vice versa, would let the grader and the
graded share state, and the number it produced would stop meaning "how does
Aegis do against an external standard" and start meaning "how does Aegis do
against itself". The eval code may read the public Aegis API, never its internals,
and the product never imports the eval harness at all.

The first bench is AgentDojo (:mod:`evals.agentdojo`): a published prompt-injection
suite we did not write, run against an *undefended* agent so the Week-0 baseline
is an honest floor. The Aegis defense is deliberately not wired in yet - the
baseline has to be the number an attacker gets with nothing standing in the way.
"""

from __future__ import annotations
