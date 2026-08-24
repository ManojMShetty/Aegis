"""Statistics for the eval harness - the reason the reported numbers are honest.

WHY A PACKAGE FOR THIS
----------------------
:mod:`evals.agentdojo` measures; this package says how much the measurement is
worth. The Week-0 baseline is 3 user tasks and 12 (user, injection) couples, and
at that size a bare percentage is not a result - two runs of the identical
configuration scored utility 100% and then 66.7%. So every rate this project
publishes carries a Wilson interval, and every "the defense improved X" claim
carries a paired test on the same tasks plus a bootstrap interval on the
difference.

WHY IT IS SEPARATE FROM THE RUNNER
----------------------------------
The runner makes network calls, spends quota, and must not be edited while a live
benchmark is in flight. The analysis is pure arithmetic over a JSON file it did
not write. Keeping them apart means the statistics can be developed, tested and
re-run offline against a recorded result forever, and that a change to how a
number is REPORTED can never change what was MEASURED.

WHY THE MATHS IS HAND-WRITTEN
-----------------------------
``scipy`` and ``statsmodels`` are declared in the ``evals`` extra but are not
installed here, and this module deliberately does not need them: a reviewer can
read the forty-line Wilson interval and the twenty-line exact McNemar test and
check them against a published table. That is worth more to a portfolio project
than a one-line import of a black box.

WHY THIS FILE RE-EXPORTS NOTHING
--------------------------------
Unlike :mod:`evals.agentdojo`, which unifies two provider modules behind one
namespace, this package holds a single module - and that module is also the CLI
entry point (``python -m evals.stats.analysis``). Importing it here would load it
twice under two names, which Python warns about at every invocation of the
documented command, in exchange for an alias layer nothing needs. Import from the
module directly::

    from evals.stats.analysis import summarize_run, wilson_interval

See :mod:`evals.stats.analysis` for the mathematics and the reporting contract.
"""

from __future__ import annotations
