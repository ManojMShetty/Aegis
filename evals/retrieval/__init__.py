"""Retrieval evaluation - whether the four arms in ``aegis.retrieval`` differ, and how.

WHY A PACKAGE FOR THIS
----------------------
:mod:`aegis.retrieval` deliberately exposes ``bm25`` / ``vector`` / ``hybrid`` /
``hybrid+rerank`` as four values of one
:class:`~aegis.retrieval.retriever.RetrievalConfig` rather than four code paths,
on the argument that "hybrid is the biggest quality win in a RAG pipeline" is a
claim to be measured and not repeated. This package is where that claim gets a
number - or, as it turns out on the committed fixture, gets a smaller number than
the folklore promises. Either way the table is produced by one command and its
inputs are committed beside it.

WHY IT SITS BESIDE ``evals.stats`` AND NOT INSIDE ``src``
---------------------------------------------------------
Same rule as :mod:`evals.agentdojo`: the thing being graded and the thing doing
the grading do not share a package. Nothing under ``src/aegis`` imports anything
here, and this package reads only the public retrieval and ingest API - the same
one an application would call. A metric that reached into an index's internals
would be measuring an implementation rather than a retriever.

WHAT IS HERE
------------
* :mod:`evals.retrieval.metrics` - recall@k, precision@k, MRR@k and nDCG@k, with
  binary relevance and an explicit contract for the undefined cases. Hand-written
  and hand-pinned, for the reason :mod:`evals.stats.analysis` gives: a reviewer
  must be able to check the arithmetic.
* :mod:`evals.retrieval.golden_set` - the format, the loader, and the committed
  fixture ``golden_set.json``. Its docstring is blunt about the fixture being
  ours rather than an external benchmark.
* :mod:`evals.retrieval.run` - the CLI: one corpus, four arms, one table.

WHY THIS FILE RE-EXPORTS NOTHING
--------------------------------
:mod:`evals.retrieval.run` is both a module and the documented entry point
(``python -m evals.retrieval.run``). Importing it here would load it twice under
two names, which Python warns about on every invocation of the documented
command. Import from the modules directly::

    from evals.retrieval.metrics import ndcg_at_k
    from evals.retrieval.golden_set import load_golden_set
"""

from __future__ import annotations
