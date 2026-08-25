"""Ranking metrics for the retrieval eval: recall, precision, MRR and nDCG.

WHY THIS MODULE EXISTS
----------------------
:mod:`aegis.retrieval` has four arms - ``bm25``, ``vector``, ``hybrid``,
``hybrid+rerank`` - and the whole point of building them behind one
:class:`~aegis.retrieval.retriever.RetrievalConfig` was that a claim like "hybrid
beats BM25 here" could be *measured* rather than assumed. Measuring it needs
numbers, and ranking numbers are unusually easy to get subtly wrong: a precision
that divides by the wrong denominator, an nDCG whose ideal ranking is computed
over the whole corpus instead of the cutoff, a mean that quietly folds an
undefined query in as a zero. None of those raise. They just move the score.

So the arithmetic lives here, in a few readable lines per metric, pinned in
``tests/evals/test_retrieval_metrics.py`` against examples worked by hand - the
same treatment :mod:`evals.stats.analysis` gives Wilson and McNemar, and for the
same reason: a reviewer must be able to check the number, not trust it.

BINARY RELEVANCE, STATED UP FRONT
---------------------------------
Every metric here assumes **binary** relevance: a chunk is either in the golden
set for a query or it is not, and every relevant chunk counts the same. There are
no graded judgements ("this one is perfect, that one is adequate"), because
grading them would mean inventing a scale, and an invented scale is a thumb on
the results wearing a number. The consequence is worth naming: with binary gains,
nDCG measures only *where* the relevant chunks landed, never *which* of them
landed first.

THE nDCG DISCOUNT
-----------------
Discounted cumulative gain at cutoff ``k``, over a ranking whose ``i``-th result
(1-indexed) has gain ``g_i`` in ``{0, 1}``::

    DCG@k  = sum over i = 1..k of  g_i / log2(i + 1)

The ``log2(i + 1)`` discount is Jarvelin and Kekalainen (2002), "Cumulated
gain-based evaluation of IR techniques", ACM TOIS 20(4):422-446. It is
``log2(i + 1)`` and not ``log2(i)`` so that rank 1 divides by ``log2(2) == 1``
rather than by zero. The ideal DCG is the same sum over the best ranking the
golden set permits **at the same cutoff** - that is, ``min(k, |relevant|)``
relevant chunks packed into the top positions::

    IDCG@k = sum over i = 1..min(k, |relevant|) of  1 / log2(i + 1)
    nDCG@k = DCG@k / IDCG@k

Computing IDCG over ``|relevant|`` instead of ``min(k, |relevant|)`` is the
common bug: a query with 20 relevant chunks evaluated at k=10 could then never
score 1.0, and the arm would be penalised for a cutoff the harness chose.

THE UNDEFINED CONTRACT (the decision this module is opinionated about)
----------------------------------------------------------------------
A query whose golden set is **empty** has nothing to find. Every function here
returns ``None`` for such a query, and :func:`summarize` drops it from the mean
and reports how many it dropped.

That is a decision, so here is the whole of it. Recall is ``0/0`` and nDCG is
``0/0`` - literally undefined, and reporting either as ``0.0`` would state that
the retriever failed at a task that was never set. Precision@k and RR@k are not
undefined: they are arithmetically ``0.0``, because no retrieved chunk can be
relevant when none exists. But that ``0.0`` is a property of the *query*, not of
the *ranking* - every retriever on earth scores it identically - so averaging it
into a table would move the arm's score without any arm having done anything.
Returning ``None`` uniformly keeps one rule instead of four: **a query that
cannot discriminate between two retrievers contributes to no mean.** The honest
place to catch such a query is the golden set itself, and
:func:`evals.retrieval.golden_set.load_golden_set` refuses one outright, so in
this harness the ``None`` path is a guard rather than a routine.

An empty *ranking* is the opposite case and is a real measurement: the retriever
was asked and returned nothing, so recall, precision, RR and nDCG are all ``0.0``
and they count.

k LARGER THAN THE RESULT LIST
-----------------------------
Truncation is by ``retrieved[:k]``, so a short list is simply scored whole.
Precision@k nonetheless divides by ``k``, not by ``len(retrieved)`` - the
convention of ``trec_eval`` and ``ranx``. This is worth saying out loud because
it has a visible consequence: an arm asked for 10 results that can only return 3,
all of them relevant, scores precision@10 of 0.3, not 1.0. That is intended.
Precision@k answers "how good is a page of ten results", and a page with seven
blanks is not a perfect page. Recall@k, MRR@k and nDCG@k are unaffected.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "MetricError",
    "MetricSummary",
    "QueryScore",
    "RetrievalMetric",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank_at_k",
    "score_query",
    "summarize",
]


class MetricError(ValueError):
    """A refusal to compute a ranking metric from input that cannot support one.

    Every raise below is the same kind of refusal - "this input would produce a
    confident number meaning something other than what the caller will read it
    as" - so they share a type, and a caller that wants to report the refusal
    rather than crash catches one exception.
    """


class RetrievalMetric(StrEnum):
    """The metrics an ablation table reports, in the order it reports them.

    A StrEnum rather than bare strings because the column order, the JSON keys
    and the per-metric lookups all have to agree; a typo in one of three string
    literals is a missing column, not an error.
    """

    HIT = "hit@k"
    RECALL = "recall@k"
    PRECISION = "precision@k"
    MRR = "mrr@k"
    NDCG = "ndcg@k"


def _top_k(retrieved: Sequence[str], k: int) -> list[str]:
    """Validate the ranking and return its top-``k`` prefix.

    Duplicates are refused rather than tolerated. A ranking that names the same
    chunk twice would let recall exceed 1.0 and would inflate DCG with a gain the
    corpus cannot supply - and it is a retriever bug, so it should surface as one
    here rather than as a suspiciously good score in a README table.
    """
    if k <= 0:
        raise MetricError(f"k must be positive, got {k}")
    prefix = list(retrieved[:k])
    if len(set(prefix)) != len(prefix):
        raise MetricError(
            "retrieved ranking repeats an id inside the top-k window; that would "
            "push recall above 1.0 and is a retriever bug, not a score"
        )
    return prefix


def recall_at_k(retrieved: Sequence[str], relevant: frozenset[str], k: int) -> float | None:
    """Fraction of the relevant chunks that appear in the top ``k``.

    ``|retrieved[:k] & relevant| / |relevant|``. The denominator is the golden
    set, never ``k``, so recall answers "did we find the answer at all, given a
    budget of k slots" - the question that matters most for RAG, because a chunk
    the retriever never returns is a chunk the generator can never cite.

    Returns ``None`` when ``relevant`` is empty - see the module docstring.
    """
    prefix = _top_k(retrieved, k)
    if not relevant:
        return None
    return sum(1 for chunk_id in prefix if chunk_id in relevant) / len(relevant)


def precision_at_k(retrieved: Sequence[str], relevant: frozenset[str], k: int) -> float | None:
    """Fraction of the top ``k`` slots filled by a relevant chunk.

    ``|retrieved[:k] & relevant| / k`` - divided by ``k`` even when fewer than
    ``k`` results came back. See the module docstring for why, and for what that
    costs a short result list.

    Returns ``None`` when ``relevant`` is empty - see the module docstring.
    """
    prefix = _top_k(retrieved, k)
    if not relevant:
        return None
    return sum(1 for chunk_id in prefix if chunk_id in relevant) / k


def reciprocal_rank_at_k(
    retrieved: Sequence[str],
    relevant: frozenset[str],
    k: int,
) -> float | None:
    """``1 / rank`` of the first relevant chunk in the top ``k``, else ``0.0``.

    Averaged over queries this is MRR, the metric that answers "how far down does
    a reader have to go". It is deliberately cut off at ``k`` rather than run over
    the whole ranking: a relevant chunk at rank 40 of a 10-result page is not
    something anyone sees, and scoring it ``0.025`` instead of ``0`` credits the
    arm for a result it did not deliver.

    Returns ``None`` when ``relevant`` is empty - see the module docstring.
    """
    prefix = _top_k(retrieved, k)
    if not relevant:
        return None
    for rank, chunk_id in enumerate(prefix, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: frozenset[str], k: int) -> float | None:
    """Normalised discounted cumulative gain at ``k``, with binary gains.

    The formula, the ``log2(i + 1)`` discount and the cutoff on the ideal ranking
    are all stated in the module docstring; this function is that arithmetic and
    nothing else. ``math.fsum`` rather than ``sum`` so the value cannot depend on
    the order the terms happen to be added in - the same reproducibility
    precaution :mod:`aegis.retrieval.embedding` takes with its dot products.

    Returns ``None`` when ``relevant`` is empty: IDCG is then ``0`` and the
    quotient is ``0/0``, which is undefined and is not ``0.0``.
    """
    prefix = _top_k(retrieved, k)
    if not relevant:
        return None
    dcg = math.fsum(
        1.0 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(prefix, start=1)
        if chunk_id in relevant
    )
    ideal = math.fsum(1.0 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
    # `ideal` cannot be zero: `relevant` is non-empty and k >= 1, so the sum
    # always contains its rank-1 term, which is exactly 1 / log2(2) == 1.0.
    return dcg / ideal


@dataclass(frozen=True, slots=True)
class QueryScore:
    """Every metric for one query under one arm, plus the counts behind them.

    The counts are fields rather than internals because a score is not readable
    without them: "recall 0.5" over a golden set of two chunks and over one of
    twenty are different results, and a per-query dump that omits ``n_relevant``
    invites the reader to average them as if they were the same.
    """

    query_id: str
    k: int
    n_relevant: int
    n_retrieved: int
    hit: float | None
    recall: float | None
    precision: float | None
    reciprocal_rank: float | None
    ndcg: float | None

    def value(self, metric: RetrievalMetric) -> float | None:
        """Lookup by metric, so the table renderer needs no attribute-name map."""
        lookup: dict[RetrievalMetric, float | None] = {
            RetrievalMetric.HIT: self.hit,
            RetrievalMetric.RECALL: self.recall,
            RetrievalMetric.PRECISION: self.precision,
            RetrievalMetric.MRR: self.reciprocal_rank,
            RetrievalMetric.NDCG: self.ndcg,
        }
        return lookup[metric]


def score_query(
    query_id: str,
    retrieved: Sequence[str],
    relevant: frozenset[str],
    k: int,
) -> QueryScore:
    """Score one ranking against one golden set.

    ``hit`` is recorded as ``0.0``/``1.0`` rather than as a ``bool`` so it
    averages like the others - but it is the only metric here that is a genuine
    Bernoulli trial per query, which is why it is the only one
    :mod:`evals.retrieval.run` puts a Wilson interval around.
    """
    prefix = _top_k(retrieved, k)
    hit: float | None = None
    if relevant:
        hit = 1.0 if any(chunk_id in relevant for chunk_id in prefix) else 0.0
    return QueryScore(
        query_id=query_id,
        k=k,
        n_relevant=len(relevant),
        n_retrieved=len(prefix),
        hit=hit,
        recall=recall_at_k(retrieved, relevant, k),
        precision=precision_at_k(retrieved, relevant, k),
        reciprocal_rank=reciprocal_rank_at_k(retrieved, relevant, k),
        ndcg=ndcg_at_k(retrieved, relevant, k),
    )


@dataclass(frozen=True, slots=True)
class MetricSummary:
    """A macro-average over queries, carrying how many queries it is over.

    ``mean`` is ``float | None`` for the same reason
    :class:`evals.stats.analysis.Interval` makes ``point`` optional: an average
    over zero scored queries is not ``0.0``, it is nothing, and the type forces
    the caller to decide what "we did not measure this" renders as.
    """

    metric: RetrievalMetric
    mean: float | None
    n_scored: int
    n_undefined: int

    @property
    def n_queries(self) -> int:
        """Every query offered to :func:`summarize`, scored or not."""
        return self.n_scored + self.n_undefined


def summarize(metric: RetrievalMetric, values: Iterable[float | None]) -> MetricSummary:
    """Macro-average the defined values and count the ones that were dropped.

    MACRO, NOT MICRO
        Each query contributes equally, whatever the size of its golden set. The
        alternative - pooling every judgement and averaging those - lets one
        query with many relevant chunks outvote five queries with one each, which
        on a hand-built fixture means the fixture's author silently chose the
        weights.

    THE DROPPED COUNT IS NOT OPTIONAL
        It is returned beside the mean so a report can never say "nDCG 0.71"
        without also being able to say "over 18 of 20 queries". Two arms averaged
        over different query subsets are not comparable, and the only way a
        reader can notice that is if both counts are on the page.
    """
    scored: list[float] = []
    n_undefined = 0
    for value in values:
        if value is None:
            n_undefined += 1
        else:
            scored.append(value)
    return MetricSummary(
        metric=metric,
        mean=(math.fsum(scored) / len(scored)) if scored else None,
        n_scored=len(scored),
        n_undefined=n_undefined,
    )
