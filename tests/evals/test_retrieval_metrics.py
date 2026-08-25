"""Tests for evals.retrieval.metrics - every metric pinned to a worked example.

WHY THESE TESTS LOOK LIKE THIS
------------------------------
A ranking metric can be wrong in a way that never raises. Divide precision by
``len(retrieved)`` instead of ``k``, compute the ideal DCG over ``|relevant|``
instead of ``min(k, |relevant|)``, let an undefined query average in as a zero -
each of those returns a plausible float in ``[0, 1]``, and a test that only
asserted "returns a float in [0, 1]" would pass against all three while the
README table moved.

So this file does what ``tests/evals/test_stats.py`` does for Wilson and McNemar:
every value is pinned against an example computed by hand, and the arithmetic is
written out in the test so a reviewer checks a number rather than trusting one.

THE RUNNING EXAMPLE
    ranking  ``[a, b, c, d, e]``, relevant ``{b, d}``, cutoff ``k = 5``. The two
    relevant chunks sit at ranks 2 and 4, so::

        recall@5    = 2/2                              = 1.0
        precision@5 = 2/5                              = 0.4
        RR@5        = 1/2                              = 0.5
        DCG@5       = 1/log2(3) + 1/log2(5)            = 1.0616063117...
        IDCG@5      = 1/log2(2) + 1/log2(3)            = 1.6309297535...
        nDCG@5      = DCG/IDCG                         = 0.6509209298...

    Chosen because it separates all four metrics: a ranking where the relevant
    chunks are first would score 1.0 on three of them and hide a discount bug.

The rest of the file is the contracts the module docstring commits to - the
undefined cases, the ``k`` larger than the result list, the IDCG cutoff - each
written as the failure it is there to prevent.
"""

from __future__ import annotations

import math

import pytest
from evals.retrieval.metrics import (
    MetricError,
    RetrievalMetric,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
    score_query,
    summarize,
)

RANKING = ["a", "b", "c", "d", "e"]
RELEVANT = frozenset({"b", "d"})


# ---------------------------------------------------------------------------
# The worked example
# ---------------------------------------------------------------------------


def test_recall_at_k_is_found_over_relevant() -> None:
    """Both relevant chunks are inside the top 5, so recall is 2/2."""
    assert recall_at_k(RANKING, RELEVANT, 5) == pytest.approx(1.0)
    # At k=3 only "b" is inside the window: 1 of the 2 relevant chunks.
    assert recall_at_k(RANKING, RELEVANT, 3) == pytest.approx(0.5)


def test_precision_at_k_divides_by_k_not_by_the_relevant_count() -> None:
    """2 relevant chunks in 5 slots is 0.4, and in 3 slots one chunk is 1/3.

    The distinction this pins: precision's denominator is the window, recall's is
    the golden set. Swapping them is invisible whenever ``k == |relevant|``.
    """
    assert precision_at_k(RANKING, RELEVANT, 5) == pytest.approx(0.4)
    assert precision_at_k(RANKING, RELEVANT, 3) == pytest.approx(1.0 / 3.0)


def test_reciprocal_rank_is_one_over_the_first_relevant_rank() -> None:
    """The first relevant chunk is at rank 2, so RR is 1/2 exactly."""
    assert reciprocal_rank_at_k(RANKING, RELEVANT, 5) == pytest.approx(0.5)


def test_ndcg_matches_the_hand_computed_value() -> None:
    """nDCG@5 of the running example, against the arithmetic spelled out here."""
    dcg = 1.0 / math.log2(3) + 1.0 / math.log2(5)
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    assert ndcg_at_k(RANKING, RELEVANT, 5) == pytest.approx(dcg / idcg)
    # And against the decimal, so a change to the discount is visible in the diff
    # rather than only inside a recomputed expression that changed with it.
    assert ndcg_at_k(RANKING, RELEVANT, 5) == pytest.approx(0.6509209298071326)


def test_ndcg_at_three_truncates_the_ranking_but_not_the_ideal() -> None:
    """At k=3 the DCG loses rank 4, while the ideal still packs both relevant
    chunks into ranks 1 and 2 - because ``min(3, 2) == 2``."""
    expected = (1.0 / math.log2(3)) / (1.0 + 1.0 / math.log2(3))
    assert ndcg_at_k(RANKING, RELEVANT, 3) == pytest.approx(expected)
    assert ndcg_at_k(RANKING, RELEVANT, 3) == pytest.approx(0.38685280723454163)


def test_the_discount_is_log2_of_rank_plus_one() -> None:
    """Rank 1 is undiscounted: ``log2(1 + 1) == 1``.

    Pinned separately because ``log2(i)`` instead of ``log2(i + 1)`` divides the
    first result by zero, and the usual "fix" for that - special-casing rank 1 -
    silently changes every other rank's weight.
    """
    single = ndcg_at_k(["x"], frozenset({"x"}), 1)
    assert single == pytest.approx(1.0)
    # Rank 2 alone, ideal at rank 1: the ratio is exactly 1/log2(3).
    assert ndcg_at_k(["y", "x"], frozenset({"x"}), 2) == pytest.approx(1.0 / math.log2(3))


def test_perfect_ranking_scores_one_on_every_metric() -> None:
    perfect = ["b", "d", "a", "c", "e"]
    assert recall_at_k(perfect, RELEVANT, 5) == pytest.approx(1.0)
    assert reciprocal_rank_at_k(perfect, RELEVANT, 5) == pytest.approx(1.0)
    assert ndcg_at_k(perfect, RELEVANT, 5) == pytest.approx(1.0)


def test_ideal_dcg_is_cut_off_at_k_so_a_perfect_page_scores_one() -> None:
    """THE BUG THIS EXISTS FOR.

    Three relevant chunks, a window of two, and the two best possible results in
    it. Computing IDCG over all three relevant chunks caps this ranking at
    ``1.6309 / 2.1309 = 0.765`` and the arm is penalised for a cutoff the harness
    chose, not for anything it did.
    """
    relevant = frozenset({"p", "q", "r"})
    assert ndcg_at_k(["p", "q", "z"], relevant, 2) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# The documented contracts
# ---------------------------------------------------------------------------


def test_k_larger_than_the_result_list_scores_what_came_back() -> None:
    """One result, one of two relevant chunks, a window of five.

    recall is 1/2 (the golden set is the denominator), precision is 1/5 (the
    window is), RR is 1 (it was first), and nDCG is 1/IDCG@5.
    """
    relevant = frozenset({"b", "d"})
    assert recall_at_k(["b"], relevant, 5) == pytest.approx(0.5)
    assert precision_at_k(["b"], relevant, 5) == pytest.approx(0.2)
    assert reciprocal_rank_at_k(["b"], relevant, 5) == pytest.approx(1.0)
    assert ndcg_at_k(["b"], relevant, 5) == pytest.approx(0.6131471927654584)


def test_precision_is_not_rescued_by_a_short_result_list() -> None:
    """Three results, all relevant, asked for ten: precision@10 is 0.3, not 1.0.

    The module docstring commits to this. It is pinned because it looks like a
    bug to anyone who has not read that paragraph, and "fixing" it would silently
    inflate every precision in the table.
    """
    relevant = frozenset({"a", "b", "c"})
    assert precision_at_k(["a", "b", "c"], relevant, 10) == pytest.approx(0.3)
    assert recall_at_k(["a", "b", "c"], relevant, 10) == pytest.approx(1.0)


def test_an_empty_ranking_is_a_measurement_and_scores_zero() -> None:
    """The retriever was asked and returned nothing. That is a result, not a gap."""
    assert recall_at_k([], RELEVANT, 5) == 0.0
    assert precision_at_k([], RELEVANT, 5) == 0.0
    assert reciprocal_rank_at_k([], RELEVANT, 5) == 0.0
    assert ndcg_at_k([], RELEVANT, 5) == 0.0


def test_a_query_with_no_relevant_documents_is_undefined_everywhere() -> None:
    """nDCG of a query with nothing to find is 0/0 - undefined, and not 0.0.

    All four return ``None`` together, which is the module's stated rule: a query
    that cannot discriminate between two retrievers contributes to no mean.
    Returning ``0.0`` from any of them would move an arm's score by an amount the
    arm had no way to influence.
    """
    empty: frozenset[str] = frozenset()
    assert recall_at_k(RANKING, empty, 5) is None
    assert precision_at_k(RANKING, empty, 5) is None
    assert reciprocal_rank_at_k(RANKING, empty, 5) is None
    assert ndcg_at_k(RANKING, empty, 5) is None


def test_reciprocal_rank_is_zero_when_the_hit_is_past_the_cutoff() -> None:
    """A relevant chunk at rank 6 of a 5-result page was never delivered."""
    ranking = ["a", "b", "c", "d", "e", "gold"]
    relevant = frozenset({"gold"})
    assert reciprocal_rank_at_k(ranking, relevant, 5) == 0.0
    assert reciprocal_rank_at_k(ranking, relevant, 6) == pytest.approx(1.0 / 6.0)


def test_a_repeated_id_in_the_window_is_refused() -> None:
    """Counting a chunk twice would put recall above 1.0 and read as a good arm."""
    with pytest.raises(MetricError, match="repeats an id"):
        recall_at_k(["b", "b"], RELEVANT, 5)
    # Outside the window it is not this module's problem: only the top-k prefix
    # is scored, so a duplicate below the cutoff cannot affect any number here.
    assert recall_at_k(["b", "d", "x", "x"], RELEVANT, 2) == pytest.approx(1.0)


@pytest.mark.parametrize("k", [0, -1])
def test_a_non_positive_cutoff_is_refused(k: int) -> None:
    with pytest.raises(MetricError, match="k must be positive"):
        ndcg_at_k(RANKING, RELEVANT, k)


# ---------------------------------------------------------------------------
# score_query and summarize
# ---------------------------------------------------------------------------


def test_score_query_carries_the_counts_that_make_it_readable() -> None:
    score = score_query("q1", RANKING, RELEVANT, 5)
    assert score.query_id == "q1"
    assert (score.k, score.n_relevant, score.n_retrieved) == (5, 2, 5)
    assert score.hit == 1.0
    assert score.value(RetrievalMetric.RECALL) == pytest.approx(1.0)
    assert score.value(RetrievalMetric.MRR) == pytest.approx(0.5)
    assert score.value(RetrievalMetric.NDCG) == pytest.approx(0.6509209298071326)


def test_score_query_records_a_miss_as_a_zero_hit_not_as_undefined() -> None:
    score = score_query("q1", ["x", "y"], RELEVANT, 2)
    assert score.hit == 0.0
    assert score.recall == 0.0


def test_summarize_drops_undefined_queries_and_says_how_many() -> None:
    """The count is the point: two arms averaged over different subsets of the
    queries are not comparable, and the only way a reader notices is if the
    denominators are printed."""
    summary = summarize(RetrievalMetric.RECALL, [1.0, 0.0, None, 0.5])
    assert summary.mean == pytest.approx(0.5)
    assert (summary.n_scored, summary.n_undefined, summary.n_queries) == (3, 1, 4)


def test_summarize_of_nothing_is_none_and_not_zero() -> None:
    """An average over zero scored queries is not 0.0 - it is nothing measured."""
    summary = summarize(RetrievalMetric.NDCG, [None, None])
    assert summary.mean is None
    assert (summary.n_scored, summary.n_undefined) == (0, 2)


def test_summarize_is_a_macro_average_not_a_pooled_one() -> None:
    """Each query weighs the same however large its golden set is.

    Pinned because the pooled alternative is a one-line change that would let a
    single many-answer query outvote several single-answer ones - which on a
    hand-built fixture means the person who wrote the labels chose the weights.
    """
    summary = summarize(RetrievalMetric.RECALL, [1.0, 0.0, 0.0])
    assert summary.mean == pytest.approx(1.0 / 3.0)
