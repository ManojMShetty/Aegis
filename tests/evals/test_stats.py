"""Tests for evals.stats.analysis - pinned against published values, offline.

WHY THESE TESTS LOOK LIKE THIS
------------------------------
A statistics module can be wrong in a way that never raises: an interval that is
merely too narrow, a p-value off by a factor of two, a bootstrap that resamples
the wrong unit. Tests that only check "returns a float between 0 and 1" would pass
against all three. So every core quantity here is pinned against a value published
somewhere a reader can look up, and the reference is named in the test:

* ``normal_quantile`` against the standard two-sided critical values
  (z = 1.959964 at 95%, 2.575829 at 99%), which appear in every statistics table.
* ``wilson_interval`` against Newcombe (1998), "Two-sided confidence intervals for
  the single proportion: comparison of seven methods", Statistics in Medicine
  17:857-872, Table I, method 3 (the score interval without continuity
  correction). Those four cases - 81/263, 15/148, 0/20, 1/29 - were chosen by
  Newcombe to span the interior and both boundaries, which is exactly the range
  this harness lives in.
* ``mcnemar_exact`` against the exact binomial test, whose values at these small
  counts are integers over powers of two and can be checked by hand: 10 discordant
  pairs all one way is 2/1024 = 0.001953125.

Two tests are about the FAILURE MODES that motivated the module rather than the
formulas:

* Wald has zero width at 0/n; Wilson must not, or a defended run scoring 0/12
  could claim perfection. Asserted directly against a Wald computation.
* Resampling couples instead of user tasks understates the interval. Asserted by
  constructing perfectly-correlated clusters and comparing the two widths.

And one is about the report rather than the maths: no rendered line may state a
change without an ``n`` and an interval beside it.

Everything is offline and deterministic. The only test that touches a real file
reads the committed ``results/week0_baseline.json`` and skips if it is absent.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
from evals.stats import analysis
from evals.stats.analysis import (
    MIN_DISCORDANT_FOR_SIGNIFICANCE,
    Metric,
    PairedObservation,
    PairingError,
    SchemaError,
    StatsError,
    cluster_bootstrap_difference,
    compare_runs,
    mcnemar_exact,
    normal_quantile,
    observations_from_results,
    paired_outcomes,
    render_comparison,
    render_summary,
    summarize_run,
    wilson_interval,
    z_for_confidence,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = REPO_ROOT / "results" / "week0_baseline.json"


# ---------------------------------------------------------------------------
# Inverse normal CDF
# ---------------------------------------------------------------------------


class TestNormalQuantile:
    """The z values every statistics table prints, to 1e-5 or better."""

    @pytest.mark.parametrize(
        ("confidence", "expected"),
        [
            (0.90, 1.644854),
            (0.95, 1.959964),
            (0.99, 2.575829),
        ],
    )
    def test_two_sided_critical_values(self, confidence: float, expected: float) -> None:
        # Published two-sided critical values: a 95% interval uses the 0.975
        # quantile. If z_for_confidence forgot to halve the tail it would return
        # 1.644854 for 0.95 and this would catch it.
        assert z_for_confidence(confidence) == pytest.approx(expected, abs=1e-6)

    def test_quantile_matches_the_forward_cdf(self) -> None:
        # A round trip through math.erfc: whatever the approximation does, the
        # value it returns must be the point where the true CDF equals p - pinned
        # across a region no published constant happens to touch. Note what this
        # does NOT prove: the Halley refinement step absorbs a subtly mistranscribed
        # rational coefficient (the composed function still lands on the right
        # answer), so this pins the defining property of the whole quantile
        # function, not the fidelity of the approximation feeding it.
        for p in (0.001, 0.01, 0.02, 0.024, 0.025, 0.3, 0.5, 0.7, 0.975, 0.99, 0.999):
            z = normal_quantile(p)
            recovered = 0.5 * math.erfc(-z / math.sqrt(2.0))
            assert recovered == pytest.approx(p, abs=1e-12)

    def test_symmetry_about_zero(self) -> None:
        for p in (0.01, 0.1, 0.25, 0.4):
            assert normal_quantile(p) == pytest.approx(-normal_quantile(1.0 - p), abs=1e-12)

    def test_median_is_zero(self) -> None:
        assert normal_quantile(0.5) == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
    def test_rejects_probabilities_outside_the_open_unit_interval(self, bad: float) -> None:
        # Returning +-inf would silently widen an interval to the whole line.
        with pytest.raises(StatsError, match="0 < p < 1"):
            normal_quantile(bad)

    @pytest.mark.parametrize("bad", [0.0, 1.0, 1.2])
    def test_rejects_impossible_confidence_levels(self, bad: float) -> None:
        with pytest.raises(StatsError, match="confidence"):
            z_for_confidence(bad)


# ---------------------------------------------------------------------------
# Wilson score interval
# ---------------------------------------------------------------------------


class TestWilsonInterval:
    """Pinned against Newcombe (1998) Table I and the harness's own sample sizes."""

    @pytest.mark.parametrize(
        ("successes", "trials", "low", "high"),
        [
            # Newcombe 1998, Statistics in Medicine 17:857-872, Table I, method 3
            # (Wilson score, no continuity correction). Interior, near-boundary,
            # and both boundaries.
            (81, 263, 0.2553, 0.3662),
            (15, 148, 0.0624, 0.1605),
            (0, 20, 0.0000, 0.1611),
            (1, 29, 0.0061, 0.1718),
        ],
    )
    def test_newcombe_table_one(self, successes: int, trials: int, low: float, high: float) -> None:
        interval = wilson_interval(successes, trials)
        assert interval.low == pytest.approx(low, abs=1e-3)
        assert interval.high == pytest.approx(high, abs=1e-3)

    @pytest.mark.parametrize(
        ("successes", "trials", "low", "high"),
        [
            # The sample sizes this harness actually reports at, computed from the
            # formula in the docstring: centre = (p + z^2/2n)/(1 + z^2/n),
            # margin = (z/(1+z^2/n)) * sqrt(p(1-p)/n + z^2/4n^2), z = 1.959964.
            (0, 10, 0.0000, 0.2775),
            (1, 12, 0.0149, 0.3539),
            (5, 10, 0.2366, 0.7634),
            (10, 10, 0.7225, 1.0000),
            (0, 12, 0.0000, 0.2425),
            (2, 3, 0.2077, 0.9385),
        ],
    )
    def test_harness_sample_sizes(
        self, successes: int, trials: int, low: float, high: float
    ) -> None:
        interval = wilson_interval(successes, trials)
        assert interval.low == pytest.approx(low, abs=1e-3)
        assert interval.high == pytest.approx(high, abs=1e-3)

    @pytest.mark.parametrize("trials", [1, 3, 10, 12, 40])
    def test_zero_successes_is_not_a_zero_width_interval(self, trials: int) -> None:
        """The exact failure mode Wald has, and the reason Wilson is used.

        A defended run scoring 0/12 must not be able to report certainty. The Wald
        half-width at p = 0 is z*sqrt(0*1/n) = 0 exactly; this asserts Wilson's
        upper bound is far from zero AND that Wald's really would be zero, so the
        test still means something if someone "simplifies" the implementation.
        """
        interval = wilson_interval(0, trials)
        wald_half_width = z_for_confidence(0.95) * math.sqrt(0.0 * 1.0 / trials)
        assert wald_half_width == 0.0
        assert interval.low == 0.0
        assert interval.high > 0.05
        assert interval.width == interval.high

    @pytest.mark.parametrize("trials", [1, 3, 10, 12, 40])
    def test_all_successes_is_not_a_zero_width_interval(self, trials: int) -> None:
        interval = wilson_interval(trials, trials)
        assert interval.high == 1.0
        assert interval.low < 0.95
        assert interval.width > 0.0

    def test_zero_trials_reports_no_estimate_and_the_widest_interval(self) -> None:
        # The documented contract: not measured is not the same as measured zero.
        interval = wilson_interval(0, 0)
        assert interval.point is None
        assert (interval.low, interval.high) == (0.0, 1.0)
        assert "n/a" in interval.as_percent_str()

    def test_intervals_stay_inside_the_unit_interval(self) -> None:
        for trials in range(1, 30):
            for successes in range(trials + 1):
                interval = wilson_interval(successes, trials)
                assert 0.0 <= interval.low <= interval.high <= 1.0

    def test_higher_confidence_is_wider(self) -> None:
        narrow = wilson_interval(1, 12, confidence=0.90)
        wide = wilson_interval(1, 12, confidence=0.99)
        assert wide.width > narrow.width

    def test_more_trials_at_the_same_rate_is_narrower(self) -> None:
        assert wilson_interval(10, 100).width < wilson_interval(1, 10).width

    def test_as_percent_str_always_carries_the_fraction(self) -> None:
        text = wilson_interval(1, 12).as_percent_str()
        assert "8.3%" in text
        assert "(1/12)" in text
        assert "95% CI" in text

    @pytest.mark.parametrize(
        ("successes", "trials"),
        [(-1, 10), (3, -1), (5, 4)],
    )
    def test_rejects_impossible_counts(self, successes: int, trials: int) -> None:
        with pytest.raises(StatsError):
            wilson_interval(successes, trials)


# ---------------------------------------------------------------------------
# Exact McNemar
# ---------------------------------------------------------------------------


class TestMcNemarExact:
    """Pinned against the exact binomial test at p = 1/2."""

    @pytest.mark.parametrize(
        ("only_a", "only_b", "expected"),
        [
            # One discordant pair: p = 2 * C(1,0)/2^1 = 1.0. The whole point of
            # the module - 1-of-12 vs 0-of-12 is not evidence.
            (1, 0, 1.0),
            # Ten one way, none the other: p = 2 * C(10,0)/2^10 = 2/1024.
            (10, 0, 0.001953125),
            # Symmetric: the smaller tail is already more than half, so the
            # doubled tail is capped at 1.
            (3, 3, 1.0),
            (4, 4, 1.0),
            # 2/2^2 = 0.5, and 2*(C(6,0)+C(6,1))/2^6 = 14/64.
            (2, 0, 0.5),
            (5, 1, 0.21875),
            (0, 2, 0.5),
            # Six the same way is the smallest count that reaches p < 0.05:
            # 2/2^6 = 0.03125.
            (6, 0, 0.03125),
            (5, 0, 0.0625),
        ],
    )
    def test_pinned_p_values(self, only_a: int, only_b: int, expected: float) -> None:
        result = mcnemar_exact(both=0, only_a=only_a, only_b=only_b, neither=0)
        assert result.p_value == pytest.approx(expected, abs=1e-12)

    def test_symmetric_in_the_two_discordant_cells(self) -> None:
        # Two-sided: swapping which condition won cannot change the p-value.
        for a, b in ((1, 4), (0, 7), (3, 8)):
            forward = mcnemar_exact(both=2, only_a=a, only_b=b, neither=5)
            reverse = mcnemar_exact(both=2, only_a=b, only_b=a, neither=5)
            assert forward.p_value == reverse.p_value

    def test_concordant_pairs_do_not_change_the_p_value(self) -> None:
        """The defining property of the test: only discordants carry information.

        If this ever fails, something is dividing by n_pairs instead of by the
        discordant count, and every reported p-value is wrong in the optimistic
        direction.
        """
        base = mcnemar_exact(both=0, only_a=3, only_b=0, neither=0)
        padded = mcnemar_exact(both=500, only_a=3, only_b=0, neither=900)
        assert padded.p_value == base.p_value

    def test_discordant_counts_are_surfaced(self) -> None:
        result = mcnemar_exact(both=4, only_a=1, only_b=0, neither=7)
        assert result.discordant_a == 1
        assert result.discordant_b == 0
        assert result.n_discordant == 1
        assert result.n_pairs == 12

    def test_no_discordant_pairs_is_no_evidence_not_a_statistic(self) -> None:
        result = mcnemar_exact(both=6, only_a=0, only_b=0, neither=6)
        assert result.p_value == 1.0
        assert result.statistic is None
        assert result.underpowered is True

    def test_chi_square_statistic_is_reported_but_unused(self) -> None:
        # (b-c)^2/(b+c) = (10-0)^2/10 = 10.0, whose chi-square p would be ~0.0016
        # rather than the exact 0.001953. Reported for readers, never the source
        # of p_value.
        result = mcnemar_exact(both=0, only_a=10, only_b=0, neither=0)
        assert result.statistic == pytest.approx(10.0)
        assert result.p_value == pytest.approx(0.001953125)

    def test_underpowered_threshold_is_the_real_boundary(self) -> None:
        """Brute-force the claim the report makes about power.

        MIN_DISCORDANT_FOR_SIGNIFICANCE says no split of fewer than 6 discordant
        pairs can reach p < 0.05. Check every split at every count rather than
        trusting the arithmetic in the comment.
        """
        for n_discordant in range(MIN_DISCORDANT_FOR_SIGNIFICANCE):
            for only_a in range(n_discordant + 1):
                result = mcnemar_exact(
                    both=0, only_a=only_a, only_b=n_discordant - only_a, neither=0
                )
                assert result.p_value >= 0.05
                assert result.underpowered is True
        best = mcnemar_exact(both=0, only_a=MIN_DISCORDANT_FOR_SIGNIFICANCE, only_b=0, neither=0)
        assert best.p_value < 0.05
        assert best.underpowered is False

    @pytest.mark.parametrize(
        "cells",
        [
            {"both": -1, "only_a": 0, "only_b": 0, "neither": 0},
            {"both": 0, "only_a": -2, "only_b": 0, "neither": 0},
            {"both": 0, "only_a": 0, "only_b": 0, "neither": -3},
        ],
    )
    def test_rejects_negative_cells(self, cells: dict[str, int]) -> None:
        with pytest.raises(StatsError, match="cannot be negative"):
            mcnemar_exact(**cells)


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------


class TestPairedOutcomes:
    def test_builds_the_two_by_two_table(self) -> None:
        a = {"t0::i0": True, "t0::i1": True, "t1::i0": False, "t1::i1": False}
        b = {"t0::i0": True, "t0::i1": False, "t1::i0": True, "t1::i1": False}
        counts = paired_outcomes(a, b)
        assert (counts.both, counts.only_a, counts.only_b, counts.neither) == (1, 1, 1, 1)
        assert counts.n_pairs == 4
        assert counts.rate_a == pytest.approx(0.5)
        assert counts.rate_b == pytest.approx(0.5)
        assert counts.difference == pytest.approx(0.0)

    def test_difference_is_defended_minus_baseline(self) -> None:
        # Baseline hijacked twice, defended never: the difference must be negative.
        a = {"t0::i0": True, "t0::i1": True, "t1::i0": False, "t1::i1": False}
        b = dict.fromkeys(a, False)
        counts = paired_outcomes(a, b)
        assert counts.difference == pytest.approx(-0.5)

    def test_refuses_a_missing_key(self) -> None:
        a = {"user_task_0::injection_task_0": True, "user_task_1::injection_task_0": False}
        b = {"user_task_0::injection_task_0": True}
        with pytest.raises(PairingError) as excinfo:
            paired_outcomes(a, b, label_a="baseline", label_b="defended")
        message = str(excinfo.value)
        assert "user_task_1::injection_task_0" in message
        assert "baseline" in message and "defended" in message
        assert "intersection" in message

    def test_refuses_an_extra_key(self) -> None:
        a = {"user_task_0::": True}
        b = {"user_task_0::": True, "user_task_9::": False}
        with pytest.raises(PairingError, match="user_task_9"):
            paired_outcomes(a, b)

    def test_refuses_same_size_but_different_keys(self) -> None:
        """The nastiest case: counts line up, tasks do not.

        An intersecting implementation would produce an empty-but-plausible table
        here instead of raising, which is exactly the confident-and-wrong answer
        the module exists to prevent.
        """
        a = {"user_task_0::injection_task_0": True, "user_task_1::injection_task_0": False}
        b = {"user_task_2::injection_task_0": True, "user_task_3::injection_task_0": False}
        with pytest.raises(PairingError):
            paired_outcomes(a, b)

    def test_names_only_a_few_offenders(self) -> None:
        a = {f"user_task_{i}::": True for i in range(20)}
        b: dict[str, bool] = {}
        with pytest.raises(PairingError) as excinfo:
            paired_outcomes(a, b)
        assert "more)" in str(excinfo.value)

    def test_empty_but_matching_key_sets_are_allowed(self) -> None:
        counts = paired_outcomes({}, {})
        assert counts.n_pairs == 0
        assert counts.rate_a is None
        assert counts.difference is None


class TestClusterOf:
    @pytest.mark.parametrize(
        ("key", "expected"),
        [
            ("user_task_0::injection_task_3", "user_task_0"),
            ("user_task_2::", "user_task_2"),
            ("user_task_7", "user_task_7"),
        ],
    )
    def test_cluster_is_the_user_task(self, key: str, expected: str) -> None:
        # The runner joins AgentDojo's tuple keys with "::" - clean results have an
        # empty injection half, injected couples do not.
        assert analysis.cluster_of(key) == expected


# ---------------------------------------------------------------------------
# Cluster bootstrap
# ---------------------------------------------------------------------------


def _clustered(n_clusters: int, per_cluster: int, hot: int) -> tuple[PairedObservation, ...]:
    """Perfectly correlated clusters: the first ``hot`` are all-True under A.

    Within a cluster every couple has the same outcome, which is the extreme of
    the correlation the real data has some of. Under A the hot clusters are all
    successes; under B nothing succeeds. The rate difference is therefore driven
    entirely by how many CLUSTERS are drawn, not by how many couples.
    """
    return tuple(
        PairedObservation(
            cluster=f"user_task_{c}",
            key=f"user_task_{c}::injection_task_{i}",
            outcome_a=c < hot,
            outcome_b=False,
        )
        for c in range(n_clusters)
        for i in range(per_cluster)
    )


def _uneven_clusters() -> tuple[PairedObservation, ...]:
    """Twelve clusters of sizes 1..12 with mixed outcomes.

    Uneven sizes mean the pooled rate of a resample depends on WHICH clusters were
    drawn and not just how many, so the statistic takes many distinct values. That
    is what makes the seed observable in the percentile endpoints.
    """
    return tuple(
        PairedObservation(
            cluster=f"user_task_{c}",
            key=f"user_task_{c}::injection_task_{i}",
            outcome_a=c % 3 == 0,
            outcome_b=c % 4 == 0,
        )
        for c in range(12)
        for i in range(c + 1)
    )


class TestClusterBootstrap:
    def test_is_deterministic_under_a_fixed_seed(self) -> None:
        pairs = _clustered(n_clusters=8, per_cluster=5, hot=4)
        first = cluster_bootstrap_difference(pairs, n_resamples=500, seed=7)
        second = cluster_bootstrap_difference(pairs, n_resamples=500, seed=7)
        assert (first.low, first.high) == (second.low, second.high)

    def test_a_different_seed_gives_a_different_draw(self) -> None:
        """Proof the seed reaches the RNG rather than being merely recorded.

        Deliberately NOT on the ``_clustered`` fixture: with 8 equal, internally
        identical clusters the difference can only take the nine values -k/8, so
        both seeds land the 2.5% and 97.5% percentiles on the same lattice points
        and an equal-endpoints assertion would prove nothing either way. Uneven
        cluster sizes give the statistic enough distinct values for the seed to
        show through.
        """
        pairs = _uneven_clusters()
        first = cluster_bootstrap_difference(pairs, n_resamples=500, seed=1)
        second = cluster_bootstrap_difference(pairs, n_resamples=500, seed=2)
        repeat = cluster_bootstrap_difference(pairs, n_resamples=500, seed=1)
        assert (first.low, first.high) != (second.low, second.high)
        assert (first.low, first.high) == (repeat.low, repeat.high)

    def test_point_estimate_is_the_observed_difference(self) -> None:
        pairs = _clustered(n_clusters=8, per_cluster=5, hot=4)
        # 20 of 40 successes under A, 0 under B.
        result = cluster_bootstrap_difference(pairs, n_resamples=200, seed=0)
        assert result.point == pytest.approx(-0.5)
        assert result.n_clusters == 8
        assert result.n_observations == 40

    def test_clustering_materially_widens_the_interval(self) -> None:
        """The reason the bootstrap resamples user tasks and not couples.

        Same 40 observations twice. Clustered, there are 8 independent units and
        the difference moves in steps of 1/8. Treated as 40 independent units (one
        cluster each - which is exactly what a naive per-observation bootstrap
        does), the same data yields an interval roughly sqrt(5) times narrower:
        the correlation has been assumed away. The clustered interval must be the
        wider one, by a lot.
        """
        clustered = _clustered(n_clusters=8, per_cluster=5, hot=4)
        # Same observations, but every one declared its own cluster: this IS the
        # naive per-observation bootstrap, expressed through the same code path.
        naive = tuple(
            PairedObservation(
                cluster=pair.key,
                key=pair.key,
                outcome_a=pair.outcome_a,
                outcome_b=pair.outcome_b,
            )
            for pair in clustered
        )
        by_cluster = cluster_bootstrap_difference(clustered, n_resamples=4000, seed=0)
        by_observation = cluster_bootstrap_difference(naive, n_resamples=4000, seed=0)

        assert by_cluster.n_clusters == 8
        assert by_observation.n_clusters == 40
        assert by_cluster.point == by_observation.point
        assert by_cluster.width > 1.5 * by_observation.width

    def test_interval_covers_the_point_estimate(self) -> None:
        pairs = _clustered(n_clusters=6, per_cluster=3, hot=2)
        result = cluster_bootstrap_difference(pairs, n_resamples=2000, seed=3)
        assert result.point is not None
        assert result.low <= result.point <= result.high

    def test_no_difference_gives_a_degenerate_interval(self) -> None:
        # Identical outcomes under both conditions: every resample has difference
        # zero, so both endpoints are zero. A bootstrap that jittered here would
        # be inventing variance.
        pairs = tuple(
            PairedObservation(
                cluster=f"user_task_{c}",
                key=f"user_task_{c}::injection_task_0",
                outcome_a=True,
                outcome_b=True,
            )
            for c in range(5)
        )
        result = cluster_bootstrap_difference(pairs, n_resamples=300, seed=0)
        assert (result.low, result.point, result.high) == (0.0, 0.0, 0.0)

    def test_wider_confidence_is_wider(self) -> None:
        # Strictly wider, not merely not-narrower: under '>=' a bootstrap that
        # ignored `confidence` altogether would return two identical intervals and
        # still pass. This fixture separates them (widths 0.5 vs 0.75), so the
        # strict comparison is what actually pins the parameter as load-bearing.
        pairs = _clustered(n_clusters=8, per_cluster=5, hot=4)
        narrow = cluster_bootstrap_difference(pairs, n_resamples=4000, seed=0, confidence=0.80)
        wide = cluster_bootstrap_difference(pairs, n_resamples=4000, seed=0, confidence=0.99)
        assert wide.width > narrow.width

    def test_no_observations_reports_the_full_range(self) -> None:
        result = cluster_bootstrap_difference((), n_resamples=10, seed=0)
        assert (result.low, result.high) == (-1.0, 1.0)
        assert result.point is None
        assert result.n_clusters == 0

    def test_records_its_own_provenance(self) -> None:
        pairs = _clustered(n_clusters=3, per_cluster=4, hot=1)
        result = cluster_bootstrap_difference(pairs, n_resamples=123, seed=99)
        assert result.n_resamples == 123
        assert result.seed == 99
        assert "pp" in result.as_points_str()

    @pytest.mark.parametrize(("resamples", "confidence"), [(0, 0.95), (-1, 0.95), (10, 1.0)])
    def test_rejects_impossible_settings(self, resamples: int, confidence: float) -> None:
        pairs = _clustered(n_clusters=2, per_cluster=2, hot=1)
        with pytest.raises(StatsError):
            cluster_bootstrap_difference(pairs, n_resamples=resamples, confidence=confidence)


class TestObservationsFromResults:
    def test_derives_clusters_from_the_runner_key_format(self) -> None:
        a = {"user_task_0::injection_task_0": True, "user_task_0::injection_task_1": False}
        b = {"user_task_0::injection_task_0": False, "user_task_0::injection_task_1": False}
        observations = observations_from_results(a, b)
        assert {o.cluster for o in observations} == {"user_task_0"}
        assert len(observations) == 2

    def test_refuses_mismatched_keys_like_paired_outcomes_does(self) -> None:
        with pytest.raises(PairingError):
            observations_from_results({"user_task_0::": True}, {"user_task_1::": True})


# ---------------------------------------------------------------------------
# Reading runner results
# ---------------------------------------------------------------------------


def _runner_payload(
    *,
    clean_utility: dict[str, bool] | None = None,
    injected_security: dict[str, bool] | None = None,
    injected_utility: dict[str, bool] | None = None,
    headline: bool = True,
) -> dict[str, Any]:
    """A minimal payload in the schema evals.agentdojo.runner writes."""
    clean_utility = clean_utility or {"user_task_0::": False, "user_task_1::": True}
    injected_security = injected_security or {
        "user_task_0::injection_task_0": True,
        "user_task_0::injection_task_1": False,
        "user_task_1::injection_task_0": False,
        "user_task_1::injection_task_1": False,
    }
    injected_utility = injected_utility or dict.fromkeys(injected_security, True)
    payload: dict[str, Any] = {
        "model": "openai/gpt-oss-120b",
        "suite": "workspace",
        "attack": "important_instructions",
        "raw": {
            "clean": {
                "utility_results": clean_utility,
                "security_results": dict.fromkeys(clean_utility, True),
            },
            "injected": {
                "utility_results": injected_utility,
                "security_results": injected_security,
                "injection_tasks_utility_results": {"injection_task_0": True},
            },
        },
    }
    if headline:
        payload["utility"] = sum(clean_utility.values()) / len(clean_utility)
        payload["asr"] = sum(injected_security.values()) / len(injected_security)
        payload["utility_under_attack"] = sum(injected_utility.values()) / len(injected_utility)
    return payload


class TestSummarizeRun:
    def test_reads_a_mapping(self) -> None:
        summary = summarize_run(_runner_payload(), label="baseline")
        assert summary.interval(Metric.UTILITY).point == pytest.approx(0.5)
        assert summary.interval(Metric.ASR).point == pytest.approx(0.25)
        assert summary.interval(Metric.UTILITY_UNDER_ATTACK).point == pytest.approx(1.0)
        assert summary.label == "baseline"

    def test_reads_a_file_and_defaults_the_label_to_the_stem(self, tmp_path: Path) -> None:
        path = tmp_path / "week9_defended.json"
        path.write_text(json.dumps(_runner_payload()), encoding="utf-8")
        summary = summarize_run(path)
        assert summary.label == "week9_defended"
        assert summary.config["model"] == "openai/gpt-oss-120b"

    def test_keeps_the_per_task_booleans(self) -> None:
        summary = summarize_run(_runner_payload())
        assert summary.results_for(Metric.ASR)["user_task_0::injection_task_0"] is True
        assert summary.results_for(Metric.UTILITY)["user_task_0::"] is False

    @pytest.mark.parametrize(
        "drop",
        [
            ("raw",),
            ("raw", "clean"),
            ("raw", "injected"),
        ],
    )
    def test_rejects_an_old_schema_file(self, drop: tuple[str, ...]) -> None:
        """An old-schema file must fail loudly, not read as zero successes."""
        payload = _runner_payload()
        node: Any = payload
        for part in drop[:-1]:
            node = node[part]
        del node[drop[-1]]
        with pytest.raises(SchemaError) as excinfo:
            summarize_run(payload)
        assert "evals.agentdojo.runner" in str(excinfo.value)

    def test_rejects_a_missing_leaf_block(self) -> None:
        payload = _runner_payload()
        del payload["raw"]["injected"]["security_results"]
        with pytest.raises(SchemaError, match=r"raw\.injected\.security_results"):
            summarize_run(payload)

    def test_rejects_non_boolean_outcomes(self) -> None:
        payload = _runner_payload()
        payload["raw"]["injected"]["security_results"]["user_task_0::injection_task_0"] = None
        payload["asr"] = 0.0
        with pytest.raises(SchemaError, match="cannot be averaged"):
            summarize_run(payload)

    def test_rejects_a_file_that_contradicts_its_own_raw_block(self) -> None:
        """The headline number and the per-task outcomes must agree.

        A hand-edited "asr": 0.0 sitting above a raw block containing a True is the
        one corruption a summariser could otherwise repeat with a straight face.
        """
        payload = _runner_payload()
        payload["asr"] = 0.0
        with pytest.raises(SchemaError, match="disagree"):
            summarize_run(payload)

    def test_tolerates_a_file_with_no_headline_fields(self) -> None:
        # The raw block is the source of truth; the headline is a cross-check when
        # present, not a requirement.
        summary = summarize_run(_runner_payload(headline=False))
        assert summary.interval(Metric.ASR).point == pytest.approx(0.25)

    def test_rejects_a_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(SchemaError, match="no such result file"):
            summarize_run(tmp_path / "nope.json")

    def test_rejects_malformed_json(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(SchemaError, match="not valid JSON"):
            summarize_run(path)


# ---------------------------------------------------------------------------
# Comparison and rendering
# ---------------------------------------------------------------------------


def _baseline_and_defended() -> tuple[Any, Any]:
    """Week-0-shaped pair: 1-of-12 ASR against 0-of-12, over 3 clusters."""
    security_baseline = {
        f"user_task_{u}::injection_task_{i}": (u == 0 and i == 0)
        for u in range(3)
        for i in range(4)
    }
    security_defended = dict.fromkeys(security_baseline, False)
    clean = {f"user_task_{u}::": u > 0 for u in range(3)}
    baseline = summarize_run(
        _runner_payload(clean_utility=clean, injected_security=security_baseline),
        label="baseline",
    )
    defended = summarize_run(
        _runner_payload(clean_utility=clean, injected_security=security_defended),
        label="defended",
    )
    return baseline, defended


class TestCompareRuns:
    def test_pairs_every_metric(self) -> None:
        baseline, defended = _baseline_and_defended()
        report = compare_runs(baseline, defended, n_resamples=500, seed=0)
        assert set(report.comparisons) == set(Metric)

    def test_the_week_zero_shaped_comparison_is_a_nothing_burger(self) -> None:
        """1-of-12 vs 0-of-12: exactly the claim the module exists to deflate."""
        baseline, defended = _baseline_and_defended()
        report = compare_runs(baseline, defended, n_resamples=2000, seed=0)
        asr = report.comparisons[Metric.ASR]
        assert asr.counts.only_a == 1
        assert asr.counts.only_b == 0
        assert asr.mcnemar.p_value == pytest.approx(1.0)
        assert asr.mcnemar.underpowered is True
        # The interval on the difference straddles or touches zero: no reduction
        # has been demonstrated.
        assert asr.bootstrap.high >= 0.0
        assert asr.bootstrap.point == pytest.approx(-1 / 12)

    def test_refuses_to_compare_different_task_sets(self) -> None:
        baseline, _ = _baseline_and_defended()
        partial = summarize_run(
            _runner_payload(
                injected_security={"user_task_0::injection_task_0": False},
            ),
            label="defended",
        )
        with pytest.raises(PairingError):
            compare_runs(baseline, partial, n_resamples=10, seed=0)


class TestRendering:
    def test_summary_table_shows_rate_fraction_and_interval(self) -> None:
        baseline, _ = _baseline_and_defended()
        text = render_summary(baseline)
        assert text.isascii()
        assert "asr" in text
        assert "1/12" in text
        assert "95% CI" in text

    def test_no_change_claim_appears_without_n_and_an_interval(self) -> None:
        """The invariant the whole reporting design exists to guarantee.

        Any line that states a change must also carry the sample size and the
        confidence interval. A future edit that splits the change onto its own
        line for tidiness breaks this test, which is the intent.
        """
        baseline, defended = _baseline_and_defended()
        text = render_comparison(compare_runs(baseline, defended, n_resamples=500, seed=0))
        change_lines = [line for line in text.splitlines() if "change" in line and "pp" in line]
        assert change_lines, "expected at least one rendered change claim"
        for line in change_lines:
            assert "n=" in line, line
            assert "CI [" in line, line
            assert "pairs" in line, line

    def test_no_line_states_a_percentage_reduction_on_its_own(self) -> None:
        # Belt and braces: scan every line carrying a signed percentage-point
        # figure, not just the ones containing the word "change".
        baseline, defended = _baseline_and_defended()
        text = render_comparison(compare_runs(baseline, defended, n_resamples=500, seed=0))
        for line in text.splitlines():
            if " pp" not in line:
                continue
            assert "CI [" in line and "n=" in line, line

    def test_comparison_shows_discordant_counts_and_the_p_value(self) -> None:
        baseline, defended = _baseline_and_defended()
        text = render_comparison(compare_runs(baseline, defended, n_resamples=500, seed=0))
        assert "discordant pairs" in text
        assert "baseline-only 1, defended-only 0" in text
        assert "McNemar exact p" in text

    def test_comparison_flags_that_the_test_could_not_have_shown_an_effect(self) -> None:
        baseline, defended = _baseline_and_defended()
        text = render_comparison(compare_runs(baseline, defended, n_resamples=500, seed=0))
        assert "POWER" in text
        assert "could not have shown an effect" in text

    def test_comparison_records_the_bootstrap_seed_and_the_cluster_unit(self) -> None:
        baseline, defended = _baseline_and_defended()
        text = render_comparison(compare_runs(baseline, defended, n_resamples=777, seed=42))
        assert "777 resamples" in text
        assert "BY USER TASK" in text
        assert "seed 42" in text

    def test_everything_rendered_is_ascii(self) -> None:
        baseline, defended = _baseline_and_defended()
        report = compare_runs(baseline, defended, n_resamples=200, seed=0)
        assert render_summary(baseline).isascii()
        assert render_comparison(report).isascii()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli:
    def test_baseline_only_prints_the_intervals(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "run.json"
        path.write_text(json.dumps(_runner_payload()), encoding="utf-8")
        assert analysis.main(["--baseline", str(path)]) == 0
        out = capsys.readouterr().out
        assert "Aegis eval statistics" in out
        assert "Paired comparison" not in out

    def test_both_files_adds_the_paired_comparison(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        base = tmp_path / "base.json"
        defended = tmp_path / "defended.json"
        security_baseline = {
            f"user_task_{u}::injection_task_{i}": (u == 0 and i == 0)
            for u in range(3)
            for i in range(4)
        }
        base.write_text(
            json.dumps(_runner_payload(injected_security=security_baseline)), encoding="utf-8"
        )
        defended.write_text(
            json.dumps(_runner_payload(injected_security=dict.fromkeys(security_baseline, False))),
            encoding="utf-8",
        )
        exit_code = analysis.main(
            ["--baseline", str(base), "--defended", str(defended), "--resamples", "200"]
        )
        assert exit_code == 0
        out = capsys.readouterr().out
        assert "Paired comparison" in out
        assert "McNemar exact p" in out

    def test_old_schema_file_fails_with_a_clear_message(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "old.json"
        path.write_text(json.dumps({"utility": 0.5, "asr": 0.1}), encoding="utf-8")
        with pytest.raises(SystemExit) as excinfo:
            analysis.main(["--baseline", str(path)])
        assert excinfo.value.code == 2
        assert "evals.agentdojo.runner" in capsys.readouterr().err

    def test_unpairable_files_fail_rather_than_intersect(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        base = tmp_path / "base.json"
        partial = tmp_path / "partial.json"
        base.write_text(json.dumps(_runner_payload()), encoding="utf-8")
        partial.write_text(
            json.dumps(
                _runner_payload(
                    injected_security={"user_task_0::injection_task_0": False},
                )
            ),
            encoding="utf-8",
        )
        with pytest.raises(SystemExit) as excinfo:
            analysis.main(["--baseline", str(base), "--defended", str(partial)])
        assert excinfo.value.code == 2
        assert "do not describe the same tasks" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# The real recorded baseline
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not BASELINE_PATH.is_file(),
    reason="results/week0_baseline.json is not present in this checkout",
)
class TestRecordedBaseline:
    """End-to-end against the committed Week-0 artifact, if it is there.

    This is the test that would catch a schema drift between the runner and the
    analysis: it reads the real file, recomputes every rate from the raw per-task
    booleans, and asserts the recomputation reproduces the headline numbers the
    runner recorded. It never makes a network call - the file is the whole input.
    """

    def test_summary_reproduces_the_recorded_point_estimates(self) -> None:
        recorded = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        summary = summarize_run(BASELINE_PATH, label="week0")
        for metric in Metric:
            if recorded.get(str(metric)) is None:
                continue
            assert summary.interval(metric).point == pytest.approx(
                recorded[str(metric)], abs=1e-12
            ), metric

    def test_asr_carries_a_wide_interval_at_this_sample_size(self) -> None:
        summary = summarize_run(BASELINE_PATH)
        asr = summary.interval(Metric.ASR)
        # Whatever the recorded ASR is, twelve-ish couples cannot pin it down: the
        # interval must be wide enough that a reader cannot mistake the point
        # estimate for a measurement of the true rate.
        assert asr.trials <= 64
        assert asr.width > 0.15

    def test_the_file_can_be_paired_with_itself(self) -> None:
        """A run against itself: zero discordant pairs, p = 1, difference 0.

        A degenerate comparison, but it exercises the whole pipeline on the real
        key format - if the "::" join or the cluster extraction were wrong, the
        pairing or the clustering would break here.
        """
        summary = summarize_run(BASELINE_PATH, label="week0")
        report = compare_runs(summary, summary, n_resamples=200, seed=0)
        asr = report.comparisons[Metric.ASR]
        assert asr.counts.only_a == 0
        assert asr.counts.only_b == 0
        assert asr.mcnemar.p_value == 1.0
        assert asr.bootstrap.point == pytest.approx(0.0)
        assert asr.bootstrap.n_clusters == len(
            {key.split("::")[0] for key in summary.results_for(Metric.ASR)}
        )
