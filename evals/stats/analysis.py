"""Honest statistics for the eval harness: intervals, paired tests, and a table.

WHY THIS MODULE EXISTS
----------------------
:mod:`evals.agentdojo.runner` produces a Week-0 baseline of three rates measured
over very few trials: utility over 3 user tasks, ASR and utility-under-attack over
12 (user, injection) couples. Two runs of the IDENTICAL configuration scored
utility 100% and then 66.7%, which is not a bug - it is what 3 Bernoulli trials
look like. A bare "ASR 8.3%" printed next to a later "ASR 0.0%" invites exactly
one reading ("the defense works") that the sample size cannot support. Every
number this project reports therefore travels with the width of what it can
actually support, and every before/after claim travels with a paired test.

WHAT IS IMPLEMENTED HERE, AND WHY BY HAND
-----------------------------------------
``scipy`` and ``statsmodels`` are declared in the ``evals`` extra but are not
installed, and installing them is out of scope for a module whose whole job is to
be *checkable*. Everything below is standard-library ``math`` only:

* :func:`normal_quantile` - the inverse normal CDF (Acklam's rational
  approximation plus one Halley step against :func:`math.erfc`), because the
  Wilson interval needs ``z`` and nothing in the stdlib provides it.
* :func:`wilson_interval` - the Wilson score interval. WHY not the textbook
  normal-approximation (Wald) interval: Wald's half-width is
  ``z * sqrt(p(1-p)/n)``, which is exactly ZERO at ``p = 0``. A defended run that
  scores 0/12 would then report "ASR 0.0% (0.0%, 0.0%)" and claim perfection from
  twelve trials. Wilson's interval at 0/12 is ``(0.0%, 24.2%)`` - the honest
  statement that twelve clean trials rule out a large ASR and nothing more.
* :func:`mcnemar_exact` - the exact binomial McNemar test, the right test for the
  same tasks run under two conditions.
* :func:`cluster_bootstrap_difference` - a percentile bootstrap on the paired
  difference in rates, resampling USER TASKS rather than couples.

THE THING THE REPORT MUST NEVER DO
----------------------------------
Print a reduction without its ``n`` and its interval. :func:`render_comparison`
emits the change, the confidence interval and the pair count on one line, from one
formatting helper (:func:`_change_line`), so the three cannot be separated by a
later edit. ``tests/evals/test_stats.py`` asserts on the rendered string that no
change claim ever appears alone.

A NOTE ON UNITS
---------------
Differences are reported in PERCENTAGE POINTS, signed, as ``defended - baseline``.
"ASR fell from 8.3% to 0%" is a change of -8.3 pp; calling it "a 100% reduction"
is the relative framing, and at one discordant pair it is theatre. The bootstrap
interval is on the percentage-point difference, so the reported point estimate and
the reported interval are in the same unit.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Conventions, named once so the CLI help, the docstrings and the code agree.
# ---------------------------------------------------------------------------

DEFAULT_CONFIDENCE = 0.95
# 10k resamples puts the Monte-Carlo error on a 95% percentile endpoint well below
# the width the small sample already forces, so more resamples would buy nothing a
# reader could see. It is cheap enough (no numpy) to stay the default.
DEFAULT_RESAMPLES = 10_000
# Fixed by default so two people running the same command get the same interval.
# The seed is part of the reported result, not an implementation detail - a
# bootstrap CI whose seed is unrecorded is not reproducible.
DEFAULT_SEED = 0

# The runner joins AgentDojo's ``(user_task_id, injection_task_id)`` tuple keys
# with this separator (see ``_stringify_keys`` there). The user task is the part
# before it, and the user task is the bootstrap CLUSTER.
KEY_JOIN = "::"

# With this many discordant pairs or fewer, the exact two-sided test CANNOT
# return p < 0.05 for ANY split: the smallest attainable p on n discordant pairs
# is 2 * 0.5**n, which is 0.0625 at n = 5 and 0.03125 at n = 6. Below the
# threshold the report says so out loud rather than letting a reader mistake a
# large p for a measured null result. Pinned by a test.
MIN_DISCORDANT_FOR_SIGNIFICANCE = 6

_RULE_WIDTH = 78


class Metric(StrEnum):
    """The three rates the runner records, named by their JSON field.

    The values are exactly the runner's top-level keys, so a summary can be
    cross-checked against the headline number in the same file without a
    translation table that could drift.
    """

    UTILITY = "utility"
    ASR = "asr"
    UTILITY_UNDER_ATTACK = "utility_under_attack"


# Which raw per-task block each headline rate is the mean of. This mirrors
# ``run_baseline`` in the runner exactly: utility is the mean of the CLEAN
# utility results, asr the mean of the INJECTED security results (True = the
# attack succeeded), utility-under-attack the mean of the injected utility
# results.
_METRIC_SOURCE: dict[Metric, tuple[str, str]] = {
    Metric.UTILITY: ("clean", "utility_results"),
    Metric.ASR: ("injected", "security_results"),
    Metric.UTILITY_UNDER_ATTACK: ("injected", "utility_results"),
}


class StatsError(ValueError):
    """Base class for every refusal in this module.

    All of them are refusals to compute something misleading, so they share a
    type: a caller that wants to report "cannot compare these" instead of
    crashing catches one exception, not four.
    """


class SchemaError(StatsError):
    """A result file is not the shape :mod:`evals.agentdojo.runner` writes.

    Raised eagerly rather than tolerated, because the alternative - reading an
    old-schema file with a missing ``raw`` block as "zero successes" - produces a
    confident number from data that was never measured.
    """


class PairingError(StatsError):
    """Two result sets do not describe the same tasks and cannot be paired.

    A paired test on the intersection of two different task sets answers a
    question nobody asked, in the voice of the question they did ask.
    """


# ---------------------------------------------------------------------------
# Inverse normal CDF
# ---------------------------------------------------------------------------

# Peter Acklam's rational approximation to the inverse normal CDF (2000). Two
# rational functions - one for the central region, one mirrored across both tails
# - with a stated relative error below 1.15e-9. The coefficients are his,
# transcribed verbatim; the region split at 0.02425 is his too.
_ACKLAM_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_ACKLAM_B = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_ACKLAM_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_ACKLAM_D = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
)
_ACKLAM_P_LOW = 0.02425


def normal_quantile(p: float) -> float:
    """The inverse standard normal CDF: the ``z`` with ``P(Z <= z) == p``.

    WHY THIS IS HERE AT ALL
        The Wilson interval needs a normal quantile and the standard library has
        no inverse CDF (:func:`math.erfc` is the forward direction only). Rather
        than take a dependency on ``scipy`` for one number, the approximation is
        forty lines a reviewer can read.

    HOW IT WORKS
        Acklam's rational approximation gives roughly nine correct digits on its
        own. One Halley step then polishes it against :func:`math.erfc`, which is
        the exact forward CDF the stdlib does provide: the residual
        ``e = Phi(x) - p`` is computed with ``erfc`` and the correction
        ``u / (1 + x*u/2)`` applied once. The result agrees with published tables
        far past the 1e-5 the tests pin.

    Args:
        p: A probability, strictly between 0 and 1.

    Returns:
        The quantile ``z``. ``normal_quantile(0.975) == 1.959963985...``.

    Raises:
        StatsError: If ``p`` is not strictly inside ``(0, 1)``. The quantile
            diverges at both ends, and returning an infinity would silently
            produce an interval covering the whole line.
    """
    if not 0.0 < p < 1.0:
        raise StatsError(f"normal_quantile needs 0 < p < 1, got p={p!r} (the quantile diverges)")

    if p < _ACKLAM_P_LOW:
        x = _acklam_tail(math.sqrt(-2.0 * math.log(p)))
    elif p <= 1.0 - _ACKLAM_P_LOW:
        q = p - 0.5
        r = q * q
        num = (
            ((((_ACKLAM_A[0] * r + _ACKLAM_A[1]) * r + _ACKLAM_A[2]) * r + _ACKLAM_A[3]) * r)
            + _ACKLAM_A[4]
        ) * r + _ACKLAM_A[5]
        den = (
            ((((_ACKLAM_B[0] * r + _ACKLAM_B[1]) * r + _ACKLAM_B[2]) * r + _ACKLAM_B[3]) * r)
            + _ACKLAM_B[4]
        ) * r + 1.0
        x = num * q / den
    else:
        x = -_acklam_tail(math.sqrt(-2.0 * math.log(1.0 - p)))

    # One Halley refinement against the true CDF. Cheap, and it turns "nine
    # digits" into "as many as a double holds", so the constants the tests pin are
    # published values rather than artefacts of the approximation.
    err = 0.5 * math.erfc(-x / math.sqrt(2.0)) - p
    u = err * math.sqrt(2.0 * math.pi) * math.exp(x * x / 2.0)
    return x - u / (1.0 + x * u / 2.0)


def _acklam_tail(q: float) -> float:
    """Acklam's lower-tail branch, evaluated at ``q = sqrt(-2 ln p)``.

    Factored out because the upper tail is the same expression negated; writing it
    once means the two tails cannot drift apart under editing.
    """
    num = (
        ((((_ACKLAM_C[0] * q + _ACKLAM_C[1]) * q + _ACKLAM_C[2]) * q + _ACKLAM_C[3]) * q)
        + _ACKLAM_C[4]
    ) * q + _ACKLAM_C[5]
    den = (((_ACKLAM_D[0] * q + _ACKLAM_D[1]) * q + _ACKLAM_D[2]) * q + _ACKLAM_D[3]) * q + 1.0
    return num / den


def z_for_confidence(confidence: float) -> float:
    """The two-sided critical value for a confidence level.

    ``z_for_confidence(0.95)`` is the 0.975 quantile, 1.959964, because a
    two-sided 95% interval leaves 2.5% in each tail. Spelling that out in one
    named function keeps an off-by-a-factor-of-two out of every call site.
    """
    if not 0.0 < confidence < 1.0:
        raise StatsError(f"confidence must be strictly between 0 and 1, got {confidence!r}")
    return normal_quantile(1.0 - (1.0 - confidence) / 2.0)


# ---------------------------------------------------------------------------
# Wilson score interval
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Interval:
    """A proportion with the interval that honestly bounds it.

    ``point`` is deliberately ``float | None``: with zero trials there is no
    estimate, and reporting ``0.0`` would read as "ran and scored zero". The
    runner's ``_mean`` makes the same choice for the same reason. Callers are
    forced by the type to decide what "we did not measure this" renders as, which
    is the point.
    """

    low: float
    high: float
    point: float | None
    successes: int
    trials: int
    confidence: float

    @property
    def width(self) -> float:
        """How much room the evidence leaves. The number readers should look at."""
        return self.high - self.low

    def as_percent_str(self) -> str:
        """One-line rendering: rate, the fraction it came from, and the interval.

        The fraction is never omitted. "8.3%" and "1/12" say different things to a
        reader and only the pair is honest, so this helper cannot emit one without
        the other.
        """
        level = f"{self.confidence * 100:.0f}%"
        rate = "n/a" if self.point is None else f"{self.point * 100:.1f}%"
        return (
            f"{rate} ({self.successes}/{self.trials}) "
            f"{level} CI [{self.low * 100:.1f}%, {self.high * 100:.1f}%]"
        )


def wilson_interval(
    successes: int,
    trials: int,
    confidence: float = DEFAULT_CONFIDENCE,
) -> Interval:
    """The Wilson score interval for a binomial proportion.

    WHY WILSON AND NOT THE NORMAL APPROXIMATION
        The Wald interval ``p +- z*sqrt(p(1-p)/n)`` collapses to zero width at
        ``p = 0`` and at ``p = 1``. Those are precisely the values this harness
        expects: a working defense scores 0/12 on ASR, and a Wald interval would
        let that run report "0.0% (0.0%, 0.0%)" - certainty, from twelve trials.
        Wilson inverts the score test instead of the Wald test, so its centre is
        pulled toward 1/2 by ``z^2/2n`` and its width stays positive at the
        boundaries. At 0/12 it reports ``(0.0%, 24.2%)``.

    THE FORMULA (checkable by hand)
        With ``p = s/n`` and ``d = 1 + z^2/n``::

            centre = (p + z^2/(2n)) / d
            margin = (z/d) * sqrt( p(1-p)/n + z^2/(4n^2) )

        At ``s = 0`` the margin equals the centre exactly, so ``low`` is 0 and
        ``high`` is twice the centre - a zero LOWER bound, never a zero WIDTH.

    THE ZERO-TRIALS CONTRACT
        ``trials == 0`` returns ``low=0.0, high=1.0, point=None``: the interval
        that states, correctly, that no evidence constrains the rate at all.
        Returning ``0.0`` as the point estimate would be a measurement claim about
        a measurement that never happened.

    Args:
        successes: Number of successes observed. Must not be negative or exceed
            ``trials``.
        trials: Number of Bernoulli trials.
        confidence: Two-sided confidence level, default 0.95.

    Returns:
        An :class:`Interval` clamped to ``[0, 1]``.

    Raises:
        StatsError: On negative counts, ``successes > trials``, or a confidence
            outside ``(0, 1)``.
    """
    if trials < 0 or successes < 0:
        raise StatsError(f"counts cannot be negative, got successes={successes}, trials={trials}")
    if successes > trials:
        raise StatsError(f"successes={successes} exceeds trials={trials}")

    z = z_for_confidence(confidence)
    if trials == 0:
        return Interval(
            low=0.0,
            high=1.0,
            point=None,
            successes=0,
            trials=0,
            confidence=confidence,
        )

    p = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    centre = (p + z2 / (2 * trials)) / denominator
    margin = (z / denominator) * math.sqrt(p * (1.0 - p) / trials + z2 / (4 * trials * trials))
    # The two boundaries are exact in the closed form - at s = 0 the margin equals
    # the centre and at s = n it equals 1 minus the centre - but the floating-point
    # evaluation of two different expressions lands a few ulps off, which showed up
    # as a "lower bound" of 2.8e-17 for a rate of zero. Snapping them is enforcing
    # the algebra, not fudging the result, and it keeps the documented contract
    # ("0/n has a zero LOWER bound, never a zero WIDTH") literally true.
    low = 0.0 if successes == 0 else max(0.0, centre - margin)
    high = 1.0 if successes == trials else min(1.0, centre + margin)
    return Interval(
        low=low,
        high=high,
        point=p,
        successes=successes,
        trials=trials,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Exact McNemar test on paired outcomes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class McNemarResult:
    """The outcome of an exact McNemar test, with the counts that produced it.

    ``discordant_a`` and ``discordant_b`` are fields, not internals, because the
    p-value alone is uninterpretable at this sample size. "ASR fell from 8.3% to
    0%" backed by ONE discordant pair is a nothing-burger, and the only way a
    reader can see that is if the discordant counts sit next to the p-value.
    """

    p_value: float
    discordant_a: int
    discordant_b: int
    n_pairs: int
    statistic: float | None

    @property
    def n_discordant(self) -> int:
        """The pairs that carry information. The effective sample size."""
        return self.discordant_a + self.discordant_b

    @property
    def underpowered(self) -> bool:
        """True when no split of these discordant pairs could reach p < 0.05.

        See :data:`MIN_DISCORDANT_FOR_SIGNIFICANCE`. This is a property of the
        design, knowable before the data: it says "this comparison could not have
        shown anything", which is a different sentence from "this comparison
        showed nothing".
        """
        return self.n_discordant < MIN_DISCORDANT_FOR_SIGNIFICANCE


def mcnemar_exact(both: int, only_a: int, only_b: int, neither: int) -> McNemarResult:
    """The exact (binomial) McNemar test on a 2x2 table of PAIRED outcomes.

    WHY PAIRED, AND WHY ONLY THE DISCORDANTS
        The defended run executes the SAME tasks as the baseline, not an
        independent sample, so a two-sample proportion test is the wrong tool - it
        would charge "task 3 is hard" to noise twice instead of cancelling it.
        McNemar conditions on the pairs that CHANGED. Tasks the attack hijacked
        under both conditions (``both``) and under neither (``neither``) say
        nothing about which condition is better; they only inflate ``n``. All the
        information is in ``only_a`` and ``only_b``, and both counts are returned
        so a report can show how thin that is.

    WHY EXACT AND NOT CHI-SQUARE
        The classical McNemar statistic ``(b-c)^2/(b+c)`` is referred to a
        chi-square distribution, an approximation that needs roughly
        ``b + c >= 25`` to be trustworthy. This harness expects ``b + c`` in the
        single digits, where the approximation's p-values are simply wrong. The
        exact test asks the question directly instead: if the condition made no
        difference, each discordant pair is a fair coin, so the number landing on
        one side is ``Binomial(b + c, 1/2)``. The two-sided p-value is twice the
        smaller tail, capped at 1. (At ``p = 1/2`` the binomial is symmetric, so
        doubling the smaller tail is identical to summing every outcome at most as
        likely as the observed one - the two usual two-sided conventions agree
        here, and no choice is being smuggled in.)

        The chi-square statistic is still reported in ``statistic`` for readers who
        expect to see it, and is explicitly NOT where the p-value came from.

    Args:
        both: Pairs where both conditions recorded True.
        only_a: Pairs True under condition A only (the "baseline-only" cell).
        only_b: Pairs True under condition B only (the "defended-only" cell).
        neither: Pairs where both conditions recorded False.

    Returns:
        A :class:`McNemarResult`. With zero discordant pairs the p-value is 1.0
        and ``statistic`` is ``None`` - there is no evidence of a difference
        because there is no evidence at all.

    Raises:
        StatsError: If any cell is negative.
    """
    cells = {"both": both, "only_a": only_a, "only_b": only_b, "neither": neither}
    negative = {name: value for name, value in cells.items() if value < 0}
    if negative:
        raise StatsError(f"2x2 cells cannot be negative, got {negative}")

    n_pairs = both + only_a + only_b + neither
    n_discordant = only_a + only_b
    if n_discordant == 0:
        return McNemarResult(
            p_value=1.0,
            discordant_a=only_a,
            discordant_b=only_b,
            n_pairs=n_pairs,
            statistic=None,
        )

    smaller = min(only_a, only_b)
    # Sum of binomial tail probabilities at p = 1/2. math.comb is exact integer
    # arithmetic, so the only rounding is the single final division.
    tail_weight = sum(math.comb(n_discordant, k) for k in range(smaller + 1))
    p_value = min(1.0, 2.0 * tail_weight / (2**n_discordant))
    statistic = (only_a - only_b) ** 2 / n_discordant
    return McNemarResult(
        p_value=p_value,
        discordant_a=only_a,
        discordant_b=only_b,
        n_pairs=n_pairs,
        statistic=statistic,
    )


# ---------------------------------------------------------------------------
# Building the paired table from two runs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PairedCounts:
    """The 2x2 paired table, plus the rates it implies.

    The rates are derived properties rather than stored fields so a reader can
    verify ``rate_a`` against the four cells without trusting that whoever built
    the object did the arithmetic.
    """

    both: int
    only_a: int
    only_b: int
    neither: int

    @property
    def n_pairs(self) -> int:
        return self.both + self.only_a + self.only_b + self.neither

    @property
    def successes_a(self) -> int:
        return self.both + self.only_a

    @property
    def successes_b(self) -> int:
        return self.both + self.only_b

    @property
    def rate_a(self) -> float | None:
        return None if self.n_pairs == 0 else self.successes_a / self.n_pairs

    @property
    def rate_b(self) -> float | None:
        return None if self.n_pairs == 0 else self.successes_b / self.n_pairs

    @property
    def difference(self) -> float | None:
        """``rate_b - rate_a``: signed, condition B minus condition A.

        With A the baseline and B the defended run, a NEGATIVE difference means
        the defended run scored lower. The sign convention is fixed here so the
        renderer and the bootstrap cannot disagree about which way is which.
        """
        a, b = self.rate_a, self.rate_b
        return None if a is None or b is None else b - a


def _require_same_keys(
    results_a: Mapping[str, bool],
    results_b: Mapping[str, bool],
    *,
    label_a: str = "A",
    label_b: str = "B",
    sample: int = 3,
) -> tuple[str, ...]:
    """Assert two result dicts describe the same tasks, or raise naming names.

    WHY THIS REFUSES INSTEAD OF INTERSECTING
        Silently pairing on the intersection is the single most dangerous thing
        this module could do. If the defended run skipped four couples - crashed,
        rate-limited, ran with a smaller ``--max-injection-tasks`` - intersecting
        would compare 8 couples against 8 different-but-overlapping couples and
        report a confident p-value for a comparison nobody ran. Refusing costs the
        caller one obvious error message; intersecting costs the reader their
        trust in every number in the file.

    Returns:
        The shared keys in sorted order, so downstream iteration is deterministic.
    """
    keys_a, keys_b = set(results_a), set(results_b)
    if keys_a == keys_b:
        return tuple(sorted(keys_a))

    missing_from_b = sorted(keys_a - keys_b)
    missing_from_a = sorted(keys_b - keys_a)
    parts = [
        f"cannot pair {label_a} ({len(keys_a)} keys) with {label_b} ({len(keys_b)} keys): "
        "the two runs do not describe the same tasks, and pairing on the "
        "intersection would compare different task sets"
    ]
    for label, missing in ((label_a, missing_from_b), (label_b, missing_from_a)):
        if not missing:
            continue
        shown = ", ".join(repr(key) for key in missing[:sample])
        more = f" (+{len(missing) - sample} more)" if len(missing) > sample else ""
        parts.append(f"only in {label}: {shown}{more}")
    raise PairingError("; ".join(parts))


def paired_outcomes(
    results_a: Mapping[str, bool],
    results_b: Mapping[str, bool],
    *,
    label_a: str = "A",
    label_b: str = "B",
) -> PairedCounts:
    """Build the 2x2 paired table from two per-task boolean dicts.

    Both dicts must be keyed the way the runner keys them - ``user_task_0::`` for
    clean results, ``user_task_0::injection_task_1`` for injected couples - and
    must carry EXACTLY the same key set. See :func:`_require_same_keys` for why a
    mismatch is a hard error rather than an intersection.

    Args:
        results_a: Condition A (conventionally the baseline).
        results_b: Condition B (conventionally the defended run).
        label_a: Name for A in error messages.
        label_b: Name for B in error messages.

    Returns:
        The :class:`PairedCounts` table.

    Raises:
        PairingError: If the key sets differ.
    """
    keys = _require_same_keys(results_a, results_b, label_a=label_a, label_b=label_b)
    both = only_a = only_b = neither = 0
    for key in keys:
        a, b = bool(results_a[key]), bool(results_b[key])
        if a and b:
            both += 1
        elif a:
            only_a += 1
        elif b:
            only_b += 1
        else:
            neither += 1
    return PairedCounts(both=both, only_a=only_a, only_b=only_b, neither=neither)


# ---------------------------------------------------------------------------
# Cluster bootstrap on the paired difference
# ---------------------------------------------------------------------------


def cluster_of(key: str) -> str:
    """The cluster a result key belongs to: its user task.

    AgentDojo runs every (user task, injection task) couple, so the twelve couples
    in the Week-0 baseline are really three user tasks seen four times each. The
    user task is the unit that varies independently; the four couples sharing one
    are not independent draws of anything.
    """
    return key.split(KEY_JOIN, 1)[0]


@dataclass(frozen=True, slots=True)
class PairedObservation:
    """One task observed under both conditions, tagged with its cluster.

    ``cluster`` is carried explicitly rather than re-derived inside the bootstrap
    so a caller with a different clustering (by suite, by injection task) can
    supply it without the resampler needing to know how keys are spelled.
    """

    cluster: str
    key: str
    outcome_a: bool
    outcome_b: bool


@dataclass(frozen=True, slots=True)
class DifferenceInterval:
    """A bootstrap interval on ``rate_b - rate_a``, with its provenance.

    ``seed`` and ``n_resamples`` are fields because a bootstrap interval is a
    function of them: quoting the interval without them quotes a number nobody
    else can reproduce.
    """

    low: float
    high: float
    point: float | None
    n_clusters: int
    n_observations: int
    n_resamples: int
    seed: int
    confidence: float

    @property
    def width(self) -> float:
        return self.high - self.low

    def as_points_str(self) -> str:
        """The difference in signed percentage POINTS, with its interval."""
        point = "n/a" if self.point is None else f"{self.point * 100:+.1f} pp"
        return (
            f"{point}  {self.confidence * 100:.0f}% CI "
            f"[{self.low * 100:+.1f}, {self.high * 100:+.1f}] pp"
        )


def observations_from_results(
    results_a: Mapping[str, bool],
    results_b: Mapping[str, bool],
    *,
    label_a: str = "A",
    label_b: str = "B",
) -> tuple[PairedObservation, ...]:
    """Zip two runner result dicts into clustered paired observations.

    Applies the same key-set check as :func:`paired_outcomes` - the bootstrap is
    no less capable of confidently comparing the wrong things.
    """
    keys = _require_same_keys(results_a, results_b, label_a=label_a, label_b=label_b)
    return tuple(
        PairedObservation(
            cluster=cluster_of(key),
            key=key,
            outcome_a=bool(results_a[key]),
            outcome_b=bool(results_b[key]),
        )
        for key in keys
    )


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile of an already-sorted sequence.

    The ordinary "type 7" definition (numpy's default): position ``q*(n-1)``,
    interpolating between the two neighbouring order statistics. Written out
    rather than imported so the interval endpoints stay as checkable as the rest.
    """
    n = len(sorted_values)
    if n == 0:
        raise StatsError("cannot take a percentile of an empty sample")
    if n == 1:
        return sorted_values[0]
    position = q * (n - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[int(position)]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _difference(pairs: Sequence[PairedObservation]) -> float | None:
    """``rate_b - rate_a`` over a bag of observations, or None if the bag is empty."""
    if not pairs:
        return None
    n = len(pairs)
    return sum(p.outcome_b for p in pairs) / n - sum(p.outcome_a for p in pairs) / n


def cluster_bootstrap_difference(
    pairs: Sequence[PairedObservation],
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
) -> DifferenceInterval:
    """Percentile bootstrap CI for ``rate_b - rate_a``, resampling BY CLUSTER.

    WHY CLUSTERS AND NOT COUPLES
        The twelve injected couples in the Week-0 baseline are three user tasks
        crossed with four injection tasks. Whether the agent gets hijacked depends
        heavily on the user task - which tools it reaches for, how much attacker
        text lands in its context - so the four couples sharing a user task rise
        and fall together. Resampling couples independently would treat twelve
        correlated observations as twelve independent ones and shrink the interval
        by roughly the square root of the cluster size, producing a tight interval
        around a number measured on three tasks. Resampling whole USER TASKS
        (every couple of a drawn task comes along, and a task may be drawn more
        than once) keeps the within-cluster correlation intact, so the interval
        reflects the three-task sample it actually came from. The tests construct
        perfectly-correlated clusters and assert the clustered interval is
        materially the wider of the two.

    WHY PERCENTILE AND NOT BCa
        Bias-corrected-and-accelerated intervals need a jackknife acceleration
        estimate that is itself unstable at three clusters. The plain percentile
        interval is what the data can support, and it is one readable loop.

    WHY THE SEED IS AN ARGUMENT
        A bootstrap interval is a random variable. Fixing ``random.Random(seed)``
        makes the reported endpoints a deterministic function of the data, so the
        README's numbers can be regenerated by anyone and a test can assert two
        calls agree exactly. The seed is recorded in the returned object for the
        same reason.

    Args:
        pairs: The paired observations, each tagged with its cluster.
        n_resamples: Bootstrap replicates. Must be positive.
        seed: Seed for :class:`random.Random`, recorded in the result.
        confidence: Two-sided level for the percentile endpoints.

    Returns:
        A :class:`DifferenceInterval`. With no observations it returns
        ``(-1.0, 1.0)`` with ``point=None`` - the full range a difference of rates
        can take, which is the honest statement that nothing constrains it.

    Raises:
        StatsError: On a non-positive ``n_resamples`` or a confidence outside
            ``(0, 1)``.
    """
    if n_resamples < 1:
        raise StatsError(f"n_resamples must be >= 1, got {n_resamples}")
    if not 0.0 < confidence < 1.0:
        raise StatsError(f"confidence must be strictly between 0 and 1, got {confidence!r}")

    if not pairs:
        return DifferenceInterval(
            low=-1.0,
            high=1.0,
            point=None,
            n_clusters=0,
            n_observations=0,
            n_resamples=n_resamples,
            seed=seed,
            confidence=confidence,
        )

    # Cluster membership in first-seen order, so the resampling indices mean the
    # same thing across runs regardless of any dict-ordering detail upstream.
    grouped: dict[str, list[PairedObservation]] = {}
    for pair in pairs:
        grouped.setdefault(pair.cluster, []).append(pair)
    clusters = list(grouped.values())
    n_clusters = len(clusters)

    rng = random.Random(seed)
    replicates: list[float] = []
    for _ in range(n_resamples):
        drawn: list[PairedObservation] = []
        for _ in range(n_clusters):
            drawn.extend(clusters[rng.randrange(n_clusters)])
        replicate = _difference(drawn)
        if replicate is not None:
            replicates.append(replicate)
    replicates.sort()

    alpha = 1.0 - confidence
    return DifferenceInterval(
        low=_percentile(replicates, alpha / 2.0),
        high=_percentile(replicates, 1.0 - alpha / 2.0),
        point=_difference(pairs),
        n_clusters=n_clusters,
        n_observations=len(pairs),
        n_resamples=n_resamples,
        seed=seed,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Reading runner result files
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunSummary:
    """One run's three rates with their Wilson intervals, plus the raw booleans.

    The raw per-task dicts are kept because the interesting comparison is paired,
    and a paired comparison needs the per-task outcomes rather than the aggregate.
    A summary that dropped them could only support the misleading kind of report.
    """

    label: str
    source: str
    intervals: Mapping[Metric, Interval]
    results: Mapping[Metric, Mapping[str, bool]]
    config: Mapping[str, str]

    def interval(self, metric: Metric) -> Interval:
        return self.intervals[metric]

    def results_for(self, metric: Metric) -> Mapping[str, bool]:
        return self.results[metric]


def _dig(payload: Mapping[str, Any], path: Sequence[str], origin: str) -> Mapping[str, Any]:
    """Walk a nested mapping, raising :class:`SchemaError` naming the failed path."""
    node: Any = payload
    for index, part in enumerate(path):
        if not isinstance(node, Mapping) or part not in node:
            missing = ".".join(path[: index + 1])
            raise SchemaError(
                f"{origin}: missing '{missing}'. This does not look like a result file "
                "written by evals.agentdojo.runner, which records "
                "raw.clean.utility_results, raw.injected.security_results and "
                "raw.injected.utility_results. Refusing to guess - re-run the "
                "benchmark to produce a current-schema file."
            )
        node = node[part]
    if not isinstance(node, Mapping):
        raise SchemaError(
            f"{origin}: '{'.'.join(path)}' is {type(node).__name__}, expected an object of "
            "per-task booleans"
        )
    return node


def _as_bool_results(block: Mapping[str, Any], origin: str, where: str) -> dict[str, bool]:
    """Coerce a raw per-task block to ``dict[str, bool]``, refusing odd values.

    AgentDojo's outcomes arrive as JSON ``true``/``false``. Anything else (a
    string, a null from a crashed task) means the run did not complete the way the
    schema claims, and averaging it would invent a number.
    """
    out: dict[str, bool] = {}
    for key, value in block.items():
        if not isinstance(value, bool):
            raise SchemaError(
                f"{origin}: {where}[{key!r}] is {value!r} ({type(value).__name__}), expected a "
                "boolean - a non-boolean outcome cannot be averaged honestly"
            )
        out[str(key)] = value
    return out


# Recorded alongside the numbers because a rate without the configuration that
# produced it cannot be compared to anything.
_CONFIG_FIELDS = ("timestamp", "model", "suite", "attack", "defense", "benchmark_version")


def _check_headline(
    payload: Mapping[str, Any],
    metric: Metric,
    interval: Interval,
    origin: str,
) -> None:
    """Cross-check a recomputed rate against the file's own headline field.

    A file whose ``asr`` disagrees with the mean of its own
    ``raw.injected.security_results`` has been edited or truncated. Both numbers
    then have equal claim to being right, which means neither is, so the file is
    refused rather than summarised.
    """
    recorded = payload.get(str(metric))
    if recorded is None:
        return
    if isinstance(recorded, bool) or not isinstance(recorded, int | float):
        raise SchemaError(f"{origin}: '{metric}' is {recorded!r}, expected a number or null")
    if interval.point is None:
        raise SchemaError(
            f"{origin}: '{metric}' records {recorded!r} but the raw block behind it is empty"
        )
    if not math.isclose(float(recorded), interval.point, abs_tol=1e-9):
        raise SchemaError(
            f"{origin}: '{metric}' records {recorded!r} but its own raw block averages "
            f"{interval.point!r} ({interval.successes}/{interval.trials}). The headline number "
            "and the per-task outcomes disagree, so neither can be trusted."
        )


def summarize_run(
    source: str | Path | Mapping[str, Any],
    label: str | None = None,
    confidence: float = DEFAULT_CONFIDENCE,
) -> RunSummary:
    """Read a runner result (path or already-parsed mapping) into a summary.

    Every rate is RECOMPUTED from the raw per-task booleans rather than taken from
    the file's headline field, and then checked against that field. The whole
    value of this module is that its numbers trace back to per-task outcomes; a
    summary that trusted the headline would be repeating a claim rather than
    checking one.

    Args:
        source: Path to a runner JSON file, or the parsed mapping.
        label: Name for the run in rendered output. Defaults to the file stem.
        confidence: Level for the Wilson intervals.

    Returns:
        The :class:`RunSummary`.

    Raises:
        SchemaError: If the file is missing, unreadable, missing the runner's
            ``raw`` block, holding non-boolean outcomes, or contradicting its own
            headline numbers.
    """
    if isinstance(source, str | Path):
        path = Path(source)
        origin = str(path)
        if not path.is_file():
            raise SchemaError(f"{origin}: no such result file")
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SchemaError(f"{origin}: not valid JSON ({exc})") from exc
        if not isinstance(parsed, Mapping):
            raise SchemaError(f"{origin}: top level is {type(parsed).__name__}, expected an object")
        payload: Mapping[str, Any] = parsed
        resolved_label = label or path.stem
    else:
        payload = source
        origin = "<mapping>"
        resolved_label = label or "run"

    intervals: dict[Metric, Interval] = {}
    results: dict[Metric, Mapping[str, bool]] = {}
    for metric, (phase, field) in _METRIC_SOURCE.items():
        outcomes = _as_bool_results(
            _dig(payload, ("raw", phase, field), origin),
            origin,
            f"raw.{phase}.{field}",
        )
        interval = wilson_interval(sum(outcomes.values()), len(outcomes), confidence=confidence)
        _check_headline(payload, metric, interval, origin)
        intervals[metric] = interval
        results[metric] = outcomes

    config = {field: str(payload[field]) for field in _CONFIG_FIELDS if field in payload}
    return RunSummary(
        label=resolved_label,
        source=origin,
        intervals=intervals,
        results=results,
        config=config,
    )


# ---------------------------------------------------------------------------
# Comparing two runs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MetricComparison:
    """Everything needed to state a before/after claim about one metric honestly."""

    metric: Metric
    baseline: Interval
    defended: Interval
    counts: PairedCounts
    mcnemar: McNemarResult
    bootstrap: DifferenceInterval


@dataclass(frozen=True, slots=True)
class ComparisonReport:
    """A paired comparison of two runs across all three metrics."""

    baseline: RunSummary
    defended: RunSummary
    comparisons: Mapping[Metric, MetricComparison]


def compare_runs(
    baseline: RunSummary,
    defended: RunSummary,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
) -> ComparisonReport:
    """Pair two summaries task-by-task and test each metric.

    The baseline is condition A and the defended run is condition B throughout, so
    a negative difference always means "the defended run scored lower". For ASR
    that is the good direction and for utility it is the bad one; the renderer
    says which, and this function deliberately encodes no preference - which
    direction counts as an improvement is a claim about intent, and the arithmetic
    should not have an opinion about it.

    Raises:
        PairingError: If any metric's key sets differ between the two runs.
    """
    comparisons: dict[Metric, MetricComparison] = {}
    for metric in Metric:
        results_a = baseline.results_for(metric)
        results_b = defended.results_for(metric)
        label_a = f"{baseline.label}.{metric}"
        label_b = f"{defended.label}.{metric}"
        counts = paired_outcomes(results_a, results_b, label_a=label_a, label_b=label_b)
        observations = observations_from_results(
            results_a, results_b, label_a=label_a, label_b=label_b
        )
        comparisons[metric] = MetricComparison(
            metric=metric,
            baseline=baseline.interval(metric),
            defended=defended.interval(metric),
            counts=counts,
            mcnemar=mcnemar_exact(
                both=counts.both,
                only_a=counts.only_a,
                only_b=counts.only_b,
                neither=counts.neither,
            ),
            bootstrap=cluster_bootstrap_difference(
                observations,
                n_resamples=n_resamples,
                seed=seed,
                confidence=confidence,
            ),
        )
    return ComparisonReport(baseline=baseline, defended=defended, comparisons=comparisons)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_WILSON_NOTE = (
    "Wilson, not Wald: the normal-approximation interval has ZERO width at 0/n,",
    "which would let a defended run claim perfection from a dozen trials.",
)

_LABEL_WIDTH = 21
# Wide enough for the longest configuration field name the runner records
# ('benchmark_version'), so the colons line up instead of one row jutting out.
_CONFIG_WIDTH = 17


def _change_line(metric: Metric, bootstrap: DifferenceInterval) -> str:
    """The one and only place a change is rendered.

    THE INVARIANT
        A percentage change never appears without the sample size and the interval
        on the SAME line. "the defense reduced ASR by 8.3 points" is a sentence a
        reader will quote out of context; "-8.3 pp  95% CI [-25.0, +0.0] pp
        n=12 pairs / 3 clusters" is not. Because every change claim is built here,
        a test can assert the invariant over the rendered string and have it hold
        for the whole report.
    """
    return (
        f"   {'change (def-base)':<{_LABEL_WIDTH}} {bootstrap.as_points_str()}  "
        f"n={bootstrap.n_observations} pairs / {bootstrap.n_clusters} clusters"
    )


def render_summary(summary: RunSummary) -> str:
    """An ASCII table of one run's rates with their Wilson intervals."""
    lines = [
        "=" * _RULE_WIDTH,
        f" Aegis eval statistics: {summary.label}",
        "=" * _RULE_WIDTH,
        f" {'source':<{_CONFIG_WIDTH}}: {summary.source}",
    ]
    lines.extend(f" {field:<{_CONFIG_WIDTH}}: {value}" for field, value in summary.config.items())
    lines.extend(
        [
            "-" * _RULE_WIDTH,
            f" {'metric':<{_LABEL_WIDTH}} {'rate':>7} {'n':>7}   interval",
            "-" * _RULE_WIDTH,
        ]
    )
    for metric in Metric:
        interval = summary.interval(metric)
        rate = "    n/a" if interval.point is None else f"{interval.point * 100:6.1f}%"
        fraction = f"{interval.successes}/{interval.trials}"
        band = (
            f"{interval.confidence * 100:.0f}% CI "
            f"[{interval.low * 100:5.1f}%, {interval.high * 100:5.1f}%]"
        )
        lines.append(f" {metric!s:<{_LABEL_WIDTH}} {rate} {fraction:>7}   {band}")
    lines.append("-" * _RULE_WIDTH)
    lines.extend(f" {note}" for note in _WILSON_NOTE)
    lines.append("=" * _RULE_WIDTH)
    return "\n".join(lines)


def render_comparison(report: ComparisonReport) -> str:
    """An ASCII table of the paired comparison, safe to paste into a README.

    Each metric gets the two point estimates with their intervals, the change with
    its bootstrap CI and pair count on one line, the discordant counts, and the
    exact McNemar p-value. When the discordant count is too small for the test to
    have been ABLE to reach significance, that is said explicitly: the difference
    between "no effect was found" and "no effect could have been found" is the
    difference between a result and a press release.
    """
    lines = [
        "=" * _RULE_WIDTH,
        f" Paired comparison: {report.baseline.label} -> {report.defended.label}",
        "=" * _RULE_WIDTH,
        f" baseline : {report.baseline.source}",
        f" defended : {report.defended.source}",
        "-" * _RULE_WIDTH,
        " changes are DEFENDED minus BASELINE in percentage points: a negative asr",
        " change is the defense working, a negative utility change is it costing",
        " something. Paired throughout - the same tasks under both conditions.",
        "-" * _RULE_WIDTH,
    ]
    for metric in Metric:
        comparison = report.comparisons[metric]
        mcnemar = comparison.mcnemar
        chi2 = "" if mcnemar.statistic is None else f"   (chi2 = {mcnemar.statistic:.3f})"
        lines.extend(
            [
                f" {metric}",
                f"   {'baseline':<{_LABEL_WIDTH}} {comparison.baseline.as_percent_str()}",
                f"   {'defended':<{_LABEL_WIDTH}} {comparison.defended.as_percent_str()}",
                _change_line(metric, comparison.bootstrap),
                f"   {'discordant pairs':<{_LABEL_WIDTH}} baseline-only "
                f"{mcnemar.discordant_a}, defended-only {mcnemar.discordant_b} "
                "(only these inform)",
                f"   {'McNemar exact p':<{_LABEL_WIDTH}} {mcnemar.p_value:.4f}{chi2}",
            ]
        )
        if mcnemar.underpowered:
            lines.extend(
                [
                    f"   {'POWER':<{_LABEL_WIDTH}} {mcnemar.n_discordant} discordant pair(s); "
                    f"fewer than {MIN_DISCORDANT_FOR_SIGNIFICANCE} can never",
                    f"   {'':<{_LABEL_WIDTH}} reach p < 0.05, so this test could not have "
                    "shown an effect.",
                ]
            )
        lines.append("-" * _RULE_WIDTH)
    bootstrap = report.comparisons[Metric.ASR].bootstrap
    lines.extend(
        [
            f" bootstrap: {bootstrap.n_resamples} resamples drawn BY USER TASK (the cluster),",
            f" seed {bootstrap.seed}. Couples sharing a user task are correlated, so",
            " resampling couples independently would report an interval narrower",
            " than the evidence behind it.",
            "=" * _RULE_WIDTH,
        ]
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_EPILOG = """\
reads the JSON evals.agentdojo.runner writes: the three headline rates plus the
raw.clean.utility_results / raw.injected.security_results /
raw.injected.utility_results blocks of per-task booleans, keyed
'user_task_0::injection_task_1'. Every rate is recomputed from those booleans and
checked against the headline field, so a hand-edited or old-schema file is refused
rather than misread.

  python -m evals.stats.analysis --baseline results/week0_baseline.json

with --defended, adds the paired comparison: the exact McNemar test on the
discordant pairs and a cluster bootstrap CI on the difference. Both runs must
cover exactly the same tasks - a partial re-run cannot be paired against a full
one, and this refuses instead of silently intersecting.
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evals.stats.analysis",
        description=(
            "Report eval rates with confidence intervals, and compare two runs with a paired test."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="runner result JSON to summarise (e.g. results/week0_baseline.json)",
    )
    parser.add_argument(
        "--defended",
        type=Path,
        default=None,
        help="second runner result JSON; adds the paired comparison against --baseline",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help=f"two-sided confidence level (default: {DEFAULT_CONFIDENCE})",
    )
    parser.add_argument(
        "--resamples",
        type=int,
        default=DEFAULT_RESAMPLES,
        help=f"bootstrap replicates (default: {DEFAULT_RESAMPLES})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"bootstrap seed, reported with the interval (default: {DEFAULT_SEED})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        baseline = summarize_run(args.baseline, label="baseline", confidence=args.confidence)
        print(render_summary(baseline))
        if args.defended is not None:
            defended = summarize_run(args.defended, label="defended", confidence=args.confidence)
            print()
            print(render_summary(defended))
            print()
            print(
                render_comparison(
                    compare_runs(
                        baseline,
                        defended,
                        n_resamples=args.resamples,
                        seed=args.seed,
                        confidence=args.confidence,
                    )
                )
            )
    except StatsError as exc:
        # A refusal here is the module working: it declined to produce a number
        # that would have been wrong. Report it as a clean CLI error, not a
        # traceback that reads like a crash.
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
