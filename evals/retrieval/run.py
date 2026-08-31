"""The retrieval ablation, as one command.

WHAT THIS DOES
--------------
Ingests the committed golden set once, queries it under three
:class:`~aegis.retrieval.retriever.RetrievalConfig` arms - ``bm25``, ``vector``,
``hybrid`` (``hybrid+rerank`` is deliberately absent; see ``_arm_configs``) - scores
each with
:mod:`evals.retrieval.metrics`, and prints one table::

    uv run python -m evals.retrieval.run

Offline, no key, no download, well under a second.

ONE CORPUS, THREE ARMS
----------------------
The corpus is built once and every arm is a
:meth:`HybridRetriever.with_config <aegis.retrieval.retriever.HybridRetriever.with_config>`
view over the *same* indexes. That is not an optimisation: two arms compared
across two ingests differ by more than the arm, and re-ingesting per arm would
re-fit the TF-IDF vocabulary and silently make the comparison a comparison of two
corpora. ``retriever.py`` was built to allow exactly this, and this module is the
first caller that depends on it.

WHICH NUMBERS GET AN INTERVAL, AND WHICH DO NOT
-----------------------------------------------
``hit@k`` is one Bernoulli trial per query - the query either found something
relevant in the top k or it did not - so it gets the Wilson interval from
:func:`evals.stats.analysis.wilson_interval`, imported rather than reimplemented.
There is exactly one confidence-interval implementation in this repository and
this module is not going to become the second.

``recall@k``, ``precision@k``, ``mrr@k`` and ``ndcg@k`` are macro-averages of
bounded continuous scores, **not** proportions. A binomial interval around them
would be arithmetic applied to the wrong distribution, so none is offered. They
are printed as means with the query count beside them and nothing more.

The arm-versus-arm comparison uses :func:`evals.stats.analysis.mcnemar_exact` on
``hit@k``, which is legitimate because the arms answer the *same* queries - the
paired design McNemar is for. It carries the same power warning the AgentDojo
comparison does, and at twenty-five queries that warning fires often: a fixture
this size can show a large difference, and cannot show a small one.

WHAT THIS CANNOT TELL YOU
-------------------------
The golden set is a fixture its own authors wrote (see
:mod:`evals.retrieval.golden_set`). The table below it is a working-pipeline
check and an arm-versus-arm sanity comparison. It is not a benchmark result, and
the CLI prints that line under every table so it cannot be pasted anywhere
without it.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from aegis.ingest.pipeline import IngestReport
from aegis.retrieval.retriever import HybridRetriever, RetrievalConfig
from evals.retrieval.golden_set import (
    DEFAULT_GOLDEN_SET_PATH,
    GoldenSet,
    GoldenSetError,
    build_corpus,
    chunk_key,
    load_golden_set,
)
from evals.retrieval.metrics import (
    MetricSummary,
    QueryScore,
    RetrievalMetric,
    score_query,
    summarize,
)
from evals.stats.analysis import (
    DEFAULT_CONFIDENCE,
    MIN_DISCORDANT_FOR_SIGNIFICANCE,
    Interval,
    McNemarResult,
    mcnemar_exact,
    wilson_interval,
)

__all__ = [
    "DEFAULT_K",
    "AblationReport",
    "ArmReport",
    "evaluate",
    "evaluate_arm",
    "main",
    "render_ablation",
]

DEFAULT_K = 5
"""Cutoff the table reports at.

Five rather than ten because the corpus is 44 chunks: at k=10 nearly a quarter of
the corpus is inside the window, every arm scores a high hit rate, and the arms
stop being distinguishable. A cutoff should be a plausible prompt budget, and
five chunks is one.
"""


def _arm_configs(k: int, candidate_k: int) -> tuple[RetrievalConfig, ...]:
    """The three ablation arms at one cutoff, in reporting order.

    Spelled out as a tuple of named constructors rather than generated from the
    toggles, so a reader can see that the arms differ in nothing but which stages
    run - and so ``bm25`` is unambiguously first, because it is the baseline every
    other row is compared against.

    WHY hybrid+rerank IS NOT HERE
    -----------------------------
    :mod:`aegis.retrieval.rerank` states the rule and this harness has to keep it:
    while :class:`IdentityReranker` is the only implementation, ``hybrid+rerank``
    is the same system as ``hybrid``, so a row for it would be a row for nothing.
    It previously produced a fourth line in the table, a fourth confidence
    interval, and a McNemar block reading ``p = 1.0000, discordant 0 / 0`` - an
    artifact quotable as "reranking tested, no difference" when nothing had been
    tested. The seam is still covered, by an equality test that constructs the
    reranked config directly; that is the right place for it, because what is
    being checked is that the seam cannot manufacture a gap, not how it scores.
    """
    return (
        RetrievalConfig.sparse_only(top_k=k, candidate_k=candidate_k),
        RetrievalConfig.vector_only(top_k=k, candidate_k=candidate_k),
        RetrievalConfig.hybrid(top_k=k, candidate_k=candidate_k),
    )


@dataclass(frozen=True, slots=True)
class ArmReport:
    """One arm's scores over the whole golden set."""

    arm: str
    k: int
    scores: tuple[QueryScore, ...]
    summaries: Mapping[RetrievalMetric, MetricSummary]
    hit_interval: Interval

    def summary(self, metric: RetrievalMetric) -> MetricSummary:
        return self.summaries[metric]

    def hits(self) -> dict[str, bool]:
        """Per-query hit outcomes, keyed by query id - the paired input to McNemar."""
        return {score.query_id: bool(score.hit) for score in self.scores}


@dataclass(frozen=True, slots=True)
class AblationReport:
    """Every arm over one golden set, plus what the corpus turned out to be."""

    golden: GoldenSet
    ingest: IngestReport
    k: int
    confidence: float
    arms: tuple[ArmReport, ...]

    @property
    def baseline(self) -> ArmReport:
        """The first arm. :func:`_arm_configs` puts ``bm25`` there deliberately."""
        return self.arms[0]


def evaluate_arm(
    retriever: HybridRetriever,
    golden: GoldenSet,
    config: RetrievalConfig,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
) -> ArmReport:
    """Run every golden query under one arm and score the rankings.

    The retriever is re-viewed through ``config`` rather than rebuilt; see the
    module docstring on why that is load-bearing rather than tidy.
    """
    arm = retriever.with_config(config)
    k = config.top_k
    scores = tuple(
        score_query(
            query.query_id,
            [chunk_key(result.chunk) for result in arm.retrieve(query.text)],
            query.relevant,
            k,
        )
        for query in golden.queries
    )
    summaries = {
        metric: summarize(metric, (score.value(metric) for score in scores))
        for metric in RetrievalMetric
    }
    hit = summaries[RetrievalMetric.HIT]
    return ArmReport(
        arm=config.arm,
        k=k,
        scores=scores,
        summaries=summaries,
        hit_interval=wilson_interval(
            successes=sum(1 for score in scores if score.hit),
            trials=hit.n_scored,
            confidence=confidence,
        ),
    )


def evaluate(
    golden: GoldenSet,
    *,
    k: int = DEFAULT_K,
    candidate_k: int | None = None,
    confidence: float = DEFAULT_CONFIDENCE,
) -> AblationReport:
    """Build the corpus once and score all three arms against it."""
    if k <= 0:
        raise GoldenSetError(f"k must be positive, got {k}")
    retriever, ingest = build_corpus(golden)
    depth = candidate_k if candidate_k is not None else max(50, k)
    return AblationReport(
        golden=golden,
        ingest=ingest,
        k=k,
        confidence=confidence,
        arms=tuple(
            evaluate_arm(retriever, golden, config, confidence=confidence)
            for config in _arm_configs(k, depth)
        ),
    )


# ---------------------------------------------------------------------------
# Comparison against the baseline arm
# ---------------------------------------------------------------------------


def paired_hit_test(baseline: ArmReport, other: ArmReport) -> McNemarResult:
    """Exact McNemar on ``hit@k``, baseline versus another arm, over the same queries.

    ``only_a`` is "the baseline found something relevant and this arm did not",
    which is the cell that has to be non-empty before an arm can be said to have
    cost anything.
    """
    left, right = baseline.hits(), other.hits()
    if set(left) != set(right):  # pragma: no cover - both arms read one golden set
        raise GoldenSetError("arms were scored over different query sets and cannot be paired")
    both = sum(1 for key in left if left[key] and right[key])
    only_a = sum(1 for key in left if left[key] and not right[key])
    only_b = sum(1 for key in left if not left[key] and right[key])
    neither = sum(1 for key in left if not left[key] and not right[key])
    return mcnemar_exact(both=both, only_a=only_a, only_b=only_b, neither=neither)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_RULE_WIDTH = 78
_ARM_WIDTH = 15

_CAVEAT = (
    "FIXTURE, NOT A BENCHMARK: this golden set was written by the authors of the",
    "retriever it grades, and is 25 queries over 44 chunks. It shows that the",
    "pipeline works and how the arms differ HERE. It is not evidence about how",
    "this system would rank real traffic - BEIR or MS MARCO would be, and neither",
    "runs offline in CI. Do not quote a number from this table without this line.",
)


def _cell(summary: MetricSummary) -> str:
    return "    n/a" if summary.mean is None else f"{summary.mean:7.3f}"


def _delta(baseline: MetricSummary, other: MetricSummary) -> str:
    if baseline.mean is None or other.mean is None:
        return "    n/a"
    return f"{other.mean - baseline.mean:+7.3f}"


def render_ablation(report: AblationReport) -> str:
    """An ASCII table of every arm, safe to paste into a README.

    Every mean is printed with the query count it is over, every ``hit@k`` with
    its Wilson interval and its fraction, and every arm-versus-baseline row with
    the exact McNemar p-value and - when it applies - the statement that the
    comparison could not have reached significance. The caveat block is not
    optional and is not conditional.
    """
    tiers = ", ".join(
        f"{tier}={count}" for tier, count in sorted(report.ingest.chunks_by_tier.items())
    )
    scored = report.baseline.summary(RetrievalMetric.HIT)
    lines = [
        "=" * _RULE_WIDTH,
        f" Retrieval ablation: {report.golden.name}",
        "=" * _RULE_WIDTH,
        f" golden set : {report.golden.source}",
        f" corpus     : {report.ingest.documents_ingested} documents, "
        f"{report.ingest.chunks_added} chunks ({tiers})",
        f" queries    : {scored.n_queries} ({scored.n_scored} scored, "
        f"{scored.n_undefined} undefined)",
        f" cutoff     : k = {report.k}",
        "-" * _RULE_WIDTH,
        f" {'arm':<{_ARM_WIDTH}}" + "".join(f"{metric!s:>13}" for metric in RetrievalMetric),
        "-" * _RULE_WIDTH,
    ]
    for arm in report.arms:
        cells = "".join(f"{_cell(arm.summary(metric)):>13}" for metric in RetrievalMetric)
        lines.append(f" {arm.arm:<{_ARM_WIDTH}}{cells}")
    lines.extend(
        [
            "-" * _RULE_WIDTH,
            " macro-averages over queries, each query weighted equally. Only hit@k is a",
            " per-query Bernoulli trial, so only hit@k gets an interval:",
        ]
    )
    for arm in report.arms:
        lines.append(f"   {arm.arm:<{_ARM_WIDTH}} {arm.hit_interval.as_percent_str()}")

    baseline = report.baseline
    lines.extend(
        [
            "-" * _RULE_WIDTH,
            f" versus the {baseline.arm} baseline, on the same {scored.n_scored} queries.",
            " deltas are differences of MEANS, not tested; the p-value tests hit@k only.",
        ]
    )
    for arm in report.arms[1:]:
        deltas = "".join(
            f"{_delta(baseline.summary(metric), arm.summary(metric)):>13}"
            for metric in RetrievalMetric
        )
        test = paired_hit_test(baseline, arm)
        lines.append(f" {arm.arm:<{_ARM_WIDTH}}{deltas}")
        lines.append(
            f"   {'hit@k McNemar':<{_ARM_WIDTH}} p = {test.p_value:.4f}, discordant "
            f"{baseline.arm}-only {test.discordant_a} / {arm.arm}-only {test.discordant_b}"
        )
        if test.underpowered:
            lines.append(
                f"   {'POWER':<{_ARM_WIDTH}} {test.n_discordant} discordant pair(s); fewer "
                f"than {MIN_DISCORDANT_FOR_SIGNIFICANCE} can never reach p < 0.05."
            )
    lines.append("-" * _RULE_WIDTH)
    lines.extend(f" {line}" for line in _CAVEAT)
    lines.append("=" * _RULE_WIDTH)
    return "\n".join(lines)


def render_per_query(report: AblationReport) -> str:
    """Per-query hit/miss for every arm - where a table of means hides the story."""
    lines = [
        "=" * _RULE_WIDTH,
        f" Per-query hit@{report.k} (1 = at least one relevant chunk in the top {report.k})",
        "=" * _RULE_WIDTH,
        f" {'query':<8}{'rel':>4}  " + "".join(f"{arm.arm:>16}" for arm in report.arms),
        "-" * _RULE_WIDTH,
    ]
    by_arm = {arm.arm: {score.query_id: score for score in arm.scores} for arm in report.arms}
    for query in report.golden.queries:
        cells = ""
        for arm in report.arms:
            score = by_arm[arm.arm][query.query_id]
            mark = "-" if score.hit is None else ("hit" if score.hit else "MISS")
            first = score.reciprocal_rank
            rank = "" if not first else f" @{round(1 / first)}"
            cells += f"{mark + rank:>16}"
        n_relevant = len(query.relevant)
        lines.append(f" {query.query_id:<8}{n_relevant:>4}  {cells}")
    lines.append("-" * _RULE_WIDTH)
    lines.append(" '@n' is the rank of the FIRST relevant chunk. 'MISS' means none in top k.")
    lines.append("=" * _RULE_WIDTH)
    return "\n".join(lines)


def report_as_json(report: AblationReport) -> dict[str, object]:
    """The same numbers as a JSON-safe mapping, for committing under ``results/``."""
    return {
        "golden_set": report.golden.name,
        "source": report.golden.source,
        "k": report.k,
        "confidence": report.confidence,
        "documents": report.ingest.documents_ingested,
        "chunks": report.ingest.chunks_added,
        "chunks_by_tier": dict(report.ingest.chunks_by_tier),
        "arms": [
            {
                "arm": arm.arm,
                "means": {str(metric): arm.summary(metric).mean for metric in RetrievalMetric},
                "n_scored": arm.summary(RetrievalMetric.HIT).n_scored,
                "n_undefined": arm.summary(RetrievalMetric.HIT).n_undefined,
                "hit_successes": arm.hit_interval.successes,
                "hit_trials": arm.hit_interval.trials,
                "hit_ci": [arm.hit_interval.low, arm.hit_interval.high],
                "per_query": {
                    score.query_id: {
                        "hit": score.hit,
                        "recall": score.recall,
                        "precision": score.precision,
                        "mrr": score.reciprocal_rank,
                        "ndcg": score.ndcg,
                    }
                    for score in arm.scores
                },
            }
            for arm in report.arms
        ],
        "caveat": " ".join(_CAVEAT),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_EPILOG = """\
scores the three retrieval arms over the committed golden set and prints one
comparison table. Everything is offline: no model download, no API key, no
network. The corpus is ingested once through the real IngestPipeline and the
real config/trust_tiers.yaml, and all three arms read those same indexes.

  uv run python -m evals.retrieval.run
  uv run python -m evals.retrieval.run --k 10 --per-query
  uv run python -m evals.retrieval.run --json results/retrieval_ablation.json

the golden set is a HAND-BUILT FIXTURE written by the authors of the retriever
it grades. It is a working-pipeline check, not a benchmark result, and the
caveat block under the table says so on every run.
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evals.retrieval.run",
        description="Score bm25 / vector / hybrid over a golden set.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--golden",
        type=Path,
        default=DEFAULT_GOLDEN_SET_PATH,
        help=f"golden set JSON (default: {DEFAULT_GOLDEN_SET_PATH.name} beside this module)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"cutoff every metric is reported at (default: {DEFAULT_K})",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=None,
        help="depth each arm retrieves before fusion (default: max(50, k))",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help=f"two-sided confidence level for the hit@k interval (default: {DEFAULT_CONFIDENCE})",
    )
    parser.add_argument(
        "--per-query",
        action="store_true",
        help="also print the per-query hit/miss grid, which is where a mean hides things",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="write the full report as JSON to this path as well as printing it",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        golden = load_golden_set(args.golden)
        report = evaluate(
            golden,
            k=args.k,
            candidate_k=args.candidate_k,
            confidence=args.confidence,
        )
    except GoldenSetError as exc:
        # A refusal here is the loader working: it declined to grade a corpus
        # whose labels no longer describe it. Report it as a CLI error, not as a
        # traceback that reads like a crash.
        parser.error(str(exc))
    print(render_ablation(report))
    if args.per_query:
        print()
        print(render_per_query(report))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report_as_json(report), indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
