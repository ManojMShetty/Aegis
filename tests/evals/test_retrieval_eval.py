"""Tests for the golden set, the corpus it builds, and the ablation CLI.

WHAT IS BEING GUARDED HERE
--------------------------
Two different things, and they fail differently.

The **loader** guards the fixture: a duplicate query id, an empty relevance list,
a chunker setting nobody recognises, a label pointing at a chunk the corpus does
not contain. Every one of those still produces a table if it is tolerated, and
the table then answers a question other than the one its headings claim. So the
tests here are mostly assertions that the loader *refuses*.

The **runner** guards the comparison: that all four arms read one corpus built
once, that the identity reranker cannot manufacture a gap, and that no rendered
rate appears without its denominator. Those are the properties that make the
README table mean what it says.

THE MEASURED NUMBERS ARE PINNED, AND THAT IS DELIBERATE
-------------------------------------------------------
``test_hybrid_does_not_beat_the_better_arm_on_this_fixture`` pins the actual
result: on this corpus every arm finds the same 22 of 25 queries, and fusion
lands strictly between BM25 and the vector arm on nDCG rather than above both.
That is the sentence the README prints, so it is a test - if a change to
retrieval makes it false, the test fails and the README has to be rewritten,
which is the correct order of events.

Everything is offline and takes well under a second.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from evals.retrieval import run as run_module
from evals.retrieval.golden_set import (
    DEFAULT_GOLDEN_SET_PATH,
    GoldenSetError,
    build_corpus,
    chunk_key,
    load_golden_set,
)
from evals.retrieval.metrics import RetrievalMetric
from evals.retrieval.run import DEFAULT_K, evaluate, main, render_ablation, render_per_query

from aegis.retrieval.retriever import RetrievalConfig

# ---------------------------------------------------------------------------
# The committed fixture
# ---------------------------------------------------------------------------


def test_the_committed_golden_set_loads() -> None:
    golden = load_golden_set()
    assert golden.name == "meridian-support-kb-v1"
    assert len(golden.documents) == 10
    assert len(golden.queries) == 25
    assert golden.source == str(DEFAULT_GOLDEN_SET_PATH)


def test_the_fixture_says_in_its_own_description_that_it_is_not_a_benchmark() -> None:
    """The caveat travels with the data, not only with the renderer.

    A reader who opens the JSON and never runs the CLI must still be told that
    the people who wrote these labels also wrote the retriever they grade.
    """
    description = load_golden_set().description.lower()
    assert "hand-built" in description or "fixture" in description
    assert "not an external benchmark" in description


def test_every_query_records_why_its_labels_are_the_labels() -> None:
    """A relevance judgement nobody wrote down is a judgement nobody can dispute."""
    for query in load_golden_set().queries:
        assert query.note.strip(), f"{query.query_id} has no note"


def test_the_corpus_ingests_through_the_real_policy_at_the_expected_tier() -> None:
    """The eval corpus is labelled by ``config/trust_tiers.yaml``, not by the eval.

    ``file://corpus/*`` is a T2_CURATED rule in the committed policy, so a change
    that broadened or dropped that rule shows up here as a tier change rather than
    silently, in a system whose whole thesis is that provenance is load-bearing.
    """
    golden = load_golden_set()
    retriever, report = build_corpus(golden)
    assert report.documents_ingested == 10
    assert report.chunks_added == 44
    assert report.chunks_by_tier == {"T2_CURATED": 44}
    assert len(retriever.chunks) == 44
    assert all(chunk.tier.label == "T2_CURATED" for chunk in retriever.chunks)


def test_every_label_resolves_to_a_chunk_that_exists() -> None:
    golden = load_golden_set()
    retriever, _ = build_corpus(golden)
    available = {chunk_key(chunk) for chunk in retriever.chunks}
    assert golden.labelled_keys <= available


def test_chunk_key_is_the_stable_prefix_of_the_content_addressed_id() -> None:
    """A label pins position, not bytes; :func:`build_corpus` re-derives the rest."""
    retriever, _ = build_corpus(load_golden_set())
    chunk = retriever.chunks[0]
    assert chunk.chunk_id.startswith(chunk_key(chunk) + ":")


# ---------------------------------------------------------------------------
# The loader refuses
# ---------------------------------------------------------------------------


def _minimal() -> dict[str, Any]:
    return {
        "name": "tiny",
        "description": "a fixture, not an external benchmark",
        "chunker": {"max_tokens": 512, "overlap_tokens": 64, "respect_headings": True},
        "documents": [{"doc_id": "d.md", "text": "# Doc\n\n## One\nalpha beta gamma\n"}],
        "queries": [{"id": "q1", "query": "alpha", "relevant_chunk_ids": ["d.md#0"]}],
    }


def _write(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_a_minimal_golden_set_round_trips(tmp_path: Path) -> None:
    golden = load_golden_set(_write(tmp_path, _minimal()))
    retriever, report = build_corpus(golden)
    assert report.chunks_added == 1
    assert chunk_key(retriever.chunks[0]) == "d.md#0"


def test_a_query_with_no_relevant_chunks_is_refused(tmp_path: Path) -> None:
    """It grades nothing: every metric on it is undefined by design, so it would
    add a row to the table that no retriever can influence."""
    payload = _minimal()
    payload["queries"][0]["relevant_chunk_ids"] = []
    with pytest.raises(GoldenSetError, match="nothing to find"):
        load_golden_set(_write(tmp_path, payload))


def test_a_duplicate_query_id_is_refused(tmp_path: Path) -> None:
    payload = _minimal()
    payload["queries"].append(dict(payload["queries"][0]))
    with pytest.raises(GoldenSetError, match="duplicate query id"):
        load_golden_set(_write(tmp_path, payload))


def test_a_duplicate_doc_id_is_refused(tmp_path: Path) -> None:
    """Two documents under one id share a chunk-key namespace, so one label would
    name two different chunks."""
    payload = _minimal()
    payload["documents"].append(dict(payload["documents"][0]))
    with pytest.raises(GoldenSetError, match="duplicate doc_id"):
        load_golden_set(_write(tmp_path, payload))


def test_a_repeated_relevant_id_is_refused(tmp_path: Path) -> None:
    payload = _minimal()
    payload["queries"][0]["relevant_chunk_ids"] = ["d.md#0", "d.md#0"]
    with pytest.raises(GoldenSetError, match="duplicate"):
        load_golden_set(_write(tmp_path, payload))


def test_an_unknown_chunker_setting_is_refused(tmp_path: Path) -> None:
    """A setting nobody reads means the ordinals were written against a split
    this loader is not going to reproduce."""
    payload = _minimal()
    payload["chunker"]["strategy"] = "semantic"
    with pytest.raises(GoldenSetError, match="unknown chunker setting"):
        load_golden_set(_write(tmp_path, payload))


def test_a_missing_field_names_the_field(tmp_path: Path) -> None:
    payload = _minimal()
    del payload["documents"]
    with pytest.raises(GoldenSetError, match="'documents'"):
        load_golden_set(_write(tmp_path, payload))


def test_a_missing_file_and_a_broken_file_both_raise_golden_set_error(tmp_path: Path) -> None:
    with pytest.raises(GoldenSetError, match="not found"):
        load_golden_set(tmp_path / "absent.json")
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(GoldenSetError, match="not valid JSON"):
        load_golden_set(broken)


def test_a_label_that_names_no_chunk_raises_instead_of_scoring_zero(tmp_path: Path) -> None:
    """THE FAILURE THIS EXISTS FOR.

    A stale label is indistinguishable from a retrieval miss once it reaches the
    metrics: the query scores 0 on everything and the arm looks worse. Refusing
    at corpus-build time is the only place the two can still be told apart.
    """
    payload = _minimal()
    payload["queries"][0]["relevant_chunk_ids"] = ["d.md#7"]
    golden = load_golden_set(_write(tmp_path, payload))
    with pytest.raises(GoldenSetError, match="not in the ingested corpus"):
        build_corpus(golden)


def test_changing_the_chunker_settings_makes_the_labels_fail_loudly(tmp_path: Path) -> None:
    """Ordinals are the labels, so the file that owns the labels owns the split."""
    payload = _minimal()
    payload["documents"][0]["text"] = "# D\n\n## One\nalpha\n\n## Two\nbeta\n"
    payload["queries"][0]["relevant_chunk_ids"] = ["d.md#1"]
    assert build_corpus(load_golden_set(_write(tmp_path, payload)))[1].chunks_added == 2

    payload["chunker"]["respect_headings"] = False
    golden = load_golden_set(_write(tmp_path, payload))
    with pytest.raises(GoldenSetError, match="not in the ingested corpus"):
        build_corpus(golden)


# ---------------------------------------------------------------------------
# The ablation
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def report() -> run_module.AblationReport:
    return evaluate(load_golden_set())


def test_only_real_arms_are_reported_in_baseline_first_order(
    report: run_module.AblationReport,
) -> None:
    """hybrid+rerank is deliberately absent: aegis.retrieval.rerank forbids
    reporting it as an arm while identity is the only implementation, because the
    row would describe the same system as `hybrid` and invite "reranking tested,
    no difference" from a table where nothing was tested."""
    assert [arm.arm for arm in report.arms] == ["bm25", "vector", "hybrid"]
    assert report.baseline.arm == "bm25"


def test_the_corpus_is_built_once_for_all_arms(monkeypatch: pytest.MonkeyPatch) -> None:
    """Not an optimisation. Re-ingesting per arm would re-fit the TF-IDF
    vocabulary, and the four rows would then be four corpora as much as four
    retrievers - the confound the whole ablation exists to exclude.
    """
    calls = 0
    original = run_module.build_corpus

    def counting(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(run_module, "build_corpus", counting)
    evaluate(load_golden_set())
    assert calls == 1


def test_every_arm_is_scored_over_the_same_queries(report: run_module.AblationReport) -> None:
    """Two means over different query subsets are not comparable at all."""
    query_ids = {arm.arm: tuple(score.query_id for score in arm.scores) for arm in report.arms}
    assert len(set(query_ids.values())) == 1
    for arm in report.arms:
        summary = arm.summary(RetrievalMetric.HIT)
        assert (summary.n_scored, summary.n_undefined) == (25, 0)


def test_the_identity_reranker_cannot_manufacture_a_gap(
    report: run_module.AblationReport,
) -> None:
    """``hybrid+rerank`` must equal ``hybrid`` exactly while the only shipped
    reranker is the identity seam. Without this, an ablation table could report a
    phantom second-stage win that is pure floating-point noise."""
    golden = run_module.load_golden_set()
    retriever, _ = run_module.build_corpus(golden)
    hybrid = run_module.evaluate_arm(
        retriever, golden, RetrievalConfig.hybrid(top_k=5, candidate_k=20)
    )
    reranked = run_module.evaluate_arm(
        retriever, golden, RetrievalConfig.hybrid_reranked(top_k=5, candidate_k=20)
    )
    assert reranked.arm == "hybrid+rerank"
    for metric in RetrievalMetric:
        assert hybrid.summary(metric).mean == reranked.summary(metric).mean


def test_hybrid_does_not_beat_the_better_arm_on_this_fixture(
    report: run_module.AblationReport,
) -> None:
    """THE MEASURED RESULT, pinned so the README cannot drift away from it.

    At k=5 all four arms find at least one relevant chunk for the same 22 of 25
    queries, so hit@k, recall@k and precision@k are identical across the table.
    The arms differ only in *ordering*, and there fusion lands strictly between
    its two inputs rather than above them: BM25 < hybrid < vector on nDCG. Both
    arms are lexical - BM25 over stems, TF-IDF cosine over the same stems - so
    they fail on the same queries, and RRF cannot recover a chunk that neither
    arm surfaced.
    """
    by_arm = {arm.arm: arm for arm in report.arms}
    assert {arm.hit_interval.successes for arm in report.arms} == {22}
    assert {arm.hit_interval.trials for arm in report.arms} == {25}

    bm25 = by_arm["bm25"].summary(RetrievalMetric.NDCG).mean
    vector = by_arm["vector"].summary(RetrievalMetric.NDCG).mean
    hybrid = by_arm["hybrid"].summary(RetrievalMetric.NDCG).mean
    assert bm25 is not None and vector is not None and hybrid is not None
    assert bm25 < hybrid < vector
    assert bm25 == pytest.approx(0.720949, abs=1e-6)
    assert vector == pytest.approx(0.746186, abs=1e-6)
    assert hybrid == pytest.approx(0.731423, abs=1e-6)


def test_the_comparison_is_underpowered_and_says_so(
    report: run_module.AblationReport,
) -> None:
    """Zero discordant pairs on hit@k: the test could not have shown an effect.

    Twenty-five queries where every arm agrees is not evidence that the arms are
    equivalent, and the renderer has to print the difference between "no effect
    was found" and "no effect could have been found".
    """
    test = run_module.paired_hit_test(report.baseline, report.arms[2])
    assert test.n_discordant == 0
    assert test.underpowered
    assert "POWER" in render_ablation(report)


def test_evaluation_is_deterministic() -> None:
    """A retrieval eval whose number moves between runs cannot be committed."""
    first = evaluate(load_golden_set())
    second = evaluate(load_golden_set())
    for left, right in zip(first.arms, second.arms, strict=True):
        for metric in RetrievalMetric:
            assert left.summary(metric).mean == right.summary(metric).mean


# ---------------------------------------------------------------------------
# Rendering and the CLI
# ---------------------------------------------------------------------------


def test_no_rate_is_rendered_without_its_denominator_and_interval(
    report: run_module.AblationReport,
) -> None:
    """The invariant ``evals.stats.analysis`` holds for changes, held here for rates.

    Every hit@k line carries the fraction it came from and the interval, on the
    same line, because "88.0%" is a sentence a reader will quote out of context
    and "88.0% (22/25) 95% CI [70.0%, 95.8%]" is not.
    """
    rendered = render_ablation(report)
    for line in rendered.splitlines():
        if "%" in line and "CI" not in line:
            pytest.fail(f"a percentage was rendered without an interval: {line!r}")
    for arm in report.arms:
        assert f"{arm.arm:<15} {arm.hit_interval.as_percent_str()}" in rendered


def test_the_caveat_is_unconditional(report: run_module.AblationReport) -> None:
    """It is not a flag and not a verbosity level. A table that can be pasted
    without the sentence that qualifies it will be."""
    rendered = render_ablation(report)
    assert "FIXTURE, NOT A BENCHMARK" in rendered
    assert "written by the authors of the" in rendered
    assert "caveat" in run_module.report_as_json(report)


def test_the_table_states_the_corpus_and_the_cutoff(
    report: run_module.AblationReport,
) -> None:
    rendered = render_ablation(report)
    assert "10 documents, 44 chunks (T2_CURATED=44)" in rendered
    assert f"k = {DEFAULT_K}" in rendered
    assert "25 (25 scored, 0 undefined)" in rendered


def test_the_per_query_grid_shows_the_three_misses(
    report: run_module.AblationReport,
) -> None:
    """A mean of 0.88 hides which queries failed; these three are in the fixture
    on purpose and each one's note says why."""
    grid = render_per_query(report)
    rows = [line for line in grid.splitlines() if re.match(r" q\d\d ", line)]
    assert len(rows) == 25
    assert sum(row.count("MISS") for row in rows) == 9  # three queries, three arms
    for query_id in ("q02", "q08", "q13"):
        row = next(line for line in grid.splitlines() if line.startswith(f" {query_id} "))
        assert row.count("MISS") == 3


def test_the_cli_runs_and_writes_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "nested" / "ablation.json"
    assert main(["--k", "3", "--per-query", "--json", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "Retrieval ablation: meridian-support-kb-v1" in printed
    assert "Per-query hit@3" in printed

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["k"] == 3
    assert [arm["arm"] for arm in payload["arms"]] == ["bm25", "vector", "hybrid"]
    assert payload["chunks_by_tier"] == {"T2_CURATED": 44}


def test_the_cli_reports_a_broken_fixture_as_an_error_not_a_traceback(
    tmp_path: Path,
) -> None:
    payload = _minimal()
    payload["queries"][0]["relevant_chunk_ids"] = []
    with pytest.raises(SystemExit) as excinfo:
        main(["--golden", str(_write(tmp_path, payload))])
    assert excinfo.value.code == 2
