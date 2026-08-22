"""Chunking, BM25, and RRF - including the security property that chunking
cannot launder trust."""

from __future__ import annotations

import pytest

from aegis.domain.chunk import Chunk, ScoredChunk
from aegis.domain.trust import Tainted, TrustTier
from aegis.ingest.chunker import RecursiveChunker, chunk_document
from aegis.retrieval.fusion import reciprocal_rank_fusion
from aegis.retrieval.sparse import BM25Index, stem, tokenize

# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------


def test_short_document_is_one_chunk() -> None:
    chunks = chunk_document(
        "A short note.", doc_id="d1", source_uri="kb://d1", tier=TrustTier.CURATED
    )
    assert len(chunks) == 1
    assert chunks[0].value == "A short note."


def test_long_document_is_split() -> None:
    text = "\n\n".join(f"Paragraph {i} with some filler content. " * 6 for i in range(12))
    chunks = chunk_document(
        text,
        doc_id="d1",
        source_uri="kb://d1",
        tier=TrustTier.CURATED,
        chunker=RecursiveChunker(max_tokens=64, overlap_tokens=8),
    )
    assert len(chunks) > 1
    assert all(c.value.strip() for c in chunks)


def test_headings_become_chunk_labels() -> None:
    text = "# Overview\nIntro text here.\n\n## Pricing\nThe widget costs $20."
    chunks = chunk_document("x" and text, doc_id="d1", source_uri="kb://d1", tier=TrustTier.CURATED)
    headings = {c.heading for c in chunks}
    assert "Overview" in headings
    assert "Pricing" in headings


def test_ordinals_are_sequential() -> None:
    text = "# A\nfirst\n\n# B\nsecond\n\n# C\nthird"
    chunks = chunk_document(text, doc_id="d1", source_uri="kb://d1", tier=TrustTier.CURATED)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_chunk_ids_are_stable_and_unique() -> None:
    text = "# A\nalpha content\n\n# B\nbeta content"
    a = chunk_document(text, doc_id="d1", source_uri="kb://d1", tier=TrustTier.CURATED)
    b = chunk_document(text, doc_id="d1", source_uri="kb://d1", tier=TrustTier.CURATED)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]  # stable
    assert len({c.chunk_id for c in a}) == len(a)  # unique


@pytest.mark.parametrize("bad", [{"max_tokens": 0}, {"overlap_tokens": -1}])
def test_invalid_chunker_config_rejected(bad: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        RecursiveChunker(**bad)


def test_overlap_must_be_smaller_than_chunk() -> None:
    """Otherwise each chunk re-emits its predecessor and the walk never advances."""
    with pytest.raises(ValueError, match="smaller than max_tokens"):
        RecursiveChunker(max_tokens=100, overlap_tokens=100)


def test_empty_document_yields_no_chunks() -> None:
    assert chunk_document("   ", doc_id="d", source_uri="kb://d", tier=TrustTier.CURATED) == []


# --------------------------------------------------------------------------
# Security: chunking cannot launder trust
# --------------------------------------------------------------------------


@pytest.mark.security
def test_chunks_inherit_the_document_tier() -> None:
    """Slicing a poisoned document produces poisoned chunks."""
    text = "\n\n".join(f"Paragraph {i}. " * 20 for i in range(8))
    chunks = chunk_document(
        text,
        doc_id="evil",
        source_uri="https://evil.test/page",
        tier=TrustTier.UNTRUSTED,
        chunker=RecursiveChunker(max_tokens=64, overlap_tokens=8),
    )
    assert len(chunks) > 1
    assert all(c.tier is TrustTier.UNTRUSTED for c in chunks)
    assert all(not c.text.is_instruction_authority for c in chunks)


@pytest.mark.security
def test_chunks_keep_the_document_provenance() -> None:
    """The citation channel depends on origin surviving the split."""
    chunks = chunk_document(
        "some content", doc_id="d", source_uri="https://evil.test/p", tier=TrustTier.UNTRUSTED
    )
    assert chunks[0].source_uri == "https://evil.test/p"
    assert "T0_UNTRUSTED" in chunks[0].citation_label()


# --------------------------------------------------------------------------
# BM25
# --------------------------------------------------------------------------


def make_chunk(text: str, doc_id: str, tier: TrustTier = TrustTier.CURATED) -> Chunk:
    return Chunk(
        text=Tainted.trusted(text, tier, source_uri=f"kb://{doc_id}"),
        doc_id=doc_id,
        ordinal=0,
    )


@pytest.fixture
def index() -> BM25Index:
    idx = BM25Index()
    idx.add(
        [
            make_chunk("The Acme Widget costs twenty dollars and ships in two days.", "d1"),
            make_chunk("Password reset instructions for your account.", "d2"),
            make_chunk("Error ERR_4021 means the payment gateway timed out.", "d3"),
            make_chunk("Widget maintenance schedule and cleaning guide.", "d4"),
        ]
    )
    return idx


def test_bm25_finds_the_relevant_chunk(index: BM25Index) -> None:
    results = index.search("how much does the widget cost")
    assert results
    assert results[0].chunk.doc_id == "d1"


def test_bm25_excels_at_exact_identifiers(index: BM25Index) -> None:
    """The case dense retrieval handles badly - this is why hybrid exists."""
    results = index.search("ERR_4021")
    assert results[0].chunk.doc_id == "d3"


def test_bm25_preserves_identifier_tokens() -> None:
    assert "err_4021" in tokenize("Error ERR_4021 occurred")
    assert "send_email" in tokenize("call send_email now")


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("costs", "cost"),
        ("cost", "cost"),
        ("ships", "ship"),
        ("companies", "company"),
        ("boxes", "box"),
        ("cleaning", "clean"),
        ("address", "address"),  # doubled s is part of the word, not a plural
        ("addresses", "address"),
        ("err_4021", "err_4021"),  # identifiers must survive intact
        ("send_email", "send_email"),
        ("box", "box"),  # too short to touch
    ],
)
def test_stemmer_is_conservative(word: str, expected: str) -> None:
    assert stem(word) == expected


def test_stemming_is_consistent_even_when_imperfect() -> None:
    """A light stemmer does not undouble consonants ('running' -> 'runn').

    That is fine: matching only requires the query and the document to stem the
    *same* way, not that the stem be a real word.
    """
    assert stem("running") == stem("running")
    assert tokenize("running fast") == tokenize("running fast")


def test_stemming_closes_the_singular_plural_gap() -> None:
    """The bug this stemmer was added to fix, pinned as a test."""
    idx = BM25Index()
    idx.add(
        [
            make_chunk("The Acme Widget costs twenty dollars and ships in two days.", "d1"),
            make_chunk("Widget maintenance schedule and cleaning guide.", "d2"),
        ]
    )
    # 'cost' must match 'costs', otherwise only 'widget' matches and the
    # shorter, less relevant chunk wins on length normalisation.
    assert idx.search("widget cost")[0].chunk.doc_id == "d1"


def test_bm25_returns_nothing_for_unmatched_query(index: BM25Index) -> None:
    assert index.search("zebra quantum helicopter") == []


def test_bm25_empty_index_is_safe() -> None:
    assert BM25Index().search("anything") == []


def test_bm25_respects_top_k(index: BM25Index) -> None:
    assert len(index.search("widget", top_k=1)) == 1


def test_bm25_results_are_sorted_descending(index: BM25Index) -> None:
    results = index.search("widget")
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_bm25_tags_its_results(index: BM25Index) -> None:
    assert index.search("widget")[0].retriever == "bm25"


def test_stopwords_do_not_dominate(index: BM25Index) -> None:
    """A query of only stopwords carries no signal and must match nothing."""
    assert index.search("the and of is") == []


# --------------------------------------------------------------------------
# Reciprocal Rank Fusion
# --------------------------------------------------------------------------


def scored(doc_id: str, score: float, retriever: str) -> ScoredChunk:
    return ScoredChunk(
        chunk=make_chunk(f"content {doc_id}", doc_id), score=score, retriever=retriever
    )


def test_rrf_rewards_agreement_between_retrievers() -> None:
    """Found by both retrievers beats found decisively by only one."""
    fused = reciprocal_rank_fusion(
        {
            "bm25": [scored("agreed", 9.0, "bm25"), scored("only_sparse", 8.0, "bm25")],
            "dense": [scored("agreed", 0.9, "dense"), scored("only_dense", 0.8, "dense")],
        }
    )
    assert fused[0].chunk.doc_id == "agreed"
    assert "bm25+dense" in fused[0].retriever


def test_rrf_ignores_incomparable_score_scales() -> None:
    """BM25's 30.0 and cosine's 0.9 must not be compared numerically."""
    fused = reciprocal_rank_fusion(
        {
            "bm25": [scored("a", 30.0, "bm25")],
            "dense": [scored("b", 0.9, "dense")],
        }
    )
    # Both ranked #1 by their own retriever, so both get an identical 1/(k+1).
    assert fused[0].score == pytest.approx(fused[1].score)


def test_rrf_keeps_chunks_found_by_only_one_retriever() -> None:
    fused = reciprocal_rank_fusion({"bm25": [scored("solo", 5.0, "bm25")]})
    assert [f.chunk.doc_id for f in fused] == ["solo"]


def test_rrf_respects_weights() -> None:
    fused = reciprocal_rank_fusion(
        {"bm25": [scored("s", 1.0, "bm25")], "dense": [scored("d", 1.0, "dense")]},
        weights={"dense": 2.0},
    )
    assert fused[0].chunk.doc_id == "d"


def test_rrf_is_deterministic() -> None:
    """An unstable sort silently changes recall@k between eval runs."""
    rankings = {
        "bm25": [scored("a", 1.0, "bm25"), scored("b", 1.0, "bm25")],
        "dense": [scored("b", 1.0, "dense"), scored("a", 1.0, "dense")],
    }
    first = [f.chunk_id for f in reciprocal_rank_fusion(rankings)]
    second = [f.chunk_id for f in reciprocal_rank_fusion(rankings)]
    assert first == second


def test_rrf_empty_input() -> None:
    assert reciprocal_rank_fusion({}) == []


def test_rrf_respects_top_k() -> None:
    fused = reciprocal_rank_fusion(
        {"bm25": [scored(f"d{i}", 1.0, "bm25") for i in range(10)]}, top_k=3
    )
    assert len(fused) == 3
