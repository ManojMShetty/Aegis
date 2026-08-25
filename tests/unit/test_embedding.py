"""The TF-IDF vector space model - weights, not just orderings.

A ranking test alone cannot tell a working IDF from a broken one: with four
documents almost any weighting puts the obviously-relevant one first, so a test
that only checks the order passes on a model that has quietly stopped
down-weighting common terms. These tests therefore assert on
:meth:`TfidfEmbedder.idf` and on the vector components directly, and use
orderings only where the ordering is the claim.
"""

from __future__ import annotations

import math

import pytest

from aegis.retrieval.embedding import (
    Embedder,
    EmbedderNotFittedError,
    FittableEmbedder,
    TfidfEmbedder,
    cosine,
    dot,
    l2_norm,
)
from aegis.retrieval.sparse import tokenize

# --------------------------------------------------------------------------
# IDF: the part that has to be measured rather than eyeballed
# --------------------------------------------------------------------------

# "widget" is in every document; "sunglasses" is in exactly one.
IDF_CORPUS = [
    "The Acme Widget costs twenty dollars.",
    "Widget maintenance and cleaning guide.",
    "Return a widget within thirty days.",
    "A widget ships with cheap sunglasses.",
]


@pytest.fixture
def idf_model() -> TfidfEmbedder:
    model = TfidfEmbedder()
    model.fit(IDF_CORPUS)
    return model


def test_idf_downweights_a_term_that_appears_in_every_document(
    idf_model: TfidfEmbedder,
) -> None:
    """A term in all four documents distinguishes none of them."""
    assert idf_model.document_frequency("widget") == 4
    assert idf_model.document_frequency("sunglass") == 1
    assert idf_model.idf("widget") < idf_model.idf("sunglass")


def test_a_ubiquitous_term_sits_exactly_on_the_idf_floor(idf_model: TfidfEmbedder) -> None:
    """ln((1+N)/(1+N)) + 1 == 1.0 - the smoothed formula's minimum.

    Pinned as an exact value because the smoothing constant is a real modelling
    choice: the unsmoothed ln(N/df) would annihilate this term at 0.0, and a
    future edit that switched formulas would otherwise pass silently.
    """
    assert idf_model.idf("widget") == pytest.approx(1.0)


def test_idf_rises_as_a_term_gets_rarer(idf_model: TfidfEmbedder) -> None:
    expected = math.log((1 + 4) / (1 + 1)) + 1.0
    assert idf_model.idf("sunglass") == pytest.approx(expected)
    assert idf_model.idf("sunglass") > 1.0


def test_an_unseen_term_scores_the_maximum_idf(idf_model: TfidfEmbedder) -> None:
    """Not used when embedding - OOV terms are dropped - but the curve must be
    monotone in rarity for the formula to be the one we claim it is."""
    assert idf_model.idf("zebra") > idf_model.idf("sunglass")


def test_idf_reaches_the_document_weights(idf_model: TfidfEmbedder) -> None:
    """The weighting must actually be applied, not merely computable.

    Both terms occur once in this text, so their weights differ only by IDF.
    """
    vector = idf_model.embed(["a widget with sunglasses"])[0]
    assert vector["sunglass"] > vector["widget"]


# --------------------------------------------------------------------------
# Ranking: topicality versus keyword stuffing
# --------------------------------------------------------------------------

STUFFING_CORPUS = [
    "Account password recovery. Open settings, choose recovery, and follow the "
    "steps to reset your password.",
    "Password password password password password. Buy cheap sunglasses today.",
    "The Acme Widget costs twenty dollars and ships in two days.",
    "Error ERR_4021 means the payment gateway timed out.",
]


def test_topical_document_outranks_a_keyword_stuffed_one() -> None:
    """The stuffed document repeats the query's most common term and answers nothing.

    Sublinear tf plus L2 normalisation is what defeats it: the fifth "password"
    adds ln(5/4) rather than another whole unit, and normalising means a
    document pointing entirely along one axis cannot also match the query's
    other three terms.
    """
    model = TfidfEmbedder()
    model.fit(STUFFING_CORPUS)
    query = model.embed(["account password recovery steps"])[0]
    topical, stuffed = model.embed(STUFFING_CORPUS[:2])

    assert cosine(query, topical) > cosine(query, stuffed)
    assert cosine(query, topical) > 2 * cosine(query, stuffed)


def test_sublinear_tf_widens_the_gap_against_keyword_stuffing() -> None:
    """Five mentions must not weigh five times one mention.

    Each model is scored against its own query vector - comparing a damped
    document to a linear query would measure nothing but the mismatch. What is
    compared is the *separation* each model achieves between the document that
    answers the question and the one that only repeats a word from it.
    """
    damped = TfidfEmbedder()
    damped.fit(STUFFING_CORPUS)
    linear = TfidfEmbedder(sublinear_tf=False)
    linear.fit(STUFFING_CORPUS)

    query = "account password recovery steps"
    separations = []
    for model in (damped, linear):
        q = model.embed([query])[0]
        topical, stuffed = model.embed(STUFFING_CORPUS[:2])
        separations.append(cosine(q, topical) / cosine(q, stuffed))

    damped_gap, linear_gap = separations
    assert damped_gap > linear_gap


def test_documents_unrelated_to_the_query_score_zero() -> None:
    model = TfidfEmbedder()
    model.fit(STUFFING_CORPUS)
    query = model.embed(["account password recovery steps"])[0]
    unrelated = model.embed([STUFFING_CORPUS[3]])[0]
    assert cosine(query, unrelated) == 0.0


# --------------------------------------------------------------------------
# Cosine and the vector arithmetic
# --------------------------------------------------------------------------


def test_cosine_of_a_vector_with_itself_is_one() -> None:
    model = TfidfEmbedder()
    model.fit(STUFFING_CORPUS)
    vector = model.embed([STUFFING_CORPUS[0]])[0]
    assert cosine(vector, vector) == pytest.approx(1.0)


def test_cosine_never_exceeds_one() -> None:
    """Float error can push a self-similarity past 1.0; the clamp exists for it."""
    model = TfidfEmbedder()
    model.fit(STUFFING_CORPUS)
    for text in STUFFING_CORPUS:
        vector = model.embed([text])[0]
        assert cosine(vector, vector) <= 1.0


def test_cosine_of_disjoint_vectors_is_zero() -> None:
    assert cosine({"a": 1.0, "b": 2.0}, {"c": 3.0, "d": 4.0}) == 0.0


def test_cosine_of_an_empty_vector_is_zero() -> None:
    assert cosine({}, {"a": 1.0}) == 0.0
    assert cosine({}, {}) == 0.0


def test_cosine_is_symmetric() -> None:
    a = {"x": 0.6, "y": 0.8}
    b = {"y": 1.0, "z": 2.0}
    assert cosine(a, b) == pytest.approx(cosine(b, a))


def test_cosine_ignores_magnitude() -> None:
    """Doubling every weight is the same direction, so the same similarity."""
    a = {"x": 1.0, "y": 1.0}
    doubled = {"x": 2.0, "y": 2.0}
    other = {"x": 1.0, "z": 5.0}
    assert cosine(a, other) == pytest.approx(cosine(doubled, other))


def test_dot_matches_cosine_on_normalised_vectors() -> None:
    """The claim the module makes about its own vectors, checked rather than asserted."""
    model = TfidfEmbedder()
    model.fit(STUFFING_CORPUS)
    a, b = model.embed([STUFFING_CORPUS[0], STUFFING_CORPUS[1]])
    assert dot(a, b) == pytest.approx(cosine(a, b))


def test_dot_is_order_independent() -> None:
    a = {"p": 0.5, "q": 0.5, "r": 0.5}
    b = {"r": 1.0, "q": 2.0, "p": 3.0}
    assert dot(a, b) == pytest.approx(dot(b, a))


def test_document_vectors_are_unit_length() -> None:
    model = TfidfEmbedder()
    model.fit(STUFFING_CORPUS)
    for vector in model.embed(STUFFING_CORPUS):
        assert l2_norm(vector) == pytest.approx(1.0)


def test_a_text_with_no_known_terms_yields_the_empty_vector() -> None:
    model = TfidfEmbedder()
    model.fit(STUFFING_CORPUS)
    assert model.embed(["zebra quantum helicopter"])[0] == {}


# --------------------------------------------------------------------------
# Fitting, vocabulary, and the protocols
# --------------------------------------------------------------------------


def test_embedding_before_fitting_raises() -> None:
    """A TF-IDF vector is meaningless without a corpus; silence would hide that."""
    with pytest.raises(EmbedderNotFittedError, match="fit"):
        TfidfEmbedder().embed(["anything"])


def test_dimension_is_the_vocabulary_size() -> None:
    model = TfidfEmbedder()
    model.fit(["alpha beta", "beta gamma"])
    assert model.dimension == 3
    assert model.vocabulary == ("alpha", "beta", "gamma")


def test_fitting_is_incremental() -> None:
    incremental = TfidfEmbedder()
    for text in IDF_CORPUS:
        incremental.fit([text])
    batch = TfidfEmbedder()
    batch.fit(IDF_CORPUS)

    assert incremental.corpus_size == batch.corpus_size == 4
    assert incremental.vocabulary == batch.vocabulary
    assert incremental.idf("widget") == pytest.approx(batch.idf("widget"))


def test_reset_forgets_everything() -> None:
    model = TfidfEmbedder()
    model.fit(IDF_CORPUS)
    model.reset()
    assert not model.is_fitted
    assert model.dimension == 0


def test_embedding_is_deterministic() -> None:
    """A vector that moves between runs makes an eval unreproducible."""
    model = TfidfEmbedder()
    model.fit(IDF_CORPUS)
    assert model.embed(IDF_CORPUS) == model.embed(IDF_CORPUS)


def test_the_embedder_shares_the_bm25_tokenizer() -> None:
    """Two tokenizers means the two retrieval arms index different corpora.

    The visible consequence: singular/plural must close here exactly as it does
    for BM25, or a query for "cost" misses a document about "costs" in one arm
    and hits it in the other.
    """
    model = TfidfEmbedder()
    model.fit(["The widget costs twenty dollars.", "Unrelated maintenance guide."])
    vector = model.embed(["widget cost"])[0]
    assert "cost" in vector
    assert set(vector) <= set(tokenize("The widget costs twenty dollars."))


def test_tfidf_embedder_satisfies_both_protocols() -> None:
    """The seam is only real if the shipped implementation actually fits it."""
    model = TfidfEmbedder()
    assert isinstance(model, Embedder)
    assert isinstance(model, FittableEmbedder)


def test_a_neural_style_embedder_needs_no_fit_method() -> None:
    """A fixed-dimension encoder is an Embedder but not a FittableEmbedder.

    This is what lets VectorIndex fit TF-IDF and leave a neural model alone.
    """

    class FixedEncoder:
        @property
        def dimension(self) -> int:
            return 3

        def embed(self, texts: list[str]) -> list[dict[str, float]]:
            return [{"0": 1.0, "1": 0.0, "2": 0.0} for _ in texts]

    encoder = FixedEncoder()
    assert isinstance(encoder, Embedder)
    assert not isinstance(encoder, FittableEmbedder)
