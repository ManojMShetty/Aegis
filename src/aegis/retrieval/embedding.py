"""Vector-space embedding: a genuine TF-IDF model, and the seam a neural one plugs into.

WHY THE DEFAULT IS NOT NEURAL
-----------------------------
The obvious way to build the vector arm of a hybrid retriever is to import
sentence-transformers, download a few hundred megabytes of weights, and call
``model.encode``. That drags torch into a project whose security core is
deliberately dependency-free, and it makes the retrieval half of this repo
unrunnable in CI and unrunnable offline - which is to say, unverifiable.

The tempting shortcut is worse than the heavy dependency. Hash each token into
one of 384 buckets, sum, normalise, and call the result an "embedding": it has
the right shape, it runs anywhere, and it means nothing. Two documents about the
same topic in different words land nowhere near each other, because hashing
destroys precisely the structure an embedding exists to capture. Any recall
number computed on top of that is a fabrication wearing a plausible interface,
and this project exists to avoid exactly that.

So the default is a real, citable model: the TF-IDF vector space model (Salton,
Wong & Yang, 1975; Sparck Jones, 1972). A document becomes a weighted term
vector; similarity is the cosine of the angle between two such vectors. Half a
century of IR is built on it, and it is still the baseline every dense retriever
is measured against.

BE PLAIN ABOUT WHAT IT IS
-------------------------
TF-IDF is LEXICAL-SEMANTIC, not neural. It knows that a rare term carries more
signal than a common one, and that a document is about the terms it uses
repeatedly and distinctively. It does NOT know that "car" and "automobile" are
the same thing. Calling it "semantic search" would be an overclaim; calling it a
vector space model is exactly correct. Where true synonymy matters, any neural
encoder that satisfies :class:`Embedder` drops in without touching
:class:`~aegis.retrieval.dense.VectorIndex`, fusion, or the retriever - that
substitution is the whole point of the protocol.

THE WEIGHTING
-------------
For term ``t`` in document ``d`` drawn from a corpus of ``N`` documents::

    tf(t, d)  = 1 + ln(count(t, d))     sublinear: the 10th mention does not
                                        make a document ten times more about it
    idf(t)    = ln((1 + N) / (1 + df(t))) + 1
    w(t, d)   = tf(t, d) * idf(t)
    vector(d) = w(., d) / L2norm(w(., d))    so a long document cannot win on
                                             length alone

The ``+1`` smoothing is the ``smooth_idf`` convention: it behaves as if the
corpus held one extra document containing every term. That avoids a division by
zero, and keeps a term appearing in *every* document at idf 1.0 rather than
annihilating it at 0.0 - a ubiquitous term is weak evidence, not forbidden
evidence. It also makes these weights directly comparable to the reference
implementation most readers already know.

Because every vector is L2-normalised, cosine similarity reduces to a plain dot
product. :func:`cosine` still divides by the norms so it stays correct for an
embedder that does not normalise; on our own vectors the divisor is 1.0.

WHY SPARSE DICTS
----------------
The vocabulary of a real corpus runs to tens of thousands of terms while a chunk
contains a hundred. A dense ``list[float]`` per document would be 99% zeros, and
the dot product would spend all its time adding them up. A ``dict[str, float]``
holding only the terms actually present is the right shape: storage and scoring
both scale with what a document says, not with what the corpus could have said.

WHY THE SAME TOKENIZER AS BM25
------------------------------
:func:`~aegis.retrieval.sparse.tokenize` is imported rather than re-written. Two
tokenizers means the two arms of the hybrid disagree about what a term *is* -
one index matching ``costs`` to ``cost`` while the other does not - and fusion
then combines two rankings built over different vocabularies. That bug never
shows up in the output: the ranking still looks plausible, it is merely worse.
One tokenizer, shared, by construction.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol, TypeAlias, runtime_checkable

from aegis.retrieval.sparse import tokenize

__all__ = [
    "Embedder",
    "EmbedderNotFittedError",
    "FittableEmbedder",
    "SparseVector",
    "TfidfEmbedder",
    "cosine",
    "dot",
    "l2_norm",
    "l2_normalize",
]

SparseVector: TypeAlias = Mapping[str, float]
"""A vector as ``{term: weight}``, carrying only its non-zero components."""


class EmbedderNotFittedError(RuntimeError):
    """An embedder needing corpus statistics was asked to embed without them.

    Raised rather than quietly returning zero vectors. A silently empty vector
    makes every cosine 0.0, so retrieval returns nothing at all and the operator
    is left debugging an "empty corpus" that is in fact fully indexed.
    """


@runtime_checkable
class Embedder(Protocol):
    """Turns text into vectors. The seam a neural encoder plugs into.

    Deliberately narrow. Anything that maps strings to sparse vectors satisfies
    it: the TF-IDF model below, a sentence-transformer wrapper that emits
    ``{str(axis): weight}``, or a SPLADE-style learned sparse encoder, which is
    natively this shape.
    """

    @property
    def dimension(self) -> int:
        """Size of the vector space (for TF-IDF, the vocabulary)."""

    def embed(self, texts: Sequence[str]) -> list[SparseVector]:
        """Vectorise a batch. The output order matches the input order."""
        ...


@runtime_checkable
class FittableEmbedder(Embedder, Protocol):
    """An embedder whose weights depend on corpus statistics.

    TF-IDF must know what the *rest* of the corpus looks like before it can
    weight one document; a neural encoder need not. That difference is a real
    property of the two model families rather than an implementation detail, so
    it gets its own protocol instead of a ``fit`` method every neural encoder
    would have to stub out. :class:`~aegis.retrieval.dense.VectorIndex` checks
    for this protocol and fits as it indexes.
    """

    def fit(self, corpus: Iterable[str]) -> None:
        """Accumulate corpus statistics from these documents."""
        ...


# ---------------------------------------------------------------------------
# vector arithmetic
# ---------------------------------------------------------------------------


def dot(a: SparseVector, b: SparseVector) -> float:
    """Inner product of two sparse vectors.

    Iterates the shorter vector, so the cost is O(min(len(a), len(b))) - for a
    short query against a long document, the difference between scanning a
    handful of terms and scanning the whole document.

    ``math.fsum`` rather than ``sum``: floating-point addition is not
    associative, so a plain sum makes the score depend on dict iteration order.
    That is a reproducibility bug in an eval, where a score that moves in the
    last bits silently reorders a tie and changes recall@k between runs.
    """
    if len(a) > len(b):
        a, b = b, a
    return math.fsum(weight * b[term] for term, weight in a.items() if term in b)


def l2_norm(vector: SparseVector) -> float:
    """Euclidean length of a sparse vector."""
    return math.sqrt(math.fsum(w * w for w in vector.values()))


def l2_normalize(vector: SparseVector) -> dict[str, float]:
    """Scale to unit length; a zero vector normalises to the empty vector.

    Normalising is what stops a long document from outranking a short one purely
    by having more words: afterwards only the *direction* of the term
    distribution matters, not its magnitude.
    """
    norm = l2_norm(vector)
    if norm == 0.0:
        return {}
    return {term: weight / norm for term, weight in vector.items()}


def cosine(a: SparseVector, b: SparseVector) -> float:
    """Cosine similarity in ``[-1, 1]``; ``0.0`` if either vector is empty.

    Kept general - it divides by both norms - so it stays correct for an
    embedder that does not normalise its output. For :class:`TfidfEmbedder`
    vectors both norms are 1.0 and this is exactly the dot product.

    The result is clamped because accumulated float error can put a vector's
    similarity with itself at 1.0000000000000002, and a "similarity" above 1 is
    the kind of impossible number that costs someone an hour downstream.
    """
    norm_a = l2_norm(a)
    norm_b = l2_norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot(a, b) / (norm_a * norm_b)))


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------


class TfidfEmbedder:
    """A TF-IDF vector space model over the shared BM25 tokenizer.

    Fitting is *incremental*, matching :meth:`~aegis.retrieval.sparse.BM25Index.add`:
    each :meth:`fit` call folds more documents into the term statistics rather
    than replacing them. The consequence is worth stating, because it is a real
    property of the model and not a wart - adding a document changes the IDF of
    every term it contains, so vectors computed before that addition are stale.
    The index owning this embedder is responsible for re-embedding; see
    :meth:`~aegis.retrieval.dense.VectorIndex.add`.
    """

    def __init__(self, *, sublinear_tf: bool = True) -> None:
        self.sublinear_tf = sublinear_tf
        """Damp the term-frequency curve. Off means raw counts, which lets a
        keyword-stuffed document dominate - the very failure sublinear tf exists
        to prevent, kept as a knob so a test can demonstrate the difference."""

        self._doc_freq: Counter[str] = Counter()
        self._n_docs: int = 0

    # -- fitted state ----------------------------------------------------

    @property
    def dimension(self) -> int:
        """Vocabulary size.

        Unlike a neural encoder's fixed 384 or 768, this grows with the corpus.
        That is what it means for the space to be spanned by observed terms
        rather than by learned axes, and it is worth surfacing rather than
        hiding behind a constant.
        """
        return len(self._doc_freq)

    @property
    def corpus_size(self) -> int:
        """Documents folded into the statistics so far."""
        return self._n_docs

    @property
    def is_fitted(self) -> bool:
        return self._n_docs > 0

    @property
    def vocabulary(self) -> tuple[str, ...]:
        """Known terms, sorted, so a dump of the model is diffable."""
        return tuple(sorted(self._doc_freq))

    def document_frequency(self, term: str) -> int:
        """How many fitted documents contain ``term`` (already stemmed)."""
        return self._doc_freq.get(term, 0)

    def idf(self, term: str) -> float:
        """Smoothed inverse document frequency. Public so the model is auditable.

        A term appearing in every document scores exactly 1.0 - the floor - and
        rarer terms score above it. Being able to assert on that from a test is
        the difference between "the ranking looked right" and "the weighting is
        right".
        """
        return math.log((1.0 + self._n_docs) / (1.0 + self.document_frequency(term))) + 1.0

    # -- fitting ---------------------------------------------------------

    def fit(self, corpus: Iterable[str]) -> None:
        """Fold documents into the term statistics. Repeated calls accumulate."""
        for text in corpus:
            self._doc_freq.update(set(tokenize(text)))
            self._n_docs += 1

    def reset(self) -> None:
        """Forget every fitted statistic, for when a corpus is rebuilt."""
        self._doc_freq = Counter()
        self._n_docs = 0

    # -- embedding -------------------------------------------------------

    def embed(self, texts: Sequence[str]) -> list[SparseVector]:
        """Vectorise a batch of texts against the fitted statistics."""
        if not self.is_fitted:
            raise EmbedderNotFittedError(
                "TfidfEmbedder.embed() needs corpus statistics: term weights are relative to "
                "the corpus, so there is no such thing as a TF-IDF vector before fit(). Call "
                "fit(corpus), or index through VectorIndex, which fits as it adds."
            )
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> SparseVector:
        counts = Counter(tokenize(text))
        weights: dict[str, float] = {}
        for term, count in counts.items():
            # Out-of-vocabulary terms are dropped. No document contains them, so
            # they contribute zero to every dot product; keeping them would only
            # inflate the query's norm, scaling all similarities by one constant
            # and leaving the ranking identical.
            if not self.document_frequency(term):
                continue
            tf = 1.0 + math.log(count) if self.sublinear_tf else float(count)
            weights[term] = tf * self.idf(term)
        return l2_normalize(weights)

    def __repr__(self) -> str:
        return f"TfidfEmbedder(docs={self._n_docs}, vocab={self.dimension})"
