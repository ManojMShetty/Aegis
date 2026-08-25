"""The vector arm of the hybrid retriever: cosine ranking over an Embedder.

WHY THE FILE IS CALLED ``dense`` AND THE RESULTS ARE TAGGED ``vector``
----------------------------------------------------------------------
"Dense retrieval" is the literature's name for the *role* this stage plays -
the arm that scores by vector similarity rather than by term overlap, and the
counterweight to BM25 in every hybrid system since Karpukhin et al. (2020). The
file keeps that name because that is the slot it fills.

The results are tagged ``vector``, not ``dense``, because with the default
:class:`~aegis.retrieval.embedding.TfidfEmbedder` the vectors are sparse and
lexical-semantic. Writing "dense" into an eval artifact would tell a reader that
a neural encoder produced the number. It did not. Swap in a neural
:class:`~aegis.retrieval.embedding.Embedder` and nothing here changes - which is
the honest version of the claim this module can make.

SAME SHAPE AS BM25, ON PURPOSE
------------------------------
:meth:`search` returns :class:`~aegis.domain.chunk.ScoredChunk` exactly as
:class:`~aegis.retrieval.sparse.BM25Index` does, so
:func:`~aegis.retrieval.fusion.reciprocal_rank_fusion` consumes both without
knowing which is which, and so an ablation can substitute one for the other
without a second code path. The two indexes also read the same
:attr:`~aegis.domain.chunk.Chunk.searchable_text` and share one tokenizer, so
they genuinely rank the same corpus rather than two subtly different ones.

WHY ADDING DOCUMENTS RE-EMBEDS THE OLD ONES
-------------------------------------------
A TF-IDF weight is relative to the corpus: index a new document containing
"widget" and the IDF of "widget" drops for every document already stored. So
when the embedder is fittable, :meth:`add` re-embeds everything rather than
appending vectors computed under stale statistics. That is O(N) per add, which
is the right trade for benchmark-sized in-memory corpora and matches
:meth:`~aegis.retrieval.sparse.BM25Index.add`, which likewise recomputes its
statistics. A persistent backend (pgvector) would batch or defer it; the
interface is what has to survive that swap, not the internals.

SECURITY
--------
Nothing in this module constructs a :class:`~aegis.domain.chunk.Chunk`. Scoring
wraps the chunk it was given, so tier and provenance reach the caller as the
same object that was indexed. Retrieval ranks untrusted content; it must never
be a place where untrusted content stops looking untrusted.
"""

from __future__ import annotations

from collections.abc import Iterable

from aegis.domain.chunk import Chunk, ScoredChunk
from aegis.retrieval.embedding import (
    Embedder,
    FittableEmbedder,
    SparseVector,
    TfidfEmbedder,
    cosine,
)

__all__ = ["VECTOR_RETRIEVER", "VectorIndex"]

VECTOR_RETRIEVER = "vector"
"""Name this index tags its results with, and the key fusion sees it under."""


class VectorIndex:
    """An in-memory vector index over :class:`~aegis.domain.chunk.Chunk` objects.

    In-memory matches :class:`~aegis.retrieval.sparse.BM25Index`: the corpora
    this project evaluates on are benchmark-sized, and it keeps the pipeline
    runnable with no database and no model download. An exhaustive cosine scan
    is also exact, which a pgvector HNSW index is not - worth remembering when
    recall numbers move after a backend swap.
    """

    def __init__(self, embedder: Embedder | None = None) -> None:
        self._embedder: Embedder = embedder if embedder is not None else TfidfEmbedder()
        self._chunks: list[Chunk] = []
        self._vectors: list[SparseVector] = []

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    @property
    def chunks(self) -> tuple[Chunk, ...]:
        return tuple(self._chunks)

    @property
    def dimension(self) -> int:
        """Dimensionality of the space these vectors live in."""
        return self._embedder.dimension

    def add(self, chunks: Iterable[Chunk]) -> None:
        """Index chunks, fitting the embedder first when it needs the corpus."""
        new = list(chunks)
        if not new:
            return

        self._chunks.extend(new)
        embedder = self._embedder

        if isinstance(embedder, FittableEmbedder):
            # Fit on the new documents only - fitting accumulates, so re-feeding
            # the whole corpus would double-count every document already seen
            # and distort every IDF. Then re-embed everything, because those
            # IDFs have just changed underneath the stored vectors.
            embedder.fit(chunk.searchable_text for chunk in new)
            self._vectors = embedder.embed([chunk.searchable_text for chunk in self._chunks])
        else:
            self._vectors.extend(embedder.embed([chunk.searchable_text for chunk in new]))

    def search(self, query: str, *, top_k: int = 50) -> list[ScoredChunk]:
        """Return the ``top_k`` chunks closest to ``query``, best first."""
        if not self._chunks:
            return []

        query_vector = self._embedder.embed([query])[0]
        if not query_vector:
            # Every query term was a stopword or absent from the vocabulary, so
            # the query has no direction in this space. Returning nothing is
            # honest; returning everything at score 0.0 would let fusion promote
            # arbitrary chunks on rank alone.
            return []

        scored: list[ScoredChunk] = []
        for chunk, vector in zip(self._chunks, self._vectors, strict=True):
            score = cosine(query_vector, vector)
            if score > 0.0:
                scored.append(ScoredChunk(chunk=chunk, score=score, retriever=VECTOR_RETRIEVER))

        # Tie-break on chunk_id so the ranking is reproducible; an unstable sort
        # silently changes recall@k between otherwise identical eval runs.
        scored.sort(key=lambda s: (-s.score, s.chunk.chunk_id))
        return scored[:top_k]

    def __repr__(self) -> str:
        return f"VectorIndex(chunks={len(self._chunks)}, embedder={self._embedder!r})"
