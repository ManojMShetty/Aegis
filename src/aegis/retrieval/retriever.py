"""The retrieval pipeline: BM25 + vector, fused by RRF, optionally reranked.

WHY HYBRID
----------
The two arms fail in opposite directions. BM25 is precise on tokens that have no
synonyms - ``ERR_4021``, ``send_email``, a version number - and blind to
paraphrase. A vector model scores by term *distribution* rather than by exact
overlap, so it tolerates a document phrasing the same idea with different
emphasis, and it will happily return something semantically adjacent that never
contains the identifier you asked for. Fusing the two is usually the single
largest quality win available in a RAG pipeline, which is why it is the default
here rather than an option.

EVERY STAGE TOGGLES, AND THAT IS THE POINT
------------------------------------------
The security half of this project is measured by turning defence layers on and
off one at a time. Retrieval gets the same treatment: ``bm25`` / ``vector`` /
``hybrid`` / ``hybrid+rerank`` are four values of :class:`RetrievalConfig`, not
four code paths. A claim like "hybrid beats BM25 on this corpus" is only worth
making if the alternative arm is one field away and reads the identical index.

INDEXING IS NOT PART OF THE ABLATION
------------------------------------
:meth:`HybridRetriever.add` always writes to both indexes, whatever the config
says. The toggles are query-time only. If turning off the vector arm also
stopped it being indexed, then switching arms would mean re-ingesting, and any
two arms compared across two ingests differ by more than the arm - the exact
confound the ablation exists to exclude.

WHY A SINGLE ARM SKIPS FUSION
-----------------------------
Running RRF over one ranking preserves the order but replaces every score with
``1 / (k + rank)`` and retags the results ``rrf(bm25)``. The ordering would be
right and the artifact would be misleading: a reader could no longer tell a
single-arm run from a fused one, and the BM25 scores - which are the thing you
actually want when debugging why a chunk ranked where it did - are gone. So one
active arm returns its own results, with its own scores and its own tag.

SECURITY: RETRIEVAL MUST NOT LAUNDER TRUST
------------------------------------------
Every :class:`~aegis.domain.chunk.ScoredChunk` this class returns wraps a
:class:`~aegis.domain.chunk.Chunk` that ingest created. No stage here - not the
indexes, not fusion, not the reranker - constructs a chunk or re-wraps its
:class:`~aegis.domain.trust.Tainted` text, so the tier and provenance that reach
the prompt builder are the ones ingest established from
``config/trust_tiers.yaml``. That is by construction, and
``tests/unit/test_hybrid_retrieval.py`` asserts it end to end, because "by
construction" is a claim that stops being true the first time someone adds a
stage that helpfully normalises the text.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from aegis.domain.chunk import Chunk, ScoredChunk
from aegis.retrieval.dense import VECTOR_RETRIEVER, VectorIndex
from aegis.retrieval.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from aegis.retrieval.rerank import IdentityReranker, Reranker
from aegis.retrieval.sparse import SPARSE_RETRIEVER, BM25Index

__all__ = ["HybridRetriever", "RetrievalConfig"]


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """Which stages run, and how wide. One object per ablation arm."""

    use_sparse: bool = True
    """Run the BM25 arm."""

    use_vector: bool = True
    """Run the vector arm."""

    use_rerank: bool = False
    """Run the second stage. Off by default: with the only shipped reranker
    being :class:`~aegis.retrieval.rerank.IdentityReranker`, turning it on
    changes nothing, and a default that looks like a feature but is a seam
    invites exactly the overclaim ``rerank.py`` argues against."""

    top_k: int = 10
    """Results handed to the caller."""

    candidate_k: int = 50
    """Depth each arm retrieves before fusion. Wider than ``top_k`` on purpose:
    fusion can only promote a chunk that at least one arm surfaced, so a
    candidate pool truncated to ``top_k`` throws away the disagreements that are
    the whole reason to fuse."""

    rrf_k: int = DEFAULT_RRF_K
    """Rank-damping constant. See :mod:`aegis.retrieval.fusion`."""

    weights: dict[str, float] = field(default_factory=dict)
    """Per-arm fusion multipliers, keyed by :data:`SPARSE_RETRIEVER` /
    :data:`VECTOR_RETRIEVER`. Empty means equal weight - start there and only
    tune against measured recall, since a weight tuned by intuition is a thumb
    on the scale wearing a number."""

    def __post_init__(self) -> None:
        if not (self.use_sparse or self.use_vector):
            raise ValueError(
                "at least one of use_sparse/use_vector must be enabled: a retriever with "
                "no arms returns nothing, which in an eval looks like a corpus problem"
            )
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.candidate_k < self.top_k:
            raise ValueError(
                f"candidate_k ({self.candidate_k}) must be at least top_k ({self.top_k}): "
                "fusion cannot return more results than its arms retrieved"
            )
        unknown = set(self.weights) - {SPARSE_RETRIEVER, VECTOR_RETRIEVER}
        if unknown:
            # A misspelled key is silently ignored by fusion, so the arm keeps
            # its default weight and the config lies about what was measured.
            raise ValueError(
                f"unknown fusion weight key(s) {sorted(unknown)}; expected "
                f"{sorted((SPARSE_RETRIEVER, VECTOR_RETRIEVER))}"
            )

    @property
    def arm(self) -> str:
        """Label for this configuration in an ablation table."""
        if self.use_sparse and self.use_vector:
            base = "hybrid"
        else:
            base = SPARSE_RETRIEVER if self.use_sparse else VECTOR_RETRIEVER
        return f"{base}+rerank" if self.use_rerank else base

    # -- the four ablation arms, as named constructors --------------------
    #
    # Spelled out rather than generated, so `RetrievalConfig.sparse_only()` reads
    # as the arm name in the results table and a reviewer can see that the arms
    # differ only in which toggles are set.

    @classmethod
    def sparse_only(cls, *, top_k: int = 10, candidate_k: int = 50) -> RetrievalConfig:
        """BM25 alone - the baseline every hybrid claim is measured against."""
        return cls(use_sparse=True, use_vector=False, top_k=top_k, candidate_k=candidate_k)

    @classmethod
    def vector_only(cls, *, top_k: int = 10, candidate_k: int = 50) -> RetrievalConfig:
        """The vector arm alone."""
        return cls(use_sparse=False, use_vector=True, top_k=top_k, candidate_k=candidate_k)

    @classmethod
    def hybrid(cls, *, top_k: int = 10, candidate_k: int = 50) -> RetrievalConfig:
        """Both arms, fused by RRF. The default."""
        return cls(use_sparse=True, use_vector=True, top_k=top_k, candidate_k=candidate_k)

    @classmethod
    def hybrid_reranked(cls, *, top_k: int = 10, candidate_k: int = 50) -> RetrievalConfig:
        """Both arms plus the second stage - identical to :meth:`hybrid` while
        the only shipped reranker is the identity seam. See :mod:`aegis.retrieval.rerank`."""
        return cls(
            use_sparse=True,
            use_vector=True,
            use_rerank=True,
            top_k=top_k,
            candidate_k=candidate_k,
        )


class HybridRetriever:
    """BM25 and vector retrieval over one corpus, fused and optionally reranked."""

    def __init__(
        self,
        *,
        sparse: BM25Index | None = None,
        vector: VectorIndex | None = None,
        reranker: Reranker | None = None,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.sparse = sparse if sparse is not None else BM25Index()
        self.vector = vector if vector is not None else VectorIndex()
        self.reranker: Reranker = reranker if reranker is not None else IdentityReranker()
        self.config = config if config is not None else RetrievalConfig()

    def __len__(self) -> int:
        return len(self.sparse)

    @property
    def chunks(self) -> tuple[Chunk, ...]:
        return self.sparse.chunks

    def add(self, chunks: Iterable[Chunk]) -> None:
        """Index chunks into both arms in one pass, regardless of config.

        See the module docstring: indexing deliberately ignores the toggles, so
        every ablation arm queries the same corpus.
        """
        batch = list(chunks)
        if not batch:
            return
        self.sparse.add(batch)
        self.vector.add(batch)

    def with_config(self, config: RetrievalConfig) -> HybridRetriever:
        """A view of this same corpus under a different arm.

        The indexes are shared, not copied: switching arms must not mean
        re-ingesting, or the comparison is between two corpora as much as
        between two retrievers.
        """
        return HybridRetriever(
            sparse=self.sparse,
            vector=self.vector,
            reranker=self.reranker,
            config=config,
        )

    def retrieve(self, query: str, *, top_k: int | None = None) -> list[ScoredChunk]:
        """Run the configured stages and return the best ``top_k`` chunks."""
        cfg = self.config
        limit = cfg.top_k if top_k is None else top_k
        if limit <= 0:
            raise ValueError("top_k must be positive")

        rankings: dict[str, Sequence[ScoredChunk]] = {}
        if cfg.use_sparse:
            rankings[SPARSE_RETRIEVER] = self.sparse.search(query, top_k=cfg.candidate_k)
        if cfg.use_vector:
            rankings[VECTOR_RETRIEVER] = self.vector.search(query, top_k=cfg.candidate_k)

        results = self._merge(rankings, cfg)

        if cfg.use_rerank:
            results = self.reranker.rerank(query, results, top_k=limit)

        return results[:limit]

    # -- internals -------------------------------------------------------

    @staticmethod
    def _merge(
        rankings: Mapping[str, Sequence[ScoredChunk]],
        cfg: RetrievalConfig,
    ) -> list[ScoredChunk]:
        """Fuse the active arms, or pass a lone arm through untouched."""
        if len(rankings) == 1:
            (only,) = rankings.values()
            return list(only)
        return reciprocal_rank_fusion(
            rankings,
            k=cfg.rrf_k,
            weights=cfg.weights or None,
        )

    def __repr__(self) -> str:
        return f"HybridRetriever(arm={self.config.arm!r}, chunks={len(self)})"
