"""The reranking seam - and an explicit refusal to fake what fills it.

WHAT RERANKING IS
-----------------
First-stage retrieval has to score every chunk in the corpus, so it can only
afford cheap scoring: term overlap, or a dot product against a vector computed
without seeing the query. A reranker gives up that cheapness in exchange for
accuracy, re-scoring only the ~50 candidates the first stage returned. The
standard instrument is a cross-encoder: query and passage are concatenated into
one sequence and pushed through a transformer, so every query token attends to
every passage token. That joint attention is exactly what a bi-encoder cannot
do, and it is why reranking typically buys several points of nDCG.

WHY THERE IS NO CROSS-ENCODER HERE
----------------------------------
A cross-encoder is a transformer. There is no version of it that runs on the
standard library, and this project's hard constraint is that everything runs
offline, in CI, with no model download. So the honest options were: ship the
seam with an identity implementation and say so, or write something that scores
candidates by some cheap heuristic and call it a "reranker".

The second option is the one worth naming explicitly, because it is what an
impressive-looking pipeline would do. Re-scoring RRF output by term overlap
would produce a plausible number, a toggle in the ablation table, and a row in
the results that means nothing - the second stage would be re-reading the same
lexical signal the first stage already used, which is not reranking, it is
double-counting. It would also make ``rerank: on`` look like a measured
improvement when it is an artifact.

SO THIS IS A SEAM, NOT A FEATURE
--------------------------------
:class:`Reranker` is a real interface with a real contract, and
:class:`IdentityReranker` is a documented no-op that fills it. A toggle whose
only implementation is identity is a seam. It earns its place by making the
integration point explicit and tested - where reranking sits in the pipeline,
what it may and may not do to a result - so that a deployment with torch
available implements one class and changes no other code. It does NOT earn a
row in a results table, and the ablation must not report "hybrid+rerank" as an
arm distinct from "hybrid" while this is the only implementation, because with
identity reranking the two arms are the same system and any gap between them is
noise.

THE CONTRACT
------------
A reranker may reorder and it may truncate. It may NOT rewrite, re-wrap or
re-create a chunk: the :class:`~aegis.domain.chunk.Chunk` that comes out must be
the object that went in, carrying the tier and provenance the ingest stage
established. A stage that rebuilt its chunks would be a place where untrusted
content could quietly stop looking untrusted, which is the one thing no part of
this pipeline is allowed to be.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from aegis.domain.chunk import ScoredChunk

__all__ = ["IdentityReranker", "Reranker"]


@runtime_checkable
class Reranker(Protocol):
    """Re-scores a shortlist of candidates against the query.

    Implementations must satisfy the contract in this module's docstring: reorder
    and truncate freely, but return the very chunks that were passed in.
    """

    @property
    def name(self) -> str:
        """Identifier recorded in eval artifacts, so a result names its reranker."""

    def rerank(
        self,
        query: str,
        results: Sequence[ScoredChunk],
        *,
        top_k: int | None = None,
    ) -> list[ScoredChunk]:
        """Return ``results`` re-ordered, best first, truncated to ``top_k``."""
        ...


@dataclass(frozen=True, slots=True)
class IdentityReranker:
    """Passes candidates through untouched. The seam's placeholder, not a model.

    Its value is negative space: it holds the position in the pipeline open and
    lets the wiring be tested (that the stage is called, that it may truncate,
    that trust survives it) without pretending a second-stage model exists. If
    an eval reports a difference between ``hybrid`` and ``hybrid+rerank`` while
    this is installed, the difference is a bug in the harness - the two are
    running the identical ranking.
    """

    name: str = "identity"

    def rerank(
        self,
        query: str,
        results: Sequence[ScoredChunk],
        *,
        top_k: int | None = None,
    ) -> list[ScoredChunk]:
        """Return the candidates unchanged, truncated to ``top_k`` if given."""
        ordered = list(results)
        return ordered if top_k is None else ordered[:top_k]
