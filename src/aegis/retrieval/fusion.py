"""Reciprocal Rank Fusion - combining rankings from different retrievers.

THE PROBLEM WITH COMBINING SCORES
---------------------------------
BM25 returns unbounded relevance scores (0 to ~30, corpus-dependent). Cosine
similarity returns roughly -1 to 1. They are not comparable, and normalising
them is fragile: min-max normalisation makes a result's score depend on whatever
else happened to be in the batch, so the same chunk scores differently depending
on its neighbours.

RRF sidesteps this entirely by throwing the scores away and using only *rank*::

    RRF(d) = sum over retrievers r of  1 / (k + rank_r(d))

A chunk ranked #1 by BM25 and #1 by dense scores highest. A chunk ranked #1 by
one and absent from the other still does well - which is the behaviour you want,
because "found decisively by one method" is real evidence.

WHY k = 60
----------
The constant dampens the difference between top ranks so a single retriever
cannot dominate through overconfidence: with k=60, rank 1 contributes 1/61 and
rank 2 contributes 1/62 - close together, so agreement across retrievers matters
more than one retriever's internal ordering. 60 is the value from the original
paper (Cormack et al., 2009) and remains the standard default.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from aegis.domain.chunk import Chunk, ScoredChunk

__all__ = ["DEFAULT_RRF_K", "reciprocal_rank_fusion"]

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[ScoredChunk]],
    *,
    k: int = DEFAULT_RRF_K,
    top_k: int | None = None,
    weights: Mapping[str, float] | None = None,
) -> list[ScoredChunk]:
    """Fuse several ranked lists into one.

    Args:
        rankings: retriever name -> its ranked results (best first).
        k: rank-damping constant. Larger flattens the contribution curve.
        top_k: truncate the fused list.
        weights: optional per-retriever multiplier, for when one retriever is
            known to be stronger on a given corpus. Defaults to equal weight -
            start there and only tune it against measured recall, since a weight
            tuned by intuition is just a thumb on the scale.

    Returns:
        Chunks ordered by fused score, each tagged with the retrievers that
        found it (e.g. ``"rrf(bm25+dense)"``) so the eval can report *which*
        strategy actually surfaced the answer.
    """
    if not rankings:
        return []

    fused: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    finders: dict[str, list[str]] = {}

    for retriever, results in rankings.items():
        weight = (weights or {}).get(retriever, 1.0)
        for rank, scored in enumerate(results, start=1):
            cid = scored.chunk_id
            fused[cid] = fused.get(cid, 0.0) + weight * (1.0 / (k + rank))
            chunks.setdefault(cid, scored.chunk)
            finders.setdefault(cid, []).append(retriever)

    out = [
        ScoredChunk(
            chunk=chunks[cid],
            score=score,
            retriever=f"rrf({'+'.join(finders[cid])})",
        )
        for cid, score in fused.items()
    ]
    # Tie-break on chunk_id so fusion is deterministic - required for a
    # reproducible eval, where an unstable sort silently changes recall@k.
    out.sort(key=lambda s: (-s.score, s.chunk_id))
    return out[:top_k] if top_k is not None else out
