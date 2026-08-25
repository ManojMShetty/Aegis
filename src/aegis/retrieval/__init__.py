"""Finding the right chunks: sparse, vector, the fusion of both, and the rerank seam."""

from aegis.retrieval.dense import VECTOR_RETRIEVER, VectorIndex
from aegis.retrieval.embedding import (
    Embedder,
    EmbedderNotFittedError,
    FittableEmbedder,
    SparseVector,
    TfidfEmbedder,
    cosine,
)
from aegis.retrieval.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from aegis.retrieval.rerank import IdentityReranker, Reranker
from aegis.retrieval.retriever import HybridRetriever, RetrievalConfig
from aegis.retrieval.sparse import SPARSE_RETRIEVER, BM25Index, stem, tokenize

__all__ = [
    "DEFAULT_RRF_K",
    "SPARSE_RETRIEVER",
    "VECTOR_RETRIEVER",
    "BM25Index",
    "Embedder",
    "EmbedderNotFittedError",
    "FittableEmbedder",
    "HybridRetriever",
    "IdentityReranker",
    "Reranker",
    "RetrievalConfig",
    "SparseVector",
    "TfidfEmbedder",
    "VectorIndex",
    "cosine",
    "reciprocal_rank_fusion",
    "stem",
    "tokenize",
]
