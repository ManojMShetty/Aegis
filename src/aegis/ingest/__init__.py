"""Turning documents into trust-labelled, retrievable chunks."""

from aegis.ingest.chunker import RecursiveChunker, chunk_document

__all__ = ["RecursiveChunker", "chunk_document"]
