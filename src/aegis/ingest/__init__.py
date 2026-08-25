"""Turning documents into trust-labelled, retrievable chunks."""

from aegis.ingest.chunker import RecursiveChunker, chunk_document
from aegis.ingest.pipeline import Document, IngestPipeline, IngestReport, load_directory

__all__ = [
    "Document",
    "IngestPipeline",
    "IngestReport",
    "RecursiveChunker",
    "chunk_document",
    "load_directory",
]
