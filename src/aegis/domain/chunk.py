"""A retrievable piece of a document, carrying its trust label.

The unit that flows through the whole pipeline: ingest produces chunks,
retrieval ranks them, the detector flags them, the quarantine reads them, and
the citation names them. Because a chunk wraps a :class:`Tainted` value, the
answer to "could an attacker have written this?" travels with the text at every
step rather than being recomputed (or forgotten) downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from aegis.domain.trust import Tainted, TrustTier, sha256_of

__all__ = ["Chunk", "ScoredChunk"]


@dataclass(frozen=True, slots=True)
class Chunk:
    """One retrievable span of text plus everything needed to trust and cite it."""

    text: Tainted[str]
    """The content, wrapped with its tier and provenance."""

    doc_id: str
    """Identifier of the document this came from."""

    ordinal: int = 0
    """Position within the document, so neighbouring chunks can be re-assembled."""

    heading: str = ""
    """Nearest enclosing heading, kept because it is often the best short label
    for a citation and it improves retrieval when prepended to the body."""

    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def value(self) -> str:
        return self.text.value

    @property
    def tier(self) -> TrustTier:
        return self.text.tier

    @property
    def source_uri(self) -> str:
        return self.text.sources[0] if self.text.sources else self.doc_id

    @property
    def chunk_id(self) -> str:
        """Stable identity: same document + position + bytes => same id."""
        return f"{self.doc_id}#{self.ordinal}:{sha256_of(self.value)[:12]}"

    @property
    def searchable_text(self) -> str:
        """What an index should see for this chunk: heading, then body.

        The heading is strong signal about what a chunk is *about* and is
        prepended for scoring only - it is never written back onto the text, so
        the cited content stays exactly what the document said.

        It lives here rather than inside one index because every retriever must
        agree on it. If BM25 indexed the heading and the vector index did not,
        the two arms would score different documents and fusion would combine
        rankings built over different corpora - a defect that never shows up as
        an error, only as quietly worse recall.
        """
        return f"{self.heading} {self.value}" if self.heading else self.value

    @property
    def is_attacker_influenced(self) -> bool:
        return self.text.is_attacker_influenced

    @property
    def detector_flags(self) -> tuple[str, ...]:
        return self.text.detector_flags

    def citation_label(self) -> str:
        """Short human-facing label. Includes the tier on purpose.

        Surfacing "this claim came from an untrusted source" in the citation is
        the visible payoff of carrying provenance all the way through.
        """
        where = self.heading or f"part {self.ordinal + 1}"
        return f"{self.source_uri} ({where}) [{self.tier.label}]"

    def __repr__(self) -> str:
        preview = self.value[:40].replace("\n", " ")
        return f"Chunk({self.chunk_id}, {self.tier.label}, {preview!r})"


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """A chunk with a retrieval score, and where that score came from.

    ``retriever`` is kept so fusion can tell a dense hit from a sparse one, and
    so the eval can report which strategy actually found the answer.
    """

    chunk: Chunk
    score: float
    retriever: str = ""

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    def __repr__(self) -> str:
        return f"ScoredChunk({self.chunk.chunk_id}, {self.score:.4f}, via={self.retriever!r})"
