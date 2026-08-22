"""Splitting documents into retrievable chunks.

WHY CHUNKING IS THE HIGHEST-LEVERAGE RETRIEVAL DECISION
-------------------------------------------------------
You cannot embed a 50-page document as one vector, so you split it. That split
decides the ceiling on retrieval quality:

* Chunks too small  -> precise matching, but the answer gets cut in half and the
                       model never sees enough context to use it.
* Chunks too large  -> rich context, but the embedding averages over many topics
                       ("dilution") and you retrieve noise.

There is no universal best size. The honest answer is that you pick a strategy,
then *measure* it (recall@k, nDCG) and tune. This module exists to make the
strategy a knob rather than a hard-coded accident.

RECURSIVE SPLITTING
-------------------
We split on the most meaningful boundary that still fits: paragraphs first, then
sentences, then whitespace, and only as a last resort mid-token. That keeps
semantically whole units together far more often than fixed-width slicing does.

SECURITY NOTE
-------------
Chunking is a pure transform, so every chunk inherits the document's tier and
provenance (``Tainted.map`` cannot raise trust). Splitting a poisoned document
produces poisoned chunks - an attacker cannot launder content by getting it
sliced, and cannot hide by straddling a boundary, because the whole document was
untrusted to begin with.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from aegis.domain.chunk import Chunk
from aegis.domain.trust import Tainted, TrustTier

__all__ = ["RecursiveChunker", "chunk_document", "estimate_tokens"]

# Split candidates, most-preferred boundary first.
_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ")

_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$", re.MULTILINE)

# Rough chars-per-token for English prose. Deliberately an estimate: a real
# tokenizer is a heavy dependency and the exact count does not change the design.
# Swap in `messages.count_tokens` when precision matters for a cost estimate.
_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Approximate token count. See ``_CHARS_PER_TOKEN`` for the caveat."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


@dataclass(frozen=True, slots=True)
class RecursiveChunker:
    """Splits text on the best available boundary that fits the size budget."""

    max_tokens: int = 512
    """Target chunk size. 512 is a common default: big enough to hold a complete
    idea, small enough that the embedding is not diluted."""

    overlap_tokens: int = 64
    """Repeat this much of the previous chunk at the start of the next one, so a
    fact spanning a boundary still appears whole in at least one chunk."""

    respect_headings: bool = True
    """Split at markdown headings first and tag each chunk with its section."""

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.overlap_tokens < 0:
            raise ValueError("overlap_tokens cannot be negative")
        if self.overlap_tokens >= self.max_tokens:
            # Otherwise each chunk re-emits its predecessor and the walk cannot
            # advance - a silent infinite loop in the worst case.
            raise ValueError("overlap_tokens must be smaller than max_tokens")

    @property
    def _max_chars(self) -> int:
        return self.max_tokens * _CHARS_PER_TOKEN

    @property
    def _overlap_chars(self) -> int:
        return self.overlap_tokens * _CHARS_PER_TOKEN

    def split(self, text: str) -> list[str]:
        """Split raw text into size-bounded pieces (no trust handling here)."""
        text = text.strip()
        if not text:
            return []
        if len(text) <= self._max_chars:
            return [text]
        return list(self._merge_with_overlap(self._split_recursive(text)))

    # -- internals -------------------------------------------------------

    def _split_recursive(self, text: str, sep_index: int = 0) -> list[str]:
        """Break ``text`` into pieces that each fit, preferring early separators."""
        if len(text) <= self._max_chars:
            return [text]

        if sep_index >= len(_SEPARATORS):
            # No separator worked: hard-slice. Rare, and better than emitting an
            # oversized chunk that will be silently truncated downstream.
            return [text[i : i + self._max_chars] for i in range(0, len(text), self._max_chars)]

        sep = _SEPARATORS[sep_index]
        parts = text.split(sep)
        if len(parts) == 1:
            return self._split_recursive(text, sep_index + 1)

        out: list[str] = []
        for i, part in enumerate(parts):
            piece = part + sep if i < len(parts) - 1 else part
            if not piece.strip():
                continue
            if len(piece) <= self._max_chars:
                out.append(piece)
            else:
                out.extend(self._split_recursive(piece, sep_index + 1))
        return out

    def _merge_with_overlap(self, pieces: Sequence[str]) -> Iterator[str]:
        """Greedily pack pieces up to the budget, carrying overlap forward."""
        buf = ""
        for piece in pieces:
            if buf and len(buf) + len(piece) > self._max_chars:
                yield buf.strip()
                buf = (buf[-self._overlap_chars :] if self._overlap_chars else "") + piece
            else:
                buf += piece
        if buf.strip():
            yield buf.strip()


def _sections(text: str) -> list[tuple[str, str]]:
    """Split markdown into ``(heading, body)`` pairs, preserving pre-heading text."""
    matches = list(_HEADING.finditer(text))
    if not matches:
        return [("", text)]

    out: list[tuple[str, str]] = []
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            out.append(("", preamble))

    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end() : end].strip()
        if body:
            out.append((m.group(2).strip(), body))
    return out


def chunk_document(
    text: str,
    *,
    doc_id: str,
    source_uri: str,
    tier: TrustTier,
    chunker: RecursiveChunker | None = None,
    metadata: dict[str, str] | None = None,
) -> list[Chunk]:
    """Split a document into trust-labelled :class:`Chunk` objects.

    Every chunk inherits ``tier`` and the document's provenance. This is the
    single place where a document's trust level is *established*, which keeps
    that decision small and reviewable (see ``config/trust_tiers.yaml``).
    """
    ch = chunker or RecursiveChunker()
    doc = Tainted.trusted(text, tier, source_uri=source_uri)

    sections = _sections(text) if ch.respect_headings else [("", text)]

    chunks: list[Chunk] = []
    ordinal = 0
    for heading, body in sections:
        for piece in ch.split(body):
            chunks.append(
                Chunk(
                    # with_value() preserves tier + provenance: a slice of
                    # untrusted text is still untrusted text.
                    text=doc.with_value(piece),
                    doc_id=doc_id,
                    ordinal=ordinal,
                    heading=heading,
                    metadata=dict(metadata or {}),
                )
            )
            ordinal += 1
    return chunks
