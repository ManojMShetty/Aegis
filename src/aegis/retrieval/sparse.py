"""Sparse (keyword) retrieval with BM25.

WHY KEEP KEYWORD SEARCH AT ALL
------------------------------
Embeddings match *meaning*, which is exactly what you want for "how do I reset
my password" -> "account recovery steps". But they are unreliable on the things
that have no synonyms: product codes, function names, error identifiers, rare
proper nouns, version numbers. Ask a dense retriever for ``ERR_4021`` and it will
cheerfully return semantically similar prose that never mentions it.

BM25 has the opposite profile - excellent on exact tokens, blind to meaning.
Which is why hybrid retrieval (this, plus dense, fused by RRF) beats either
alone, and why that single change is usually the biggest quality win in a RAG
pipeline.

IMPLEMENTED DIRECTLY, ON PURPOSE
--------------------------------
BM25 is ~40 lines. Writing it keeps the retrieval core dependency-free (matching
the security core's design), makes the ranking auditable rather than a black box,
and means the whole pipeline still runs with no ML stack installed.

THE FORMULA
-----------
For query term ``q`` and document ``D``::

    score(D, q) = IDF(q) * ( f(q,D) * (k1 + 1) )
                           / ( f(q,D) + k1 * (1 - b + b * |D| / avgdl) )

* ``f(q,D)``  term frequency - more mentions means more relevant, but with
              *saturation*: the 10th mention adds far less than the 2nd.
* ``k1``      how fast that saturation kicks in (1.2-2.0 typical).
* ``b``       how much to penalise long documents (0.75 typical). Without it,
              long documents win simply by containing more words.
* ``IDF``     rare terms carry more signal than common ones.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence

from aegis.domain.chunk import Chunk, ScoredChunk

__all__ = ["SPARSE_RETRIEVER", "BM25Index", "stem", "tokenize"]

SPARSE_RETRIEVER = "bm25"
"""Name this index tags its results with, and the key fusion sees it under.

A constant rather than a literal because the retriever, the fusion weights and
the ablation table all have to spell it the same way; a typo in a weights dict
silently applies the default weight instead of raising.
"""

_TOKEN = re.compile(r"[a-z0-9_]+")

# Common English words carry little signal and appear in nearly every document.
# Kept deliberately short: an aggressive stoplist removes tokens that matter in
# technical text (e.g. "no", "not" flip meaning; "it"/"is" appear in code).
# A prose word list reads far better as text than as 30 quoted literals.
_STOPWORD_TEXT = (
    "a an and are as at be by for from has have he in is it its of on or "
    "that the to was were will with this these those there"
)
_STOPWORDS: frozenset[str] = frozenset(_STOPWORD_TEXT.split())


def stem(token: str) -> str:
    """Conservative suffix stripping, so ``costs`` matches a query for ``cost``.

    Without this, BM25 treats ``cost`` and ``costs`` as unrelated terms and a
    perfectly relevant document loses to a shorter, less relevant one on length
    normalisation alone. That failure is easy to miss because the ranking still
    *looks* plausible - which is exactly why retrieval needs measuring, not
    eyeballing.

    Deliberately lighter than a full Porter stemmer: aggressive stemming
    conflates distinct technical terms, and this index's whole advantage over
    dense retrieval is exact-token precision on identifiers. Tokens of 3 chars
    or fewer, and anything ending in a digit, are left alone so ``ERR_4021`` and
    ``send_email`` survive intact.
    """
    if len(token) <= 3 or token[-1].isdigit():
        return token
    for suffix, replacement, min_stem in (
        ("ies", "y", 1),
        ("sses", "ss", 1),
        ("ing", "", 3),
        ("ed", "", 3),
        ("es", "", 2),
        ("s", "", 2),
    ):
        if token.endswith(suffix):
            stem_part = token[: -len(suffix)]
            if len(stem_part) < min_stem:
                continue
            # "address" must not become "addres": a doubled s is part of the word.
            if suffix == "s" and token.endswith("ss"):
                continue
            return stem_part + replacement
    return token


def tokenize(text: str, *, drop_stopwords: bool = True, apply_stem: bool = True) -> list[str]:
    """Lowercase, split on non-alphanumerics, drop stopwords, and stem.

    Underscores and digits are kept inside tokens so identifiers like
    ``send_email`` and ``ERR_4021`` survive as single searchable units - which is
    precisely the case dense retrieval handles badly.
    """
    tokens = _TOKEN.findall(text.lower())
    if drop_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS]
    if apply_stem:
        tokens = [stem(t) for t in tokens]
    return tokens


class BM25Index:
    """An in-memory BM25 index over :class:`Chunk` objects.

    In-memory is the right default here: the corpora this project evaluates on
    are benchmark-sized, and it keeps the pipeline runnable with no database.
    The interface is the part that matters - a Postgres ``tsvector`` backend can
    replace the internals without changing a caller.
    """

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._tokens: list[list[str]] = []
        self._freqs: list[Counter[str]] = []
        self._doc_freq: Counter[str] = Counter()
        self._avg_len: float = 0.0

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> tuple[Chunk, ...]:
        return tuple(self._chunks)

    def add(self, chunks: Iterable[Chunk]) -> None:
        """Index chunks. Safe to call repeatedly; statistics are recomputed."""
        for chunk in chunks:
            # Chunk.searchable_text prepends the heading for scoring only. It is
            # a property of the chunk rather than of this index precisely so the
            # vector index sees the identical text (see aegis.domain.chunk).
            tokens = tokenize(chunk.searchable_text)
            self._chunks.append(chunk)
            self._tokens.append(tokens)
            self._freqs.append(Counter(tokens))
            self._doc_freq.update(set(tokens))
        self._recompute()

    def _recompute(self) -> None:
        total = sum(len(t) for t in self._tokens)
        self._avg_len = (total / len(self._tokens)) if self._tokens else 0.0

    def _idf(self, term: str) -> float:
        """Inverse document frequency, BM25+ smoothed.

        The ``+1`` inside the log keeps IDF positive: the raw Robertson formula
        goes negative for terms appearing in more than half the corpus, which on
        a small corpus lets a common term *subtract* from a document's score.
        """
        n = len(self._chunks)
        df = self._doc_freq.get(term, 0)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, *, top_k: int = 50) -> list[ScoredChunk]:
        """Return the ``top_k`` best-matching chunks, highest score first."""
        if not self._chunks:
            return []

        terms = tokenize(query)
        if not terms:
            return []

        scored: list[ScoredChunk] = []
        for i, chunk in enumerate(self._chunks):
            freqs = self._freqs[i]
            length = len(self._tokens[i])
            score = 0.0
            for term in terms:
                f = freqs.get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1.0 - self.b + self.b * length / (self._avg_len or 1.0))
                score += self._idf(term) * (f * (self.k1 + 1.0)) / denom
            if score > 0.0:
                scored.append(ScoredChunk(chunk=chunk, score=score, retriever=SPARSE_RETRIEVER))

        scored.sort(key=lambda s: (-s.score, s.chunk.chunk_id))
        return scored[:top_k]


def rank_ids(results: Sequence[ScoredChunk]) -> list[str]:
    """Chunk ids in rank order - the input fusion actually consumes."""
    return [r.chunk_id for r in results]
