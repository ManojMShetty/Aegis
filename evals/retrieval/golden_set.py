"""The golden set: hand-labelled queries, the corpus they run against, and the loader.

WHAT A GOLDEN SET IS, AND WHAT THIS ONE IS NOT
----------------------------------------------
A golden set is a list of queries, each paired with the chunk ids a human decided
were the right answers. Given one, a retriever can be scored instead of admired.

**This one is a fixture we wrote, not a benchmark someone else wrote.** It is
twenty-five queries over ten short documents about an invented product. It was
built by the same people who built the retriever, which is precisely the
conflict of interest :mod:`evals` exists to avoid, and it is stated here rather
than buried: a number measured on this file is a *sanity check that the pipeline
works and that the arms differ*, not evidence about how Aegis would do on real
traffic. BEIR, MS MARCO and LoTTE are the external standards; none of them fits
in a repository that must run offline in CI with no download, so the honest move
is a small local fixture with the limitation printed next to the result. The
README says the same thing beside the table.

WHY THE CORPUS LIVES IN THE SAME FILE AS THE LABELS
---------------------------------------------------
Documents and relevance judgements are one artifact, not two. If the corpus were
a directory of markdown files and the labels a JSON beside it, then editing a
sentence in a document could silently move a chunk boundary and re-point every
label after it, and nothing would fail - the eval would just quietly grade a
different question. One file cannot half-change.

WHAT A "CHUNK ID" MEANS IN THIS FILE
------------------------------------
:attr:`Chunk.chunk_id <aegis.domain.chunk.Chunk.chunk_id>` is
``doc_id#ordinal:sha256[:12]`` - content-addressed, so it changes whenever the
text does. That is exactly right for the corpus and impossible for a human to
write down. The golden set therefore labels the stable prefix,
``doc_id#ordinal``, produced by :func:`chunk_key`. A person can read the JSON,
count the sections in the document above it, and check a label by eye.

The cost of dropping the hash is that a label no longer pins the *bytes* of the
chunk, only its position - so :func:`build_corpus` re-derives every key from the
freshly ingested corpus and **raises** if a labelled key is missing. A golden set
that points at a chunk the corpus does not contain is a broken fixture, and the
alternative to raising is scoring that query zero, which reads as a retrieval
failure. The chunker settings are pinned in the JSON for the same reason: change
them and the ordinals move, so the file that owns the labels owns the split too.

THE CORPUS IS INGESTED, NOT FAKED
---------------------------------
:func:`build_corpus` runs the real
:class:`~aegis.ingest.pipeline.IngestPipeline` against the real
``config/trust_tiers.yaml``, with source URIs under ``file://corpus/`` so the
policy labels them T2_CURATED. Constructing :class:`~aegis.domain.chunk.Chunk`
objects directly here would have been three lines shorter and would have meant
the eval measured a corpus that never passed through the code path production
uses.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aegis.config.policy import SecurityPolicy
from aegis.domain.chunk import Chunk
from aegis.ingest.chunker import RecursiveChunker
from aegis.ingest.pipeline import Document, IngestPipeline, IngestReport
from aegis.retrieval.retriever import HybridRetriever

__all__ = [
    "DEFAULT_GOLDEN_SET_PATH",
    "ChunkerSpec",
    "GoldenDocument",
    "GoldenQuery",
    "GoldenSet",
    "GoldenSetError",
    "build_corpus",
    "chunk_key",
    "load_golden_set",
]

DEFAULT_GOLDEN_SET_PATH = Path(__file__).with_name("golden_set.json")
"""The fixture committed beside this module. Small enough to run in CI in well
under a second, which is the only reason it is a fixture and not a download."""

CORPUS_URI_PREFIX = "file://corpus/"
"""Prefix every fixture document is ingested under.

Matches the ``file://corpus/*`` rule in ``config/trust_tiers.yaml``, so the eval
corpus is labelled T2_CURATED by the policy rather than by this module. Where a
document's trust comes from is a security decision, and it stays in the policy
file even when the document is a test fixture.
"""


class GoldenSetError(ValueError):
    """The golden set file is not usable as written.

    Raised eagerly, and never downgraded to a warning. Every condition it covers
    - a missing field, a duplicate id, an empty relevance list, a label pointing
    at a chunk that does not exist - has a plausible-looking failure mode where
    the eval still produces a table, and the table is then measuring something
    other than what its column headings say.
    """


def chunk_key(chunk: Chunk) -> str:
    """The stable, human-writable identity of a chunk: ``doc_id#ordinal``.

    See the module docstring for why the content hash is dropped and what
    :func:`build_corpus` does to compensate.
    """
    return f"{chunk.doc_id}#{chunk.ordinal}"


@dataclass(frozen=True, slots=True)
class ChunkerSpec:
    """The chunker settings the labels were written against.

    Pinned in the JSON rather than defaulted in code: ordinals are the labels, and
    a chunker default that changed in ``aegis.ingest.chunker`` would re-point
    every label in the file without touching the file. With the settings here, the
    same change makes :func:`build_corpus` raise instead.
    """

    max_tokens: int = 512
    overlap_tokens: int = 64
    respect_headings: bool = True

    def build(self) -> RecursiveChunker:
        return RecursiveChunker(
            max_tokens=self.max_tokens,
            overlap_tokens=self.overlap_tokens,
            respect_headings=self.respect_headings,
        )


@dataclass(frozen=True, slots=True)
class GoldenDocument:
    """One corpus document, exactly as it will be ingested."""

    doc_id: str
    text: str

    @property
    def source_uri(self) -> str:
        """Derived, not stored, so no document in the fixture can be given a URI
        that lands it in a different trust tier than the rest of the corpus."""
        return f"{CORPUS_URI_PREFIX}{self.doc_id}"

    def as_document(self) -> Document:
        return Document(text=self.text, source_uri=self.source_uri, doc_id=self.doc_id)


@dataclass(frozen=True, slots=True)
class GoldenQuery:
    """One query and the chunk keys a human judged relevant to it."""

    query_id: str
    text: str
    relevant: frozenset[str]
    note: str = ""
    """Why these chunks and not others. Optional in the schema, present on every
    entry in the committed fixture: a relevance judgement nobody wrote down is a
    judgement nobody can dispute."""


@dataclass(frozen=True, slots=True)
class GoldenSet:
    """A loaded, validated fixture."""

    name: str
    description: str
    chunker: ChunkerSpec
    documents: tuple[GoldenDocument, ...]
    queries: tuple[GoldenQuery, ...]
    source: str = "<memory>"

    @property
    def labelled_keys(self) -> frozenset[str]:
        """Every chunk key named by any query."""
        return frozenset().union(*(query.relevant for query in self.queries))

    def __repr__(self) -> str:
        return f"GoldenSet({self.name!r}, docs={len(self.documents)}, queries={len(self.queries)})"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _require(payload: Mapping[str, Any], field: str, origin: str) -> Any:
    if field not in payload:
        raise GoldenSetError(f"{origin}: missing required field {field!r}")
    return payload[field]


def _as_str(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GoldenSetError(f"{where}: expected a non-empty string, got {value!r}")
    return value


def _parse_chunker(payload: Any, origin: str) -> ChunkerSpec:
    if not isinstance(payload, Mapping):
        raise GoldenSetError(f"{origin}: 'chunker' must be an object")
    unknown = set(payload) - {"max_tokens", "overlap_tokens", "respect_headings"}
    if unknown:
        raise GoldenSetError(f"{origin}: unknown chunker setting(s) {sorted(unknown)}")
    try:
        return ChunkerSpec(
            max_tokens=int(payload.get("max_tokens", 512)),
            overlap_tokens=int(payload.get("overlap_tokens", 64)),
            respect_headings=bool(payload.get("respect_headings", True)),
        )
    except (TypeError, ValueError) as exc:
        raise GoldenSetError(f"{origin}: bad chunker settings: {exc}") from exc


def _parse_documents(payload: Any, origin: str) -> tuple[GoldenDocument, ...]:
    if not isinstance(payload, Sequence) or isinstance(payload, str) or not payload:
        raise GoldenSetError(f"{origin}: 'documents' must be a non-empty list")
    documents: list[GoldenDocument] = []
    seen: set[str] = set()
    for index, entry in enumerate(payload):
        where = f"{origin}: documents[{index}]"
        if not isinstance(entry, Mapping):
            raise GoldenSetError(f"{where}: expected an object")
        doc_id = _as_str(_require(entry, "doc_id", where), f"{where}.doc_id")
        if doc_id in seen:
            # Two documents under one id would silently share a chunk-key
            # namespace, and a label would then name two different chunks.
            raise GoldenSetError(f"{where}: duplicate doc_id {doc_id!r}")
        seen.add(doc_id)
        documents.append(
            GoldenDocument(doc_id=doc_id, text=_as_str(_require(entry, "text", where), where))
        )
    return tuple(documents)


def _parse_queries(payload: Any, origin: str) -> tuple[GoldenQuery, ...]:
    if not isinstance(payload, Sequence) or isinstance(payload, str) or not payload:
        raise GoldenSetError(f"{origin}: 'queries' must be a non-empty list")
    queries: list[GoldenQuery] = []
    seen: set[str] = set()
    for index, entry in enumerate(payload):
        where = f"{origin}: queries[{index}]"
        if not isinstance(entry, Mapping):
            raise GoldenSetError(f"{where}: expected an object")
        query_id = _as_str(_require(entry, "id", where), f"{where}.id")
        if query_id in seen:
            raise GoldenSetError(f"{where}: duplicate query id {query_id!r}")
        seen.add(query_id)

        raw_relevant = _require(entry, "relevant_chunk_ids", where)
        if not isinstance(raw_relevant, Sequence) or isinstance(raw_relevant, str):
            raise GoldenSetError(f"{where}: 'relevant_chunk_ids' must be a list")
        relevant = frozenset(_as_str(item, f"{where}.relevant_chunk_ids") for item in raw_relevant)
        if not relevant:
            # See evals.retrieval.metrics: such a query grades nothing, every
            # metric on it is undefined, and letting it in would put a row in the
            # table that no retriever can influence.
            raise GoldenSetError(
                f"{where}: query {query_id!r} has no relevant chunks; a query with "
                "nothing to find cannot grade a retriever"
            )
        if len(relevant) != len(raw_relevant):
            raise GoldenSetError(f"{where}: 'relevant_chunk_ids' contains a duplicate")

        queries.append(
            GoldenQuery(
                query_id=query_id,
                text=_as_str(_require(entry, "query", where), f"{where}.query"),
                relevant=relevant,
                note=str(entry.get("note", "")),
            )
        )
    return tuple(queries)


def load_golden_set(path: Path | str = DEFAULT_GOLDEN_SET_PATH) -> GoldenSet:
    """Read and validate a golden set file.

    Validation is deliberately noisy. Every check here guards a failure that would
    otherwise still produce a table: a duplicate id makes one label unreachable, an
    empty relevance list adds a row nothing can score, an unknown chunker setting
    means the ordinals in the file were written against a different split.

    Raises:
        GoldenSetError: on any schema problem, with the JSON path that caused it.
    """
    location = Path(path)
    try:
        raw = json.loads(location.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GoldenSetError(f"golden set not found: {location}") from exc
    except json.JSONDecodeError as exc:
        raise GoldenSetError(f"{location}: not valid JSON: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise GoldenSetError(f"{location}: top level must be a JSON object")

    origin = str(location)
    return GoldenSet(
        name=_as_str(_require(raw, "name", origin), f"{origin}.name"),
        description=_as_str(_require(raw, "description", origin), f"{origin}.description"),
        chunker=_parse_chunker(raw.get("chunker", {}), origin),
        documents=_parse_documents(_require(raw, "documents", origin), origin),
        queries=_parse_queries(_require(raw, "queries", origin), origin),
        source=origin,
    )


# ---------------------------------------------------------------------------
# Corpus construction
# ---------------------------------------------------------------------------


def build_corpus(
    golden: GoldenSet,
    *,
    policy: SecurityPolicy | None = None,
) -> tuple[HybridRetriever, IngestReport]:
    """Ingest the fixture into a retriever, and check every label resolves.

    The retriever is returned with the default :class:`RetrievalConfig` - the
    caller picks an arm with
    :meth:`HybridRetriever.with_config <aegis.retrieval.retriever.HybridRetriever.with_config>`,
    which shares these indexes rather than re-ingesting. That is what keeps the
    four arms honest: they read one corpus, built once.

    Raises:
        GoldenSetError: if any labelled chunk key is absent from the ingested
            corpus. See the module docstring on why this raises rather than
            scoring the query zero.
    """
    retriever = HybridRetriever()
    pipeline = IngestPipeline(
        retriever=retriever,
        policy=policy if policy is not None else SecurityPolicy.load(),
        chunker=golden.chunker.build(),
    )
    report = pipeline.ingest(document.as_document() for document in golden.documents)

    available = {chunk_key(chunk) for chunk in retriever.chunks}
    missing = sorted(golden.labelled_keys - available)
    if missing:
        raise GoldenSetError(
            f"{golden.source}: {len(missing)} labelled chunk id(s) are not in the "
            f"ingested corpus: {missing[:5]}"
            + (" ..." if len(missing) > 5 else "")
            + ". The document text or the chunker settings changed since the "
            "labels were written; re-check them rather than letting these "
            "queries score zero."
        )
    return retriever, report
