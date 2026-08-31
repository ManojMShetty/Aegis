"""Load, chunk, label, index - the single pass that turns documents into a corpus.

WHY THIS IS ONE PASS AND NOT FOUR SCRIPTS
-----------------------------------------
Ingest is where a document acquires the two things the rest of the system relies
on: its trust tier and its chunk identity. Splitting that across a loader, a
chunker, a labelling step and two indexers means four places where a document
can arrive labelled differently - and the failure is silent, because a chunk
that came in at the wrong tier still retrieves perfectly well. It is only wrong
later, at the capability gate, where it now looks trustworthy enough to act on.
So the tier is resolved once, from the policy, next to the chunking, and the
result goes into both indexes together.

WHERE TRUST COMES FROM
----------------------
:meth:`SecurityPolicy.resolve_tier <aegis.config.policy.SecurityPolicy.resolve_tier>`
answers "what tier does this source start at?" by matching the source URI
against ``config/trust_tiers.yaml``. This module never decides a tier itself and
never accepts one from a caller: the policy file is the audit surface, and a
pipeline that could override it would make the file decorative. An unmatched
source falls to ``default_tier``, which is UNTRUSTED in the shipped policy and when
the key is absent - so forgetting to add a rule fails closed, at the cost of a
document being less trusted than it deserved rather than more. A policy file that
raises ``default_tier`` gives that up, and the loader does not stop it.

IDEMPOTENCE, AND ITS HONEST LIMIT
---------------------------------
Ingesting the same document twice must not double the corpus - otherwise every
duplicate inflates the document frequencies that both BM25 and TF-IDF weight
with, and the corpus quietly becomes a different corpus on each re-run.
De-duplication is by content hash at two levels: an unchanged document is
skipped before it is chunked, and any chunk whose
:attr:`~aegis.domain.chunk.Chunk.chunk_id` is already present is dropped. That
id is already content-addressed (``doc_id#ordinal:sha256[:12]``), so identical
bytes produce an identical id and re-ingest is a no-op.

The limit, stated plainly: this makes ingest idempotent, not update-correct.
Re-ingesting a *modified* document adds the chunks that changed and leaves the
superseded ones in the index, because both in-memory indexes are append-only.
Deleting is the job of the persistent backend that eventually replaces them
(a Postgres corpus table keyed on ``doc_id``); pretending to support it here by
rebuilding the indexes on every edit would hide the fact that the interface has
no delete.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from aegis.config.policy import SecurityPolicy
from aegis.domain.chunk import Chunk
from aegis.domain.trust import TrustTier, sha256_of
from aegis.ingest.chunker import RecursiveChunker, chunk_document
from aegis.retrieval.retriever import HybridRetriever

__all__ = ["Document", "IngestPipeline", "IngestReport", "TierConflictError", "load_directory"]

DEFAULT_CORPUS_URI_PREFIX = "file://corpus/"
"""URI prefix :func:`load_directory` builds source URIs under.

Chosen to match the ``file://corpus/*`` rule already in
``config/trust_tiers.yaml``: a file loaded from a curated corpus directory is
T2_CURATED, and one loaded from anywhere else matches no rule and falls to
UNTRUSTED. Where a document sits on disk is a trust claim, so it is made in the
policy file rather than in this module.
"""

_TEXT_SUFFIXES: tuple[str, ...] = (".md", ".txt", ".rst")


class TierConflictError(RuntimeError):
    """The same chunk arrived at two different trust tiers.

    Raised rather than resolved, because both resolutions are wrong. Ingest is the
    only place a document's trust level is established, and the capability gate
    acts on whatever it decided - so a corpus that quietly holds a stale tier is a
    corpus that authorises actions on content the current source would not permit.
    """


@dataclass(frozen=True, slots=True)
class Document:
    """A document as it arrives, before it is split or labelled."""

    text: str
    source_uri: str
    """Origin identifier. This, and only this, decides the trust tier."""

    doc_id: str = ""
    """Stable corpus identity. Defaults to :attr:`source_uri`, which is usually
    what you want - two ingests of the same URI are the same document, and that
    is exactly the judgement idempotence rests on."""

    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def identity(self) -> str:
        return self.doc_id or self.source_uri

    @property
    def content_sha256(self) -> str:
        return sha256_of(self.text)


@dataclass(frozen=True, slots=True)
class IngestReport:
    """What one ingest call actually did.

    Returned rather than logged because the skip counts are the interesting
    number: an ingest that reports zero chunks added and N documents skipped is
    idempotence working, and an ingest that reports N documents ingested every
    time it runs is a duplicate-detection bug that would otherwise be invisible
    until the retrieval scores drifted.
    """

    documents_ingested: int = 0
    documents_skipped: int = 0
    chunks_added: int = 0
    chunks_skipped: int = 0
    chunks_by_tier: dict[str, int] = field(default_factory=dict)
    """Chunks added, keyed by :attr:`~aegis.domain.trust.TrustTier.label`. A
    corpus that is 100% T2_CURATED when it was supposed to include scraped pages
    means a source rule matched too broadly - visible here, invisible anywhere
    else until an injection lands."""

    def __repr__(self) -> str:
        return (
            f"IngestReport(docs={self.documents_ingested}+{self.documents_skipped} skipped, "
            f"chunks={self.chunks_added}+{self.chunks_skipped} skipped, "
            f"tiers={self.chunks_by_tier})"
        )


class IngestPipeline:
    """Turns documents into trust-labelled chunks indexed in both retrieval arms."""

    def __init__(
        self,
        *,
        retriever: HybridRetriever | None = None,
        policy: SecurityPolicy | None = None,
        chunker: RecursiveChunker | None = None,
    ) -> None:
        self.retriever = retriever if retriever is not None else HybridRetriever()
        self.policy = policy if policy is not None else SecurityPolicy.load()
        self.chunker = chunker if chunker is not None else RecursiveChunker()
        self._doc_hashes: dict[tuple[str, str], str] = {}
        self._chunk_tiers: dict[str, TrustTier] = {}

    def __len__(self) -> int:
        """Documents currently ingested."""
        return len(self._doc_hashes)

    @property
    def chunk_count(self) -> int:
        return len(self._chunk_tiers)

    def ingest(self, documents: Iterable[Document]) -> IngestReport:
        """Chunk, label and index a batch. Re-ingesting unchanged input is a no-op."""
        pending: list[Chunk] = []
        ingested = 0
        skipped_docs = 0
        skipped_chunks = 0
        by_tier: dict[str, int] = {}

        for document in documents:
            digest = document.content_sha256
            # The key includes the SOURCE, not just the identity, because the tier
            # is a property of where content came from. Keying on identity alone
            # meant the same bytes re-ingested from a different URI were skipped
            # before `resolve_tier` ever ran, so the FIRST source's tier survived a
            # source change - in the one module whose job is establishing trust.
            #
            # Same bytes from a new source is a re-label, and it is cheap: the
            # chunks are content-addressed, so `_new_chunks` recognises them as
            # duplicates and only the tier decision is redone.
            fingerprint = (document.identity, document.source_uri)
            if self._doc_hashes.get(fingerprint) == digest:
                # Same identity, same source, same bytes: nothing to do, and
                # skipping before chunking keeps a re-run cheap as well as correct.
                skipped_docs += 1
                continue

            tier = self.policy.resolve_tier(document.source_uri)
            fresh, duplicates = self._new_chunks(document, tier)
            skipped_chunks += duplicates

            self._doc_hashes[fingerprint] = digest
            ingested += 1
            if fresh:
                pending.extend(fresh)
                by_tier[tier.label] = by_tier.get(tier.label, 0) + len(fresh)

        # One call, both indexes - see the module docstring on why labelling and
        # indexing are not allowed to drift apart.
        self.retriever.add(pending)

        return IngestReport(
            documents_ingested=ingested,
            documents_skipped=skipped_docs,
            chunks_added=len(pending),
            chunks_skipped=skipped_chunks,
            chunks_by_tier=by_tier,
        )

    def ingest_text(
        self,
        text: str,
        *,
        source_uri: str,
        doc_id: str = "",
        metadata: Mapping[str, str] | None = None,
    ) -> IngestReport:
        """Single-document convenience wrapper around :meth:`ingest`."""
        return self.ingest(
            [
                Document(
                    text=text,
                    source_uri=source_uri,
                    doc_id=doc_id,
                    metadata=dict(metadata or {}),
                )
            ]
        )

    def ingest_directory(
        self,
        root: Path | str,
        *,
        suffixes: Sequence[str] = _TEXT_SUFFIXES,
        uri_prefix: str = DEFAULT_CORPUS_URI_PREFIX,
    ) -> IngestReport:
        """Load a directory of text documents and ingest them."""
        return self.ingest(load_directory(root, suffixes=suffixes, uri_prefix=uri_prefix))

    # -- internals -------------------------------------------------------

    def _new_chunks(self, document: Document, tier: TrustTier) -> tuple[list[Chunk], int]:
        """Split ``document`` and drop chunks already in the corpus."""
        chunks = chunk_document(
            document.text,
            doc_id=document.identity,
            source_uri=document.source_uri,
            tier=tier,
            chunker=self.chunker,
            metadata=document.metadata,
        )
        fresh: list[Chunk] = []
        duplicates = 0
        for chunk in chunks:
            known = self._chunk_tiers.get(chunk.chunk_id)
            if known is not None:
                if known is not tier:
                    # The same chunk id arriving at a DIFFERENT tier. Both indexes
                    # are append-only, so there is no honest way to satisfy this:
                    # keeping the old tier leaves content labelled by a source it no
                    # longer comes from, and appending the new one puts two trust
                    # answers for one chunk id into a corpus the gate will later read.
                    #
                    # So it fails closed and says what to do. Silently keeping the
                    # first answer is the failure mode this raise exists to replace -
                    # a stale tier is invisible, and the capability gate acts on it.
                    raise TierConflictError(
                        f"chunk {chunk.chunk_id!r} was already ingested at "
                        f"{known.label} and this source would label it {tier.label} "
                        f"({document.source_uri}). The indexes are append-only, so a "
                        "re-label needs a fresh IngestPipeline rather than an update."
                    )
                duplicates += 1
                continue
            self._chunk_tiers[chunk.chunk_id] = tier
            fresh.append(chunk)
        return fresh, duplicates

    def __repr__(self) -> str:
        return f"IngestPipeline(docs={len(self)}, chunks={self.chunk_count})"


def load_directory(
    root: Path | str,
    *,
    suffixes: Sequence[str] = _TEXT_SUFFIXES,
    uri_prefix: str = DEFAULT_CORPUS_URI_PREFIX,
) -> list[Document]:
    """Read text documents under ``root`` into :class:`Document` objects.

    Walks recursively, sorted, so two ingests of the same tree produce the same
    ordinals and therefore the same chunk ids - directory iteration order is
    filesystem-dependent, and an ingest whose chunk ids change between runs is
    an ingest whose idempotence check silently stops working.

    Source URIs are ``uri_prefix`` plus the path relative to ``root``, which is
    what the policy matches on. Files are decoded as UTF-8 with replacement: one
    bad byte in one file should not abort a corpus load, and the replacement
    character is visible in the chunk rather than silently dropped.
    """
    base = Path(root)
    if not base.is_dir():
        raise NotADirectoryError(f"corpus root is not a directory: {base}")

    wanted = {s.lower() for s in suffixes}
    documents: list[Document] = []
    for path in sorted(p for p in base.rglob("*") if p.is_file()):
        if path.suffix.lower() not in wanted:
            continue
        relative = path.relative_to(base).as_posix()
        documents.append(
            Document(
                text=path.read_text(encoding="utf-8", errors="replace"),
                source_uri=f"{uri_prefix}{relative}",
                doc_id=relative,
                metadata={"path": relative},
            )
        )
    return documents
