"""Corpus ingestion: chunk documents, embed them, and build the hybrid index.

Artefacts (under ``data/``):
  * ``chroma/``             — persistent vector store (ChromaDB, cosine)
  * ``index/generations/<id>/chunks.json`` — immutable registry with provenance
  * ``index/manifest.json`` — atomic pointer to the published index generation

There is deliberately no pickled index. BM25 is rebuilt in memory from the chunk
registry at load time: the registry is small, JSON is inspectable, and
``pickle.load`` on a local file is an arbitrary-code-execution path in a codebase
that is otherwise careful about boundaries.
"""

import hashlib
import json
import os
import re
import unicodedata
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import chromadb

from finsight.config import Settings
from finsight.llm import build_embeddings
from finsight.rag.index import IndexIntegrityError, IndexMissingError, chunks_path, write_manifest
from finsight.rag.lifecycle import lease_new_generation
from finsight.rag.models import Chunk, Document, SourceMetadata

_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002fa1f]")
# Bound provider request sizes as well as the local vector-store transaction.
_EMBED_BATCH_SIZE = 128


def tokenize(text: str) -> list[str]:
    """Unicode words plus Han character tokens for unsegmented CJK text.

    NFKC/casefold normalize compatibility forms and case. This is deliberately a
    dependency-free lexical baseline, not linguistic segmentation or stemming.
    """
    normalized = unicodedata.normalize("NFKC", text).casefold()
    separated = _CJK.sub(lambda match: f" {match.group()} ", normalized)
    return _WORD.findall(separated)


def _split_text(
    text: str,
    chunk_size: int,
    overlap: int,
    *,
    max_tokens: int | None = None,
    token_counter: Callable[[str], int] | None = None,
) -> list[str]:
    """Paragraph-aware chunker: pack paragraphs into ~chunk_size windows.

    Optional token bounds use UTF-8 content bytes unless an exact tokenizer is
    injected. Provider framing is outside this content budget. When that bound
    shrinks a chunk below the requested overlap, overlap is clipped to preserve
    at least one character of forward progress. No source characters are skipped.
    """
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size")
    if max_tokens is not None and (type(max_tokens) is not int or max_tokens <= 0):
        raise ValueError("max_tokens must be a positive integer")
    normalized = "\n\n".join(p.strip() for p in text.split("\n\n") if p.strip())
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        if end < len(normalized):
            # Prefer a paragraph boundary only when it still advances past overlap.
            boundary = normalized.rfind("\n\n", start + overlap + 1, end)
            if boundary != -1:
                end = boundary + 2
        if max_tokens is not None:
            if token_counter is None:
                # Byte slicing cannot split a Unicode character: drop only the
                # incomplete final character and leave it for the next chunk.
                prefix = normalized[start:end].encode("utf-8")[:max_tokens]
                end = start + len(prefix.decode("utf-8", errors="ignore"))
            else:
                # Prefix token counts need not be monotonic (BPE merges can
                # lower the count), so do not binary-search an assumed ordering.
                while end > start:
                    count = token_counter(normalized[start:end])
                    if type(count) is not int or count < 0:
                        raise ValueError("Token counter must return a nonnegative integer")
                    if count <= max_tokens:
                        break
                    end -= 1
            if end == start:
                raise ValueError(
                    "Token budget cannot fit a nonempty chunk at character "
                    f"{start}; increase max_tokens or supply the embedding model's tokenizer"
                )
        chunks.append(normalized[start:end])
        if end == len(normalized):
            break
        start = max(start + 1, end - overlap)
    return chunks


def load_documents(corpus_dir: Path) -> list[Document]:
    """Load every ``*.md`` file under ``corpus_dir`` as a Document."""
    docs: list[Document] = []
    for path in sorted(corpus_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        title = lines[0].lstrip("# ").strip() if lines else path.stem
        metadata_path = path.with_suffix(".metadata.json")
        try:
            metadata = (
                SourceMetadata.model_validate_json(metadata_path.read_text(encoding="utf-8"))
                if metadata_path.exists()
                else SourceMetadata()
            )
        except (ValueError, OSError) as exc:
            raise ValueError(f"Invalid source metadata at {metadata_path}: {exc}") from exc
        provenance = metadata.model_dump()
        provenance["revision"] = (
            metadata.revision or "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
        )
        docs.append(
            Document(
                doc_id=path.stem,
                title=title or path.stem,
                text=text,
                source=path.name,
                **provenance,
            )
        )
    return docs


def chunk_documents(
    docs: list[Document],
    chunk_size: int,
    overlap: int,
    *,
    max_tokens: int | None = None,
    token_counter: Callable[[str], int] | None = None,
) -> list[Chunk]:
    """Split documents into overlapping chunks, preserving provenance."""
    chunks: list[Chunk] = []
    for doc in docs:
        for pos, text in enumerate(
            _split_text(
                doc.text, chunk_size, overlap, max_tokens=max_tokens, token_counter=token_counter
            )
        ):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}:{pos}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    text=text,
                    position=pos,
                    source_url=doc.source_url,
                    published_at=doc.published_at,
                    retrieved_at=doc.retrieved_at,
                    revision=doc.revision,
                )
            )
    return chunks


def chunk_to_dict(chunk: Chunk) -> dict[str, Any]:
    return asdict(chunk)


def chunk_from_dict(data: dict[str, Any]) -> Chunk:
    if not isinstance(data, dict):
        raise ValueError("Each registry chunk must be an object")
    for key in ("chunk_id", "doc_id", "title", "text"):
        if not isinstance(data.get(key), str):
            raise ValueError(f"Chunk {key} must be a string")
    if not data["chunk_id"] or not data["doc_id"]:
        raise ValueError("Chunk identifiers must not be empty")
    if type(data.get("position")) is not int or data["position"] < 0:
        raise ValueError("Chunk position must be a nonnegative integer")
    provenance = SourceMetadata.model_validate(
        {key: data.get(key) for key in ("source_url", "published_at", "retrieved_at", "revision")}
    )
    return Chunk(
        chunk_id=str(data["chunk_id"]),
        doc_id=str(data["doc_id"]),
        title=str(data["title"]),
        text=str(data["text"]),
        position=int(data["position"]),
        **provenance.model_dump(),
    )


@dataclass(frozen=True)
class BuildStats:
    chunk_count: int
    embedded_chunks: int
    reused_chunks: int
    generation: str


def build_index(
    settings: Settings,
    *,
    incremental: bool = True,
    on_stats: Callable[[BuildStats], None] | None = None,
) -> int:
    """Build an immutable generation, then atomically publish it.

    Prior generations remain available to in-flight readers. Failed builds never
    alter the currently published registry or collection. Returns chunk count.
    """
    docs = load_documents(settings.corpus_dir)
    if not docs:
        raise RuntimeError(f"No .md files found in {settings.corpus_dir}")

    chunks = chunk_documents(
        docs, settings.chunk_size, settings.chunk_overlap, max_tokens=settings.chunk_max_tokens
    )
    if not chunks:
        raise RuntimeError("Corpus produced no chunks; check chunk_size configuration.")

    embeddings = build_embeddings(settings)
    cached: dict[str, list[float]] = {}
    if incremental:
        from finsight.rag.retrieve import HybridRetriever

        try:
            with HybridRetriever(settings, embeddings=embeddings) as previous:
                cached = previous.embedding_cache()
        except (IndexMissingError, IndexIntegrityError):
            # An absent, legacy, corrupt, or incompatible generation cannot be a
            # vector cache. A fresh build remains safe and repairs the index.
            pass

    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    batch_size = min(_EMBED_BATCH_SIZE, client.get_max_batch_size())
    generation = uuid.uuid4().hex
    physical_collection = f"generation-{generation}"
    lease = lease_new_generation(settings, generation)
    embedded = reused = 0
    try:
        collection = client.create_collection(
            name=physical_collection, metadata={"hnsw:space": "cosine"}
        )
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            missing = [c for c in batch if c.text not in cached]
            fresh = embeddings.embed_documents([c.text for c in missing]) if missing else []
            if len(fresh) != len(missing):
                raise RuntimeError("Embedding provider returned an incorrect vector count")
            fresh_by_text = {c.text: vector for c, vector in zip(missing, fresh, strict=True)}
            vectors = [cached[c.text] if c.text in cached else fresh_by_text[c.text] for c in batch]
            embedded += len(missing)
            reused += len(batch) - len(missing)
            collection.add(
                ids=[c.chunk_id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[
                    {"doc_id": c.doc_id, "title": c.title, "position": c.position} for c in batch
                ],
                embeddings=cast(Any, vectors),
            )

        stored = collection.get()
        if dict(zip(stored["ids"], stored["documents"] or [], strict=True)) != {
            c.chunk_id: c.text for c in chunks
        }:
            raise RuntimeError("New vector generation failed content validation")

        registry = chunks_path(settings.index_dir, generation)
        with registry.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps([chunk_to_dict(c) for c in chunks], indent=2))
            stream.flush()
            os.fsync(stream.fileno())
        write_manifest(
            settings.index_dir,
            chunks,
            collection=settings.collection_name,
            embed_provider=settings.embed_provider,
            embed_model=settings.embed_model,
            generation=generation,
            vector_collection=physical_collection,
            embed_base_url=settings.resolved_embed_base_url(),
            embed_revision=settings.embed_revision,
        )
    finally:
        lease.close()
    if on_stats is not None:
        on_stats(BuildStats(len(chunks), embedded, reused, generation))
    return len(chunks)
