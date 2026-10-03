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

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, cast

import chromadb

from finsight.config import Settings
from finsight.llm import build_embeddings
from finsight.rag.index import chunks_path, write_manifest
from finsight.rag.models import Chunk, Document

_WORD = re.compile(r"[a-z0-9]+")
# Bound provider request sizes as well as the local vector-store transaction.
_EMBED_BATCH_SIZE = 128


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenizer, shared by ingestion and BM25 queries.

    Lives here (not duplicated in the retriever) so the indexed terms and the
    query terms can never diverge.
    """
    return _WORD.findall(text.lower())


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Paragraph-aware chunker: pack paragraphs into ~chunk_size windows.

    Deliberately simple and transparent — no hidden tokenizer assumptions — which
    makes the indexing behaviour easy to reason about and test.
    """
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size")
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
        chunks.append(normalized[start:end])
        if end == len(normalized):
            break
        start = end - overlap
    return chunks


def load_documents(corpus_dir: Path) -> list[Document]:
    """Load every ``*.md`` file under ``corpus_dir`` as a Document."""
    docs: list[Document] = []
    for path in sorted(corpus_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        title = lines[0].lstrip("# ").strip() if lines else path.stem
        docs.append(
            Document(doc_id=path.stem, title=title or path.stem, text=text, source=path.name)
        )
    return docs


def chunk_documents(docs: list[Document], chunk_size: int, overlap: int) -> list[Chunk]:
    """Split documents into overlapping chunks, preserving provenance."""
    chunks: list[Chunk] = []
    for doc in docs:
        for pos, text in enumerate(_split_text(doc.text, chunk_size, overlap)):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}:{pos}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    text=text,
                    position=pos,
                )
            )
    return chunks


def chunk_to_dict(chunk: Chunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "title": chunk.title,
        "text": chunk.text,
        "position": chunk.position,
    }


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
    return Chunk(
        chunk_id=str(data["chunk_id"]),
        doc_id=str(data["doc_id"]),
        title=str(data["title"]),
        text=str(data["text"]),
        position=int(data["position"]),
    )


def build_index(settings: Settings) -> int:
    """Build an immutable generation, then atomically publish it.

    Prior generations remain available to in-flight readers. Failed builds never
    alter the currently published registry or collection. Returns chunk count.
    """
    docs = load_documents(settings.corpus_dir)
    if not docs:
        raise RuntimeError(f"No .md files found in {settings.corpus_dir}")

    chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise RuntimeError("Corpus produced no chunks; check chunk_size configuration.")

    embeddings = build_embeddings(settings)
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    batch_size = min(_EMBED_BATCH_SIZE, client.get_max_batch_size())
    generation = uuid.uuid4().hex
    physical_collection = f"generation-{generation}"
    collection = client.create_collection(
        name=physical_collection, metadata={"hnsw:space": "cosine"}
    )
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors = embeddings.embed_documents([c.text for c in batch])
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
    registry.parent.mkdir(parents=True, exist_ok=True)
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
    )
    return len(chunks)
