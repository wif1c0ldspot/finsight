"""Atomic publication manifest binding an immutable registry and vector generation.

A rebuild creates fresh artefacts and replaces only the manifest once complete.
Each reader resolves that manifest once, so concurrent ingestion cannot mix
old provenance with new vectors. Earlier manifest versions require a rebuild.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit, urlunsplit

from finsight.rag.models import Chunk

#: Bump when the on-disk layout changes incompatibly.
MANIFEST_VERSION = 4

MANIFEST_NAME = "manifest.json"
CHUNKS_NAME = "chunks.json"


class IndexMissingError(RuntimeError):
    """No usable index on disk."""


class IndexIntegrityError(RuntimeError):
    """The index artefacts disagree with each other or with current settings."""


@dataclass(frozen=True)
class IndexManifest:
    """What was indexed, and with what."""

    version: int
    chunk_count: int
    chunk_ids: tuple[str, ...]
    collection: str
    embed_provider: str
    embed_model: str
    created_at: str
    generation: str
    vector_collection: str
    content_hash: str
    embed_endpoint_hash: str
    embed_revision: str | None

    def to_json(self) -> dict[str, object]:
        return {
            "version": self.version,
            "chunk_count": self.chunk_count,
            "chunk_ids": list(self.chunk_ids),
            "collection": self.collection,
            "embed_provider": self.embed_provider,
            "embed_model": self.embed_model,
            "created_at": self.created_at,
            "generation": self.generation,
            "vector_collection": self.vector_collection,
            "content_hash": self.content_hash,
            "embed_endpoint_hash": self.embed_endpoint_hash,
            "embed_revision": self.embed_revision,
        }

    @classmethod
    def from_json(cls, data: dict[str, object]) -> IndexManifest:
        try:
            return cls(
                version=int(cast(Any, data["version"])),
                chunk_count=int(cast(Any, data["chunk_count"])),
                chunk_ids=tuple(str(x) for x in cast(Any, data["chunk_ids"])),
                collection=str(data["collection"]),
                embed_provider=str(data["embed_provider"]),
                embed_model=str(data["embed_model"]),
                created_at=str(data["created_at"]),
                generation=str(data["generation"]),
                vector_collection=str(data["vector_collection"]),
                content_hash=str(data["content_hash"]),
                embed_endpoint_hash=str(data["embed_endpoint_hash"]),
                embed_revision=cast(str | None, data["embed_revision"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise IndexIntegrityError(
                f"Index manifest is malformed ({exc}). Re-run: finsight ingest"
            ) from exc


def manifest_path(index_dir: Path) -> Path:
    return index_dir / MANIFEST_NAME


def chunks_path(index_dir: Path, generation: str = "") -> Path:
    if generation:
        if not _valid_generation(generation):
            raise IndexIntegrityError("Invalid index generation. Re-run: finsight ingest")
        return index_dir / "generations" / generation / CHUNKS_NAME
    return index_dir / CHUNKS_NAME


def _valid_generation(value: str) -> bool:
    return len(value) == 32 and all(c in "0123456789abcdef" for c in value)


def content_hash(chunks: list[Chunk]) -> str:
    payload = json.dumps([asdict(c) for c in chunks], sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _endpoint_hash(base_url: str | None) -> str:
    """Bind the embedding route without persisting credential-bearing URLs."""
    if base_url is None:
        normalized = ""
    else:
        parts = urlsplit(base_url.strip())
        normalized = urlunsplit(parts._replace(path=parts.path.rstrip("/")))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def write_manifest(
    index_dir: Path,
    chunks: list[Chunk],
    *,
    collection: str,
    embed_provider: str,
    embed_model: str,
    generation: str = "",
    vector_collection: str | None = None,
    embed_base_url: str | None = None,
    embed_revision: str | None = None,
) -> None:
    manifest = IndexManifest(
        version=MANIFEST_VERSION,
        chunk_count=len(chunks),
        chunk_ids=tuple(c.chunk_id for c in chunks),
        collection=collection,
        embed_provider=embed_provider,
        embed_model=embed_model,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        generation=generation,
        vector_collection=vector_collection or collection,
        content_hash=content_hash(chunks),
        embed_endpoint_hash=_endpoint_hash(embed_base_url),
        embed_revision=embed_revision,
    )
    index_dir.mkdir(parents=True, exist_ok=True)
    temporary = index_dir / f".manifest-{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(manifest.to_json(), indent=2))
            stream.flush()
            os.fsync(stream.fileno())
        from finsight.rag.lifecycle import lifecycle_lock

        with lifecycle_lock(index_dir):
            temporary.replace(manifest_path(index_dir))
    finally:
        temporary.unlink(missing_ok=True)


def read_manifest(index_dir: Path) -> IndexManifest:
    path = manifest_path(index_dir)
    if not path.exists():
        raise IndexMissingError(
            f"No index manifest at {path}. Build the index first: finsight ingest"
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise IndexIntegrityError(
            f"Index manifest is not valid JSON ({exc}). Re-run: finsight ingest"
        ) from exc
    if not isinstance(raw, dict):
        raise IndexIntegrityError("Index manifest must be a JSON object.")
    if raw.get("version") in (1, 2, 3) and "chunk_count" in raw:
        raise IndexIntegrityError("Legacy index needs rebuilding. Re-run: finsight ingest")
    return IndexManifest.from_json(raw)


def validate_manifest(
    manifest: IndexManifest,
    chunks: list[Chunk],
    *,
    collection: str,
    embed_provider: str,
    embed_model: str,
    embed_base_url: str | None = None,
    embed_revision: str | None = None,
) -> None:
    """Raise :class:`IndexIntegrityError` if the manifest contradicts reality.

    This is the check that was missing: it catches a truncated write, a chunk
    registry edited by hand, a stale manifest, and — most importantly — an
    embedding-model change that would otherwise make every stored vector
    meaningless while still returning results.
    """
    problems: list[str] = []

    if manifest.version != MANIFEST_VERSION:
        problems.append(f"manifest version {manifest.version} != expected {MANIFEST_VERSION}")
    if manifest.chunk_count != len(chunks):
        problems.append(
            f"manifest records {manifest.chunk_count} chunks, registry has {len(chunks)}"
        )
    if set(manifest.chunk_ids) != {c.chunk_id for c in chunks}:
        problems.append("manifest chunk ids do not match the chunk registry")
    if manifest.content_hash != content_hash(chunks):
        problems.append("manifest content hash does not match the chunk registry")
    if manifest.embed_revision != embed_revision:
        problems.append("embedding revision changed; stored vectors are not comparable")
    if manifest.embed_endpoint_hash != _endpoint_hash(embed_base_url):
        problems.append("embedding endpoint changed; stored vectors are not comparable")
    if manifest.collection != collection:
        problems.append(
            f"index built against collection {manifest.collection!r}, settings say {collection!r}"
        )
    if manifest.embed_model != embed_model or manifest.embed_provider != embed_provider:
        problems.append(
            f"index embedded with {manifest.embed_provider}/{manifest.embed_model}, "
            f"settings say {embed_provider}/{embed_model} — stored vectors are not "
            f"comparable; re-run: finsight ingest"
        )

    if problems:
        raise IndexIntegrityError(
            "Index integrity check failed:\n  - "
            + "\n  - ".join(problems)
            + "\nRebuild the index with: finsight ingest"
        )
