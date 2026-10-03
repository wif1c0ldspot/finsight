"""Local POSIX index leases and explicit, conservative generation cleanup.

flock coordinates Linux/macOS processes sharing one local filesystem. It does not
provide distributed leases or network-filesystem guarantees. Cleanup is dry-run
by default and only recognizes generations created with this module's marker.
"""

from __future__ import annotations

import fcntl
import json
import math
import shutil
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import chromadb
from chromadb.errors import NotFoundError

if TYPE_CHECKING:
    from finsight.config import Settings
    from finsight.rag.index import IndexManifest

_MARKER = ".finsight-generation.json"


@contextmanager
def lifecycle_lock(index_dir: Path) -> Iterator[None]:
    """Serialize publication and lease acquisition with cleanup selection."""
    index_dir.mkdir(parents=True, exist_ok=True)
    with (index_dir / ".lifecycle.lock").open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class GenerationLease:
    """A held shared lock, released explicitly or by a reader's finalizer."""

    def __init__(self, path: Path, *, exclusive: bool = False):
        self._stream = path.open("a+b")
        try:
            mode = fcntl.LOCK_EX | fcntl.LOCK_NB if exclusive else fcntl.LOCK_SH
            fcntl.flock(self._stream.fileno(), mode)
        except BaseException:
            self._stream.close()
            raise

    def close(self) -> None:
        if not self._stream.closed:
            fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
            self._stream.close()


def lease_current_generation(index_dir: Path) -> tuple[IndexManifest, GenerationLease]:
    from finsight.rag.index import chunks_path, read_manifest

    with lifecycle_lock(index_dir):
        manifest = read_manifest(index_dir)
        directory = chunks_path(index_dir, manifest.generation).parent
        # Missing registries should retain the established actionable diagnostic.
        if not directory.is_dir():
            from finsight.rag.index import IndexMissingError

            raise IndexMissingError("Index generation is missing. Re-run: finsight ingest")
        return manifest, GenerationLease(directory / ".lease")


def lease_new_generation(settings: Settings, generation: str) -> GenerationLease:
    from finsight.rag.index import chunks_path

    with lifecycle_lock(settings.index_dir):
        directory = chunks_path(settings.index_dir, generation).parent
        directory.mkdir(parents=True, exist_ok=False)
        lease = GenerationLease(directory / ".lease")
        try:
            marker = {
                "owner": "finsight-local-generation-v1",
                "generation": generation,
                "collection": f"generation-{generation}",
                "chroma_dir": str(settings.chroma_dir.resolve()),
                "created_at": time.time(),
            }
            (directory / _MARKER).write_text(json.dumps(marker), encoding="utf-8")
        except BaseException:
            lease.close()
            raise
        return lease


@dataclass(frozen=True)
class GenerationCleanup:
    generation: str
    action: str


def _owned_marker(directory: Path, settings: Settings) -> dict[str, Any] | None:
    if directory.is_symlink() or not directory.is_dir():
        return None
    try:
        raw = json.loads((directory / _MARKER).read_text(encoding="utf-8"))
        expected = directory.name
        if (
            not isinstance(raw, dict)
            or len(expected) != 32
            or any(c not in "0123456789abcdef" for c in expected)
            or raw.get("owner") != "finsight-local-generation-v1"
            or raw.get("generation") != expected
            or raw.get("collection") != f"generation-{expected}"
            or raw.get("chroma_dir") != str(settings.chroma_dir.resolve())
            or type(raw.get("created_at")) not in (float, int)
            or not math.isfinite(raw["created_at"])
        ):
            return None
        return raw
    except (OSError, ValueError):
        return None


def cleanup_generations(
    settings: Settings, *, keep: int = 2, min_age_seconds: float = 86400, dry_run: bool = True
) -> list[GenerationCleanup]:
    """Inspect/remove managed inactive generations, retaining newest and young ones.

    The published generation is always retained in addition to lease protection.
    ``keep`` counts the newest owned generations, including the published one.
    Unmarked directories and collections are never deleted. Failed builds become
    eligible only after their builder lease is released.
    """
    if type(keep) is not int or keep < 0:
        raise ValueError("keep must be a nonnegative integer")
    if (
        type(min_age_seconds) not in (float, int)
        or not math.isfinite(min_age_seconds)
        or min_age_seconds < 0
    ):
        raise ValueError("min_age_seconds must be finite and nonnegative")
    from finsight.rag.index import IndexMissingError, read_manifest

    results: list[GenerationCleanup] = []
    with lifecycle_lock(settings.index_dir):
        try:
            current = read_manifest(settings.index_dir).generation
        except IndexMissingError:
            current = None
        # A malformed publication pointer aborts cleanup rather than guessing.
        directories = settings.index_dir / "generations"
        owned = []
        for directory in sorted(directories.iterdir()) if directories.exists() else []:
            marker = _owned_marker(directory, settings)
            if marker is None:
                results.append(GenerationCleanup(directory.name, "unmanaged"))
            else:
                owned.append((directory, marker))
        owned.sort(key=lambda item: item[1]["created_at"], reverse=True)
        now = time.time()
        client = None
        for rank, (directory, marker) in enumerate(owned):
            name = directory.name
            if name == current:
                action = "published"
            elif rank < keep:
                action = "retained"
            elif now - marker["created_at"] < min_age_seconds:
                action = "young"
            else:
                try:
                    lease = GenerationLease(directory / ".lease", exclusive=True)
                except BlockingIOError:
                    results.append(GenerationCleanup(name, "leased"))
                    continue
                try:
                    if dry_run:
                        action = "would_delete"
                    else:
                        if client is None:
                            client = chromadb.PersistentClient(path=str(settings.chroma_dir))
                        try:
                            client.delete_collection(marker["collection"])
                        except NotFoundError:
                            pass  # A failed build may never have created its collection.
                        shutil.rmtree(directory)
                        action = "deleted"
                finally:
                    lease.close()
            results.append(GenerationCleanup(name, action))
    return results
