"""Tests for index manifest integrity.

The index used to be three artefacts with nothing tying them together, so a
partial re-index meant a bare ``KeyError`` at query time — or results computed
from vectors produced by a different embedding model. These tests pin the
validation that now catches both.
"""

import json
from pathlib import Path

import pytest

from finsight.config import Settings
from finsight.rag.index import (
    MANIFEST_VERSION,
    IndexIntegrityError,
    IndexMissingError,
    read_manifest,
    validate_manifest,
    write_manifest,
)
from finsight.rag.models import Chunk


def _chunks(*ids: str) -> list[Chunk]:
    return [
        Chunk(chunk_id=cid, doc_id=cid.split(":")[0], title="t", text="body", position=0)
        for cid in ids
    ]


def _write(tmp_path: Path, chunks: list[Chunk], **overrides: str) -> None:
    params = {
        "collection": "finsight",
        "embed_provider": "ollama",
        "embed_model": "nomic-embed-text",
    }
    params.update(overrides)
    write_manifest(tmp_path, chunks, **params)  # type: ignore[arg-type]


def test_manifest_round_trips(tmp_path: Path):
    chunks = _chunks("a:0", "b:0")
    _write(tmp_path, chunks)
    manifest = read_manifest(tmp_path)
    assert manifest.version == MANIFEST_VERSION
    assert manifest.chunk_count == 2
    assert manifest.chunk_ids == ("a:0", "b:0")
    assert manifest.embed_model == "nomic-embed-text"


def test_missing_manifest_raises_with_a_usable_message(tmp_path: Path):
    with pytest.raises(IndexMissingError, match="finsight ingest"):
        read_manifest(tmp_path)


def test_malformed_manifest_raises_integrity_error(tmp_path: Path):
    (tmp_path / "manifest.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(IndexIntegrityError, match="not valid JSON"):
        read_manifest(tmp_path)


def test_manifest_missing_required_keys_is_rejected(tmp_path: Path):
    (tmp_path / "manifest.json").write_text(json.dumps({"version": 1}), encoding="utf-8")
    with pytest.raises(IndexIntegrityError, match="malformed"):
        read_manifest(tmp_path)


def test_valid_manifest_passes_validation(tmp_path: Path):
    chunks = _chunks("a:0", "b:0")
    _write(tmp_path, chunks)
    # Should not raise.
    validate_manifest(
        read_manifest(tmp_path),
        chunks,
        collection="finsight",
        embed_provider="ollama",
        embed_model="nomic-embed-text",
    )


def test_chunk_count_mismatch_is_caught(tmp_path: Path):
    _write(tmp_path, _chunks("a:0", "b:0"))
    with pytest.raises(IndexIntegrityError, match="records 2 chunks, registry has 1"):
        validate_manifest(
            read_manifest(tmp_path),
            _chunks("a:0"),
            collection="finsight",
            embed_provider="ollama",
            embed_model="nomic-embed-text",
        )


def test_chunk_id_drift_is_caught(tmp_path: Path):
    """Two registries with matching counts but different ids must still fail."""
    _write(tmp_path, _chunks("a:0", "b:0"))
    with pytest.raises(IndexIntegrityError, match="chunk ids do not match"):
        validate_manifest(
            read_manifest(tmp_path),
            _chunks("a:0", "c:0"),
            collection="finsight",
            embed_provider="ollama",
            embed_model="nomic-embed-text",
        )


def test_embedding_model_change_is_caught(tmp_path: Path):
    """The highest-value check: stored vectors from another model are meaningless."""
    chunks = _chunks("a:0")
    _write(tmp_path, chunks, embed_model="nomic-embed-text")
    with pytest.raises(IndexIntegrityError, match="not comparable"):
        validate_manifest(
            read_manifest(tmp_path),
            chunks,
            collection="finsight",
            embed_provider="ollama",
            embed_model="text-embedding-3-small",
        )


def test_collection_name_change_is_caught(tmp_path: Path):
    chunks = _chunks("a:0")
    _write(tmp_path, chunks, collection="finsight")
    with pytest.raises(IndexIntegrityError, match="collection"):
        validate_manifest(
            read_manifest(tmp_path),
            chunks,
            collection="something-else",
            embed_provider="ollama",
            embed_model="nomic-embed-text",
        )


def test_settings_drive_the_collection_name():
    """The collection name is configuration, not a literal duplicated in two files."""
    assert Settings(collection_name="custom").collection_name == "custom"
    assert Settings().collection_name == "finsight"


@pytest.mark.parametrize("version", [1, 2])
def test_manifest_without_endpoint_identity_requires_rebuild(tmp_path, version):
    _write(tmp_path, _chunks("a:0"))
    path = tmp_path / "manifest.json"
    data = json.loads(path.read_text())
    data["version"] = version
    data.pop("embed_endpoint_hash")
    path.write_text(json.dumps(data))
    with pytest.raises(IndexIntegrityError, match="Legacy index.*finsight ingest"):
        read_manifest(tmp_path)
