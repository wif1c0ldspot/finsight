"""Embedding reuse and POSIX generation leases, including separate processes."""

import gc
import json
import multiprocessing

import chromadb
import pytest
from langchain_core.embeddings import Embeddings

from finsight.config import Settings
from finsight.rag import ingest
from finsight.rag.index import read_manifest
from finsight.rag.lifecycle import cleanup_generations
from finsight.rag.retrieve import HybridRetriever


class CountingEmbeddings(Embeddings):
    def __init__(self):
        self.documents = []

    def embed_documents(self, texts):
        self.documents.extend(texts)
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


@pytest.fixture
def indexed(tmp_path, monkeypatch):
    settings = Settings(
        corpus_dir=tmp_path / "corpus", index_dir=tmp_path / "index", chroma_dir=tmp_path / "chroma"
    )
    settings.corpus_dir.mkdir()
    (settings.corpus_dir / "a.md").write_text("first financial document")
    (settings.corpus_dir / "b.md").write_text("second financial document")
    embeddings = CountingEmbeddings()
    monkeypatch.setattr(ingest, "build_embeddings", lambda _: embeddings)
    ingest.build_index(settings)
    embeddings.documents.clear()
    return settings, embeddings


def test_unchanged_changed_and_moved_chunks_reuse_embeddings(indexed):
    settings, embeddings = indexed
    stats = []
    ingest.build_index(settings, on_stats=stats.append)
    assert embeddings.documents == []
    assert (stats[-1].reused_chunks, stats[-1].embedded_chunks) == (2, 0)
    (settings.corpus_dir / "a.md").rename(settings.corpus_dir / "moved.md")
    (settings.corpus_dir / "b.md").write_text("changed financial document")
    ingest.build_index(settings, on_stats=stats.append)
    assert embeddings.documents == ["changed financial document"]
    assert (stats[-1].reused_chunks, stats[-1].embedded_chunks) == (1, 1)
    with HybridRetriever(settings, embeddings=embeddings) as retriever:
        assert {result.chunk.doc_id for result in retriever.retrieve("financial")} == {"moved", "b"}


@pytest.mark.parametrize(
    "update",
    [
        {"embed_model": "another-model"},
        {"embed_revision": "deployment-2"},
        {"embed_base_url": "http://another.local:11434"},
    ],
)
def test_embedding_identity_changes_bypass_cache(indexed, update):
    settings, embeddings = indexed
    changed = settings.model_copy(update=update)
    stats = []
    ingest.build_index(changed, on_stats=stats.append)
    assert len(embeddings.documents) == 2
    assert stats[-1].reused_chunks == 0
    assert read_manifest(settings.index_dir).embed_revision == changed.embed_revision


def test_full_rebuild_opt_out_embeds_every_chunk(indexed):
    settings, embeddings = indexed
    ingest.build_index(settings, incremental=False)
    assert len(embeddings.documents) == 2


def test_legacy_manifest_without_revision_is_freshly_rebuilt(indexed):
    settings, embeddings = indexed
    path = settings.index_dir / "manifest.json"
    raw = json.loads(path.read_text())
    raw.pop("embed_revision")
    path.write_text(json.dumps(raw))
    ingest.build_index(settings)
    assert len(embeddings.documents) == 2


def test_failed_embedding_build_is_reclaimable_and_preserves_publication(indexed, monkeypatch):
    settings, embeddings = indexed
    before = read_manifest(settings.index_dir)
    (settings.corpus_dir / "a.md").write_text("changed document")

    def fail(texts):
        raise RuntimeError("provider failed")

    monkeypatch.setattr(embeddings, "embed_documents", fail)
    with pytest.raises(RuntimeError, match="provider failed"):
        ingest.build_index(settings)
    assert read_manifest(settings.index_dir) == before
    preview = cleanup_generations(settings, keep=0, min_age_seconds=0)
    assert {row.action for row in preview} == {"published", "would_delete"}
    applied = cleanup_generations(settings, keep=0, min_age_seconds=0, dry_run=False)
    assert {row.action for row in applied} == {"published", "deleted"}
    with HybridRetriever(settings, embeddings=embeddings) as retriever:
        assert len(retriever.retrieve("financial")) == 2


def test_failed_cache_read_keeps_previous_publication(indexed, monkeypatch):
    settings, _ = indexed
    before = read_manifest(settings.index_dir)

    def fail(self):
        raise RuntimeError("cache storage failed")

    monkeypatch.setattr(HybridRetriever, "embedding_cache", fail)
    with pytest.raises(RuntimeError, match="cache storage failed"):
        ingest.build_index(settings)
    assert read_manifest(settings.index_dir) == before


def test_cleanup_preserves_reader_then_deletes_after_close(indexed):
    settings, embeddings = indexed
    reader = HybridRetriever(settings, embeddings=embeddings)
    old = reader.generation
    ingest.build_index(settings)
    assert (
        dict(
            (r.generation, r.action)
            for r in cleanup_generations(settings, keep=0, min_age_seconds=0)
        )[old]
        == "leased"
    )
    assert reader.retrieve("financial")
    reader.close()
    with pytest.raises(RuntimeError, match="closed"):
        reader.retrieve("financial")
    preview = cleanup_generations(settings, keep=0, min_age_seconds=0)
    assert dict((r.generation, r.action) for r in preview)[old] == "would_delete"
    assert (settings.index_dir / "generations" / old).exists()
    cleanup_generations(settings, keep=0, min_age_seconds=0, dry_run=False)
    assert not (settings.index_dir / "generations" / old).exists()


def test_reader_finalizer_releases_lease(indexed):
    settings, embeddings = indexed
    reader = HybridRetriever(settings, embeddings=embeddings)
    old = reader.generation
    ingest.build_index(settings)
    del reader
    gc.collect()
    actions = {
        r.generation: r.action for r in cleanup_generations(settings, keep=0, min_age_seconds=0)
    }
    assert actions[old] == "would_delete"


def test_cleanup_respects_retention_and_ignores_unmanaged_data(indexed):
    settings, _ = indexed
    ingest.build_index(settings)
    ingest.build_index(settings)
    unknown = settings.index_dir / "generations" / "user-data"
    unknown.mkdir()
    (unknown / "important.txt").write_text("keep")
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    client.create_collection("user-collection")
    rows = cleanup_generations(settings, keep=2, min_age_seconds=0)
    assert sorted(row.action for row in rows) == [
        "published",
        "retained",
        "unmanaged",
        "would_delete",
    ]
    rows = cleanup_generations(settings, keep=0, min_age_seconds=86400, dry_run=False)
    assert sorted(row.action for row in rows) == ["published", "unmanaged", "young", "young"]
    assert (unknown / "important.txt").read_text() == "keep"
    assert client.get_collection("user-collection")


def _hold_reader(settings, ready, release):
    with HybridRetriever(settings, embeddings=CountingEmbeddings()) as reader:
        assert reader.retrieve("financial")
        ready.set()
        if not release.wait(20):
            raise RuntimeError("Reader release timed out")
        assert reader.retrieve("financial")


def _hold_builder(settings, ready, release):
    class BlockingEmbeddings(CountingEmbeddings):
        def embed_documents(self, texts):
            ready.set()
            if not release.wait(20):
                raise RuntimeError("Builder release timed out")
            return super().embed_documents(texts)

    ingest.build_embeddings = lambda _: BlockingEmbeddings()
    ingest.build_index(settings, incremental=False)


@pytest.mark.parametrize("worker", [_hold_reader, _hold_builder])
def test_other_process_lease_protects_active_generation(indexed, worker):
    settings, _ = indexed
    context = multiprocessing.get_context("spawn")
    ready, release = context.Event(), context.Event()
    process = context.Process(target=worker, args=(settings, ready, release))
    process.start()
    try:
        assert ready.wait(15), f"child did not acquire lease, exit={process.exitcode}"
        if worker is _hold_reader:
            ingest.build_index(settings)
        rows = cleanup_generations(settings, keep=0, min_age_seconds=0, dry_run=False)
        assert "leased" in {row.action for row in rows}
        assert "deleted" not in {row.action for row in rows}
        release.set()
        process.join(15)
        assert process.exitcode == 0
    finally:
        release.set()
        if process.is_alive():
            process.terminate()
        process.join(5)


def test_metadata_only_change_reuses_vectors_but_updates_provenance(indexed):
    settings, embeddings = indexed
    (settings.corpus_dir / "a.metadata.json").write_text(
        '{"source_url":"https://example.com/report","revision":"source-revision-2"}'
    )
    ingest.build_index(settings)
    assert embeddings.documents == []
    with HybridRetriever(settings, embeddings=embeddings) as retriever:
        result = next(hit for hit in retriever.retrieve("financial") if hit.chunk.doc_id == "a")
        assert result.chunk.revision == "source-revision-2"
        assert result.chunk.source_url == "https://example.com/report"


@pytest.mark.parametrize("age", [-1, float("nan"), float("inf")])
def test_cleanup_rejects_unsafe_age_bounds(indexed, age):
    settings, _ = indexed
    with pytest.raises(ValueError, match="finite and nonnegative"):
        cleanup_generations(settings, min_age_seconds=age, dry_run=False)
