"""Real persistent-store regressions for generation publication and retrieval."""

import json

import chromadb
import pytest
from langchain_core.embeddings import Embeddings

from finsight.config import Settings
from finsight.rag import ingest
from finsight.rag.index import IndexIntegrityError, chunks_path, read_manifest
from finsight.rag.retrieve import HybridRetriever


class FixedEmbeddings(Embeddings):
    def __init__(self, dimensions: int = 3):
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] * self.dimensions for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0] * self.dimensions


@pytest.fixture
def indexed(tmp_path, monkeypatch):
    settings = Settings(
        corpus_dir=tmp_path / "corpus",
        index_dir=tmp_path / "index",
        chroma_dir=tmp_path / "chroma",
    )
    settings.corpus_dir.mkdir()
    (settings.corpus_dir / "a.md").write_text("# Report\n\nOriginal financial text")
    embeddings = FixedEmbeddings()
    monkeypatch.setattr(ingest, "build_embeddings", lambda _: embeddings)
    ingest.build_index(settings)
    return settings, embeddings


def test_failed_publication_preserves_old_vectors_and_registry(indexed, monkeypatch):
    settings, embeddings = indexed
    before = read_manifest(settings.index_dir)
    (settings.corpus_dir / "a.md").write_text("# Report\n\nReplacement financial text")

    def interrupt(*args, **kwargs):
        raise RuntimeError("interrupted before publication")

    monkeypatch.setattr(ingest, "write_manifest", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        ingest.build_index(settings)
    assert read_manifest(settings.index_dir) == before
    retriever = HybridRetriever(settings, embeddings=embeddings)
    assert "Original" in retriever.retrieve("financial")[0].chunk.text


def test_rebuild_changes_dimensions_without_invalidating_old_readers(indexed, monkeypatch):
    settings, embeddings = indexed
    old = HybridRetriever(settings, embeddings=embeddings)
    new_embeddings = FixedEmbeddings(4)
    monkeypatch.setattr(ingest, "build_embeddings", lambda _: new_embeddings)
    (settings.corpus_dir / "a.md").write_text("# Report\n\nReplacement financial text")
    ingest.build_index(settings)
    new = HybridRetriever(settings, embeddings=new_embeddings)
    assert new.generation != old.generation
    assert "Replacement" in new.retrieve("financial")[0].chunk.text
    assert "Original" in old.retrieve("financial")[0].chunk.text


def test_registry_text_mutation_with_same_ids_is_rejected(indexed):
    settings, embeddings = indexed
    manifest = read_manifest(settings.index_dir)
    registry = chunks_path(settings.index_dir, manifest.generation)
    data = json.loads(registry.read_text())
    data[0]["text"] = "different content, same ID"
    registry.write_text(json.dumps(data))
    with pytest.raises(IndexIntegrityError, match="content hash"):
        HybridRetriever(settings, embeddings=embeddings)


def test_vector_document_mutation_with_same_ids_is_rejected(indexed):
    settings, embeddings = indexed
    manifest = read_manifest(settings.index_dir)
    collection = chromadb.PersistentClient(path=str(settings.chroma_dir)).get_collection(
        manifest.vector_collection
    )
    collection.update(ids=["a:0"], documents=["changed"], embeddings=[[1.0] * 3])
    with pytest.raises(IndexIntegrityError, match="documents disagree"):
        HybridRetriever(settings, embeddings=embeddings)


def test_mcp_reloads_published_generation(indexed, monkeypatch):
    from finsight.mcp import server

    settings, embeddings = indexed
    monkeypatch.setattr(server, "_retriever", None)
    monkeypatch.setattr(server, "get_settings", lambda: settings)
    monkeypatch.setattr(
        server, "HybridRetriever", lambda s: HybridRetriever(s, embeddings=embeddings)
    )
    first = server._get_retriever()
    (settings.corpus_dir / "a.md").write_text("# Report\n\nReplacement financial text")
    ingest.build_index(settings)
    second = server._get_retriever()
    assert second is not first
    assert "Replacement" in second.retrieve("financial")[0].chunk.text
    assert server._get_retriever() is second


def test_no_lexical_match_does_not_contribute_to_fusion(indexed):
    settings, embeddings = indexed
    retriever = HybridRetriever(settings, embeddings=embeddings)
    assert retriever._bm25_search("absenttoken", 4) == []
    assert retriever._bm25_search("financial", 4)[0][0] == "a:0"
    assert "bm25" not in retriever.retrieve("absenttoken")[0].component_scores


def test_oversized_paragraph_and_overlap_stay_bounded():
    text = "x" * 2000 + "\n\n" + "y" * 90
    chunks = ingest._split_text(text, chunk_size=100, overlap=20)
    assert all(0 < len(chunk) <= 100 for chunk in chunks)
    assert chunks[0] + "".join(chunk[20:] for chunk in chunks[1:]) == text


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, 10), (10, -1)])
def test_invalid_chunking_settings_fail_fast(size, overlap):
    with pytest.raises(ValueError, match="chunk_size"):
        ingest._split_text("text", size, overlap)


@pytest.mark.parametrize("text,query", [("!!! ??? ...", "!!!")])
def test_empty_sparse_vocabulary_uses_dense_retrieval(indexed, text, query):
    settings, embeddings = indexed
    (settings.corpus_dir / "a.md").write_text(text, encoding="utf-8")
    ingest.build_index(settings)

    retriever = HybridRetriever(settings, embeddings=embeddings)
    results = retriever.retrieve(query)
    assert [result.chunk.text for result in results] == [text]
    assert results[0].component_scores == {"vector": 1.0}
    # Even a tokenizable query cannot find sparse evidence in this corpus.
    assert retriever._bm25_search("financial", 4) == []


def test_mixed_sparse_vocabulary_preserves_matching_chunks(indexed):
    settings, embeddings = indexed
    (settings.corpus_dir / "a.md").write_text("公司利润增长", encoding="utf-8")
    (settings.corpus_dir / "b.md").write_text("financial earnings", encoding="utf-8")
    ingest.build_index(settings)

    retriever = HybridRetriever(settings, embeddings=embeddings)
    results = retriever.retrieve("financial")
    assert [result.chunk.doc_id for result in results] == ["b", "a"]
    assert "bm25" in results[0].component_scores
    assert results[1].component_scores == {"vector": 1.0}
    assert retriever._bm25_search("absenttoken", 4) == []
    assert retriever._bm25_search("利润", 4)[0][0] == "a:0"


@pytest.mark.parametrize(
    "payload",
    ["{broken", "null", "{}", "[null]", '[{"chunk_id":"a:0"}]'],
)
def test_malformed_registry_reports_rebuild_guidance(indexed, payload):
    settings, embeddings = indexed
    manifest = read_manifest(settings.index_dir)
    chunks_path(settings.index_dir, manifest.generation).write_text(payload)
    with pytest.raises(IndexIntegrityError, match="malformed.*finsight ingest"):
        HybridRetriever(settings, embeddings=embeddings)


@pytest.mark.parametrize("field,value", [("text", None), ("position", True), ("position", -1)])
def test_invalid_chunk_schema_reports_rebuild_guidance(indexed, field, value):
    settings, embeddings = indexed
    manifest = read_manifest(settings.index_dir)
    registry = chunks_path(settings.index_dir, manifest.generation)
    raw = json.loads(registry.read_text())
    raw[0][field] = value
    registry.write_text(json.dumps(raw))
    with pytest.raises(IndexIntegrityError, match="malformed.*finsight ingest"):
        HybridRetriever(settings, embeddings=embeddings)


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5, "2"])
def test_direct_retrieval_rejects_invalid_top_k(indexed, top_k):
    settings, embeddings = indexed
    retriever = HybridRetriever(settings, embeddings=embeddings)
    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve("financial", top_k=top_k)


@pytest.mark.parametrize("query", ["   ", "x" * 2001])
def test_direct_retrieval_validates_query(indexed, query):
    settings, embeddings = indexed
    retriever = HybridRetriever(settings, embeddings=embeddings)
    with pytest.raises(ValueError, match="Query"):
        retriever.retrieve(query)


def test_top_k_expands_candidate_pool_and_caps_to_corpus(indexed):
    settings, embeddings = indexed
    for number in range(12):
        (settings.corpus_dir / f"doc{number}.md").write_text(f"financial document {number}")
    ingest.build_index(settings)
    retriever = HybridRetriever(settings, embeddings=embeddings)
    assert len(retriever.retrieve("unmatched", top_k=10)) == 10
    assert len(retriever.retrieve("unmatched", top_k=100)) == 13


def _small_batch_client(monkeypatch, settings, *, fail_second=False, max_batch_size=2):
    """Exercise actual Chroma writes behind an artificially small client limit."""
    from types import SimpleNamespace

    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    additions = []

    def create_collection(**kwargs):
        collection = client.create_collection(**kwargs)

        def add(**kwargs):
            additions.append(len(kwargs["ids"]))
            assert len(kwargs["ids"]) <= 2
            if fail_second and len(additions) == 2:
                raise RuntimeError("second batch failed")
            collection.add(**kwargs)

        return SimpleNamespace(add=add, get=collection.get)

    monkeypatch.setattr(
        ingest.chromadb,
        "PersistentClient",
        lambda **kwargs: SimpleNamespace(
            get_max_batch_size=lambda: max_batch_size, create_collection=create_collection
        ),
    )
    return client, additions


@pytest.mark.parametrize("chroma_limit,embedding_limit", [(2, 128), (1000, 2)])
def test_ingestion_batches_vectors_and_embeddings(
    indexed, monkeypatch, chroma_limit, embedding_limit
):
    settings, embeddings = indexed
    for number in range(4):
        (settings.corpus_dir / f"doc{number}.md").write_text(f"financial document {number}")
    client, additions = _small_batch_client(monkeypatch, settings, max_batch_size=chroma_limit)
    monkeypatch.setattr(ingest, "_EMBED_BATCH_SIZE", embedding_limit)
    requests = []
    original_embed = embeddings.embed_documents

    def embed(texts):
        requests.append(len(texts))
        return original_embed(texts)

    monkeypatch.setattr(embeddings, "embed_documents", embed)
    assert ingest.build_index(settings) == 5
    assert requests == additions == [2, 2, 1]
    manifest = read_manifest(settings.index_dir)
    retriever = HybridRetriever(
        settings,
        embeddings=embeddings,
        collection=client.get_collection(manifest.vector_collection),
    )
    assert len(retriever.retrieve("financial", top_k=5)) == 5


def test_failed_later_batch_keeps_published_generation(indexed, monkeypatch):
    settings, embeddings = indexed
    before = read_manifest(settings.index_dir)
    for number in range(4):
        (settings.corpus_dir / f"doc{number}.md").write_text(f"financial document {number}")
    client, additions = _small_batch_client(monkeypatch, settings, fail_second=True)
    with pytest.raises(RuntimeError, match="second batch"):
        ingest.build_index(settings)
    assert additions == [2, 2]
    assert read_manifest(settings.index_dir) == before
    retriever = HybridRetriever(
        settings,
        embeddings=embeddings,
        collection=client.get_collection(before.vector_collection),
    )
    assert "Original" in retriever.retrieve("financial")[0].chunk.text


def test_embedding_endpoint_change_requires_rebuild_before_query(indexed):
    settings, embeddings = indexed
    settings = settings.model_copy(
        update={
            "embed_provider": "openai_compatible",
            "embed_base_url": "http://original.example/v1",
        }
    )
    ingest.build_index(settings)
    changed = settings.model_copy(update={"embed_base_url": "http://other.example/v1"})
    with pytest.raises(IndexIntegrityError, match="embedding endpoint changed"):
        HybridRetriever(changed, embeddings=embeddings)


def test_embedding_endpoint_trailing_slash_is_equivalent(indexed):
    settings, embeddings = indexed
    changed = settings.model_copy(
        update={"embed_base_url": settings.resolved_embed_base_url().rstrip("/") + "/"}
    )
    retriever = HybridRetriever(changed, embeddings=embeddings)
    assert "Original" in retriever.retrieve("financial")[0].chunk.text


def test_manifest_does_not_store_endpoint_credentials(indexed):
    settings, embeddings = indexed
    endpoint = "https://alice:secret-password@embed.example/v1/?key=secret-key"
    changed = settings.model_copy(update={"embed_base_url": endpoint})
    ingest.build_index(changed)
    serialized = (settings.index_dir / "manifest.json").read_text()
    assert "alice" not in serialized
    assert "secret-password" not in serialized
    assert "secret-key" not in serialized
    assert "embed.example" not in serialized
    assert len(read_manifest(settings.index_dir).embed_endpoint_hash) == 64
    retriever = HybridRetriever(changed, embeddings=embeddings)
    assert retriever.retrieve("financial")
