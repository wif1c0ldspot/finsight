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


@pytest.mark.parametrize("text,query", [("公司利润增长", "利润"), ("!!! ??? ...", "!!!")])
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
    assert retriever._bm25_search("利润", 4) == []
