"""Source provenance, metadata filtering, and Unicode lexical regressions."""

import json

import pytest
from langchain_core.embeddings import Embeddings
from pydantic import ValidationError

from finsight.config import Settings
from finsight.mcp import tools
from finsight.rag import ingest
from finsight.rag.format import render_context
from finsight.rag.models import RetrievalFilter, SourceMetadata
from finsight.rag.retrieve import HybridRetriever


class DirectedEmbeddings(Embeddings):
    def __init__(self):
        self.queries = []

    def embed_documents(self, texts):
        return [[0.0, 1.0] if "New" in text else [1.0, 0.0] for text in texts]

    def embed_query(self, text):
        self.queries.append(text)
        return [1.0, 0.0]


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    settings = Settings(
        corpus_dir=tmp_path / "corpus",
        index_dir=tmp_path / "index",
        chroma_dir=tmp_path / "chroma",
        retrieval_candidates=1,
    )
    settings.corpus_dir.mkdir()
    for doc_id, text, published in [
        ("old", "# Old\n\nrevenue revenue", "2020-01-01"),
        ("new", "# New\n\nrevenue 公司利润增长", "2025-06-01"),
        ("undated", "# Undated\n\nrevenue", None),
    ]:
        (settings.corpus_dir / f"{doc_id}.md").write_text(text)
        if published is not None:
            (settings.corpus_dir / f"{doc_id}.metadata.json").write_text(
                json.dumps(
                    {
                        "source_url": f"https://example.com/{doc_id}",
                        "published_at": published,
                        "retrieved_at": "2026-10-03T12:00:00+08:00",
                    }
                )
            )
    embeddings = DirectedEmbeddings()
    monkeypatch.setattr(ingest, "build_embeddings", lambda _: embeddings)
    ingest.build_index(settings)
    return settings, HybridRetriever(settings, embeddings=embeddings), embeddings


def test_metadata_round_trips_registry_tools_and_context(corpus):
    settings, retriever, _ = corpus
    result = retriever.retrieve("revenue", filters=RetrievalFilter(doc_ids=["new"]))[0]
    chunk = result.chunk
    assert chunk.source_url == "https://example.com/new"
    assert chunk.published_at == "2025-06-01"
    assert chunk.retrieved_at == "2026-10-03T12:00:00+08:00"
    assert chunk.revision.startswith("sha256:")
    assert tools.from_payload(tools.to_payload(result)) == result
    record = tools.get_document_record(settings.corpus_dir, "new")
    assert record["published_at"] == chunk.published_at
    assert record["text"] == tools.get_document(settings.corpus_dir, "new")
    listed = next(d for d in tools.list_documents(settings.corpus_dir) if d["doc_id"] == "new")
    assert listed["revision"] == chunk.revision
    context = render_context([result])
    assert "source: https://example.com/new" in context.text
    assert "published: 2025-06-01" in context.text
    assert chunk.revision in context.text
    for budget in [0, 10, len(context.text) - 1, len(context.text)]:
        bounded = render_context([result], budget)
        assert len(bounded.text) <= budget
        assert bool(bounded.citations) == (budget >= len(context.text))


@pytest.mark.parametrize(
    "filters",
    [
        RetrievalFilter(doc_ids=["new"]),
        RetrievalFilter(source_urls=["https://example.com/new"]),
        RetrievalFilter(published_after="2025-06-01", published_before="2025-06-01"),
        RetrievalFilter(doc_ids=["old", "new"], published_after="2021-01-01"),
    ],
)
def test_filters_apply_before_dense_and_sparse_candidate_limits(corpus, filters):
    _, retriever, _ = corpus
    result = retriever.retrieve("revenue", top_k=1, filters=filters)
    assert [item.chunk.doc_id for item in result] == ["new"]
    assert set(result[0].component_scores) == {"vector", "bm25"}
    adapted = tools.make_retrieve_tool(retriever, filters)("revenue")
    assert [item.chunk.doc_id for item in adapted] == ["new"]


@pytest.mark.parametrize(
    "filters",
    [
        RetrievalFilter(doc_ids=[]),
        RetrievalFilter(doc_ids=["absent"]),
        RetrievalFilter(source_urls=[]),
        RetrievalFilter(published_after="2030-01-01"),
        RetrievalFilter(doc_ids=["undated"], published_before="2030-01-01"),
    ],
)
def test_unmatched_filters_skip_embedding_calls(corpus, filters):
    _, retriever, embeddings = corpus
    assert retriever.retrieve("revenue", filters=filters) == []
    assert embeddings.queries == []


def test_legacy_markdown_has_no_invented_dates(corpus):
    settings, _, _ = corpus
    record = tools.get_document_record(settings.corpus_dir, "undated")
    assert record["source_url"] is None
    assert record["published_at"] is None
    assert record["retrieved_at"] is None
    assert record["revision"].startswith("sha256:")


@pytest.mark.parametrize(
    "metadata",
    [
        {"published_at": "2025-02-30"},
        {"published_at": "20250601"},
        {"retrieved_at": "yesterday"},
        {"source_url": "relative/path"},
        {"source_url": "https://user:secret@example.com"},
        {"revision": " "},
        {"published_at": 2025},
        {"unknown": "field"},
        [],
        None,
    ],
)
def test_sidecar_schema_is_strict(tmp_path, metadata):
    (tmp_path / "a.md").write_text("# A\n\nbody")
    (tmp_path / "a.metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Invalid source metadata"):
        ingest.load_documents(tmp_path)


def test_revision_digest_changes_with_content_and_explicit_revision_wins(tmp_path):
    document = tmp_path / "a.md"
    document.write_text("first")
    first = ingest.load_documents(tmp_path)[0].revision
    document.write_text("second")
    assert ingest.load_documents(tmp_path)[0].revision != first
    (tmp_path / "a.metadata.json").write_text('{"revision": "release-17"}')
    assert ingest.load_documents(tmp_path)[0].revision == "release-17"


@pytest.mark.parametrize(
    "filters",
    [
        {"published_after": "2025-01-01", "published_before": "2020-01-01"},
        {"doc_ids": [""]},
        {"doc_ids": "new"},
        {"published_after": "not-date"},
        {"unknown": "field"},
    ],
)
def test_filter_schema_is_strict(filters):
    with pytest.raises(ValidationError):
        RetrievalFilter.model_validate(filters)


def test_unicode_tokenizer_preserves_words_and_han_terms(corpus):
    assert ingest.tokenize("CAFÉ Straße ＦＩＮＡＮＣＥ") == ["café", "strasse", "finance"]
    assert ingest.tokenize("公司利润增长") == list("公司利润增长")
    _, retriever, _ = corpus
    assert [cid for cid, _ in retriever._bm25_search("利润", 4)] == ["new:0"]


def test_sidecar_accepts_iso_date_and_datetime():
    assert SourceMetadata(retrieved_at="2026-10-03").retrieved_at == "2026-10-03"
    assert SourceMetadata(retrieved_at="2026-10-03T12:00:00Z").retrieved_at.endswith("Z")


def test_mcp_transport_accepts_typed_filters_and_returns_provenance(corpus, monkeypatch):
    import asyncio

    from fastmcp import Client

    from finsight.mcp import server

    settings, retriever, _ = corpus
    monkeypatch.setattr(server, "_get_retriever", lambda: retriever)
    monkeypatch.setattr(server, "get_settings", lambda: settings)

    async def request():
        async with Client(server.mcp) as client:
            result = await client.call_tool(
                "search_documents", {"query": "revenue", "filters": {"doc_ids": ["new"]}}
            )
            payload = json.loads(result.content[0].text)
            assert payload[0]["doc_id"] == "new"
            assert payload[0]["source_url"] == "https://example.com/new"
            record = await client.call_tool("get_document_record", {"doc_id": "new"})
            assert json.loads(record.content[0].text)["published_at"] == "2025-06-01"

    asyncio.run(request())
