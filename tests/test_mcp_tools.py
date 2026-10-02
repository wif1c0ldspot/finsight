"""Tests for the MCP tool layer.

The tools are now the single implementation of the retrieval surface: the
standalone MCP server registers these functions and the agent graph calls the same
ones. These tests pin the payload contract that both rely on.
"""

from pathlib import Path
from typing import Any, cast

import pytest

from finsight.mcp import tools
from finsight.rag.models import Chunk, RetrievedChunk
from finsight.rag.retrieve import HybridRetriever


class FakeRetriever:
    def __init__(self, chunk_ids: tuple[str, ...] = ("a:0", "b:1")) -> None:
        self.queries: list[tuple[str, int | None]] = []
        self._chunk_ids = chunk_ids

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        self.queries.append((query, top_k))
        return [
            RetrievedChunk(
                chunk=Chunk(
                    chunk_id=cid,
                    doc_id=cid.split(":")[0],
                    title=f"Title {cid}",
                    text=f"body {cid}",
                    position=int(cid.split(":")[1]),
                ),
                score=0.5,
                rank=index,
                method="rrf",
                component_scores={"vector": 0.5, "bm25": 1.25},
            )
            for index, cid in enumerate(self._chunk_ids, start=1)
        ]


def test_search_documents_returns_json_safe_payloads():
    retriever = FakeRetriever()
    payloads = tools.search_documents(cast(HybridRetriever, retriever), "founding year", top_k=2)

    assert retriever.queries == [("founding year", 2)]
    assert len(payloads) == 2
    first = payloads[0]
    assert first["chunk_id"] == "a:0"
    assert first["doc_id"] == "a"
    assert first["text"] == "body a:0"
    assert first["score"] == 0.5
    assert first["rank"] == 1
    assert first["method"] == "rrf"
    assert first["component_scores"] == {"vector": 0.5, "bm25": 1.25}


def test_payload_round_trip_preserves_the_result():
    """The graph rehydrates tool payloads, so nothing may be lost in transit."""
    retriever = FakeRetriever()
    original = retriever.retrieve("q")
    for result in original:
        restored = tools.from_payload(tools.to_payload(result))
        assert restored.chunk == result.chunk
        assert restored.score == result.score
        assert restored.rank == result.rank
        assert restored.component_scores == result.component_scores


def test_make_retrieve_tool_returns_typed_results():
    retriever = FakeRetriever()
    retrieve = tools.make_retrieve_tool(cast(HybridRetriever, retriever))
    results = retrieve("founding year")

    assert isinstance(results[0], RetrievedChunk)
    assert results[0].chunk.chunk_id == "a:0"


def test_from_payload_tolerates_missing_optional_fields():
    minimal: dict[str, Any] = {
        "chunk_id": "a:0",
        "doc_id": "a",
        "title": "t",
        "text": "body",
        "position": 0,
    }
    result = tools.from_payload(minimal)
    assert result.score == 0.0
    assert result.component_scores is None


def test_list_documents_reads_the_corpus(tmp_path: Path):
    (tmp_path / "a.md").write_text("# Alpha\n\ntext", encoding="utf-8")
    (tmp_path / "b.md").write_text("# Beta\n\ntext", encoding="utf-8")

    documents = tools.list_documents(tmp_path)
    assert [d["doc_id"] for d in documents] == ["a", "b"]
    assert documents[0]["title"] == "Alpha"
    assert documents[0]["source"] == "a.md"


def test_get_document_returns_full_text(tmp_path: Path):
    (tmp_path / "a.md").write_text("# Alpha\n\nfull body text", encoding="utf-8")
    assert "full body text" in tools.get_document(tmp_path, "a")


def test_get_document_raises_for_unknown_id(tmp_path: Path):
    (tmp_path / "a.md").write_text("# Alpha\n\ntext", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown document"):
        tools.get_document(tmp_path, "missing")


def test_server_tools_are_registered():
    """The server must expose exactly the tools documented in the README."""
    import asyncio

    from finsight.mcp.server import mcp

    names = {tool.name for tool in asyncio.run(mcp.list_tools())}
    assert {"search_documents", "list_documents", "get_document"} <= names
