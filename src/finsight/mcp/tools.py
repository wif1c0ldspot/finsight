"""MCP tool layer: retrieval exposed as tools.

This is the single implementation of the retrieval tool surface. Two callers share
it:

* ``mcp/server.py`` registers these functions with FastMCP so any external MCP host
  can drive them over stdio or HTTP;
* the agent graph calls the *same* functions in-process when
  ``FINSIGHT_USE_MCP_TOOLS`` is set.

Previously the agent bypassed the tools entirely and called ``HybridRetriever``
directly, so the MCP server was a parallel demo and the two paths could drift.
Routing both through this module means one contract, two transports.

Tool payloads are plain JSON-serialisable dicts — that is what an MCP transport
can carry — so :func:`make_retrieve_tool` rehydrates them back into typed
``RetrievedChunk`` objects for the graph. The round-trip is deliberate: it keeps
the graph typed while still proving the tool contract works.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from finsight.guardrails.validation import validate_query
from finsight.rag.ingest import chunk_from_dict, chunk_to_dict, load_documents
from finsight.rag.models import RetrievalFilter, RetrievedChunk
from finsight.rag.retrieve import HybridRetriever

#: Adapter type used by the graph's retrieve node.
RetrieveFn = Callable[[str], list[RetrievedChunk]]


def to_payload(result: RetrievedChunk) -> dict[str, Any]:
    """Serialise a retrieval result into a transport-safe dict."""
    chunk = result.chunk
    return {
        **chunk_to_dict(chunk),
        "score": result.score,
        "rank": result.rank,
        "method": result.method,
        "component_scores": result.component_scores,
    }


def from_payload(payload: dict[str, Any]) -> RetrievedChunk:
    """Rebuild a typed result from a tool payload."""
    chunk = chunk_from_dict(payload)
    component_scores = payload.get("component_scores")
    return RetrievedChunk(
        chunk=chunk,
        score=float(payload.get("score", 0.0)),
        rank=int(payload.get("rank", 0)),
        method="rrf",
        component_scores=dict(component_scores) if component_scores else None,
    )


def validate_search_request(query: str, top_k: int | None = None) -> str:
    """Validate shared in-process/transport inputs before opening an index."""
    if not isinstance(query, str):
        raise ValueError("Query must be a string.")
    cleaned = validate_query(query)
    if top_k is not None and (type(top_k) is not int or top_k <= 0):
        raise ValueError("top_k must be a positive integer.")
    return cleaned


def search_documents(
    retriever: HybridRetriever,
    query: str,
    top_k: int | None = None,
    filters: RetrievalFilter | None = None,
) -> list[dict[str, Any]]:
    """Hybrid (vector + BM25) search over the indexed corpus."""
    cleaned = validate_search_request(query, top_k)
    results = (
        retriever.retrieve(cleaned, top_k=top_k)
        if filters is None
        else retriever.retrieve(cleaned, top_k=top_k, filters=filters)
    )
    return [to_payload(result) for result in results]


def list_documents(corpus_dir: Path) -> list[dict[str, str | None]]:
    """List the documents available in the corpus."""
    return [
        {
            "doc_id": d.doc_id,
            "title": d.title,
            "source": d.source,
            "source_url": d.source_url,
            "published_at": d.published_at,
            "retrieved_at": d.retrieved_at,
            "revision": d.revision,
        }
        for d in load_documents(corpus_dir)
    ]


def get_document(corpus_dir: Path, doc_id: str) -> str:
    """Return the full text of a document by id."""
    for document in load_documents(corpus_dir):
        if document.doc_id == doc_id:
            return document.text
    raise ValueError(f"Unknown document: {doc_id}")


def get_document_record(corpus_dir: Path, doc_id: str) -> dict[str, str | None]:
    """Return full document text and recorded provenance without inferred dates."""
    from dataclasses import asdict

    for document in load_documents(corpus_dir):
        if document.doc_id == doc_id:
            return asdict(document)
    raise ValueError(f"Unknown document: {doc_id}")


def make_retrieve_tool(
    retriever: HybridRetriever, filters: RetrievalFilter | None = None, top_k: int | None = None,
) -> RetrieveFn:
    """Adapt the retrieval tool into the callable the graph's retrieve node expects."""

    def retrieve(query: str) -> list[RetrievedChunk]:
        return [from_payload(p) for p in search_documents(
            retriever, query, top_k=top_k, filters=filters
        )]

    return retrieve
