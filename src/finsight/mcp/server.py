"""MCP server exposing the retrieval tools over the Model Context Protocol.

The tool bodies live in :mod:`finsight.mcp.tools` and are shared with the agent
graph, so the server and the in-process agent cannot drift apart.

Run standalone via stdio (default for MCP hosts), or HTTP:
    uv run python -m finsight.mcp.server
    uv run python -m finsight.mcp.server --http   # streamable-http on :8000
"""

import sys
from typing import Annotated, Any, Literal

from fastmcp import FastMCP
from pydantic import Field

from finsight.config import get_settings
from finsight.mcp import tools
from finsight.rag.index import read_manifest
from finsight.rag.retrieve import HybridRetriever

mcp = FastMCP("finsight", strict_input_validation=True)

_retriever: HybridRetriever | None = None


def _get_retriever() -> HybridRetriever:
    global _retriever
    settings = get_settings()
    generation = read_manifest(settings.index_dir).generation
    if _retriever is None or _retriever.generation != generation:
        _retriever = HybridRetriever(settings)
    return _retriever


@mcp.tool()
def search_documents(
    query: str, top_k: Annotated[int | None, Field(strict=True, gt=0)] = None,
) -> list[dict[str, Any]]:
    """Hybrid (vector + BM25) search over the indexed financial corpus."""
    cleaned = tools.validate_search_request(query, top_k)
    return tools.search_documents(_get_retriever(), cleaned, top_k=top_k)


@mcp.tool()
def list_documents() -> list[dict[str, str]]:
    """List the documents available in the indexed corpus."""
    return tools.list_documents(get_settings().corpus_dir)


@mcp.tool()
def get_document(doc_id: str) -> str:
    """Return the full text of a document by its id."""
    return tools.get_document(get_settings().corpus_dir, doc_id)


if __name__ == "__main__":
    transport: Literal["stdio", "streamable-http"] = (
        "streamable-http" if "--http" in sys.argv else "stdio"
    )
    mcp.run(transport=transport)
