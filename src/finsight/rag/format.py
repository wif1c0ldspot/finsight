"""Bounded prompt context with an explicit citation-to-evidence mapping."""

from dataclasses import dataclass

from finsight.rag.models import RetrievedChunk

_TRUNCATION_MARKER = "[... truncated ...]"


@dataclass(frozen=True)
class RenderedContext:
    text: str
    citations: dict[int, RetrievedChunk]


def render_context(
    chunks: list[RetrievedChunk], max_chars: int | None = None
) -> RenderedContext:
    """Keep whole chunks within the budget, preserving their original references.

    Oversized chunks are omitted, including the first one. A truncation marker is
    included only when it fits; the citation map always describes exactly the
    evidence present in the returned text.
    """
    if max_chars is not None and max_chars < 0:
        raise ValueError("max_chars must be non-negative")
    blocks: list[str] = []
    citations: dict[int, RetrievedChunk] = {}
    used = 0
    for number, chunk in enumerate(chunks, start=1):
        block = f"[{number}] ({chunk.chunk.doc_id}) {chunk.chunk.text}"
        cost = len(block) + (2 if blocks else 0)
        if max_chars is not None and used + cost > max_chars:
            continue
        blocks.append(block)
        citations[number] = chunk
        used += cost
    if len(citations) < len(chunks):
        cost = len(_TRUNCATION_MARKER) + (2 if blocks else 0)
        if max_chars is None or used + cost <= max_chars:
            blocks.append(_TRUNCATION_MARKER)
    return RenderedContext(text="\n\n".join(blocks), citations=citations)


def format_context(chunks: list[RetrievedChunk], max_chars: int | None = None) -> str:
    """Compatibility helper for callers that only need the rendered text."""
    return render_context(chunks, max_chars).text
