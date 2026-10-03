"""Bounded prompt context with an explicit citation-to-evidence mapping."""

from collections.abc import Callable
from dataclasses import dataclass

from finsight.rag.models import RetrievedChunk

_TRUNCATION_MARKER = "[... truncated ...]"


@dataclass(frozen=True)
class RenderedContext:
    text: str
    citations: dict[int, RetrievedChunk]


def render_source(chunk: RetrievedChunk, number: int) -> str:
    """Render one complete evidence record for generation and claim verification."""
    source = chunk.chunk
    provenance = "; ".join(
        f"{label}: {value}"
        for label, value in (
            ("source", source.source_url), ("published", source.published_at),
            ("retrieved", source.retrieved_at), ("revision", source.revision),
        )
        if value is not None
    )
    header = f"[{number}] ({source.doc_id})"
    return f"{header} [{provenance}] {source.text}" if provenance else f"{header} {source.text}"


def render_context(
    chunks: list[RetrievedChunk], max_chars: int | None = None, *,
    max_tokens: int | None = None, token_counter: Callable[[str], int] | None = None,
) -> RenderedContext:
    """Keep whole chunks within the budget, preserving their original references.

    Oversized chunks are omitted, including the first one. A truncation marker is
    included only when it fits; the citation map always describes exactly the
    evidence present in the returned text.
    """
    if max_chars is not None and max_chars < 0:
        raise ValueError("max_chars must be non-negative")
    if max_tokens is not None and (type(max_tokens) is not int or max_tokens < 0):
        raise ValueError("max_tokens must be a nonnegative integer")

    def fits(text: str) -> bool:
        if max_chars is not None and len(text) > max_chars:
            return False
        if max_tokens is None:
            return True
        # Callers may inject their exact tokenizer. UTF-8 bytes conservatively
        # bound content for byte-based tokenizers; provider framing is separate.
        count = token_counter(text) if token_counter else len(text.encode("utf-8"))
        if type(count) is not int or count < 0:
            raise ValueError("Token counter must return a nonnegative integer")
        return count <= max_tokens

    blocks: list[str] = []
    citations: dict[int, RetrievedChunk] = {}
    for number, chunk in enumerate(chunks, start=1):
        block = render_source(chunk, number)
        if not fits("\n\n".join([*blocks, block])):
            continue
        blocks.append(block)
        citations[number] = chunk
    if len(citations) < len(chunks):
        if fits("\n\n".join([*blocks, _TRUNCATION_MARKER])):
            blocks.append(_TRUNCATION_MARKER)
    return RenderedContext(text="\n\n".join(blocks), citations=citations)


def format_context(chunks: list[RetrievedChunk], max_chars: int | None = None) -> str:
    """Compatibility helper for callers that only need the rendered text."""
    return render_context(chunks, max_chars).text
