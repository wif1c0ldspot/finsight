"""Input/output guardrails: PII redaction, query validation, grounding assessment.

The grounding helpers previously returned a boolean presence check and were called
from the evaluation harness alone, so nothing in the live path was enforced.
:func:`assess_grounding` now returns a structured verdict — including *dangling*
citations that point at chunks which do not exist — which the answer node consumes
and the graph routes on.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from finsight.rag.models import RetrievedChunk

# Illustrative patterns — tune per domain. Order matters (email before phone).
_PII_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"), "[EMAIL]"),
    (re.compile(r"\b(?:\+?65[-\s]?)?[89]\d{7}\b"), "[PHONE]"),
    (re.compile(r"\b[STFG]\d{7}[A-Z]\b"), "[NRIC]"),
    (re.compile(r"\b\d{4}[-\s]\d{4}[-\s]\d{4}[-\s]\d{4}\b"), "[CARD]"),
]

_CITATION = re.compile(r"\[(\d+)\]")
_MAX_QUERY_CHARS = 2000


def redact_pii(text: str) -> str:
    """Replace email/phone/NRIC/card patterns with placeholders."""
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def validate_query(text: str) -> str:
    """Validate and normalise a user query. Raises ValueError on bad input."""
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Query is empty.")
    if len(cleaned) > _MAX_QUERY_CHARS:
        raise ValueError(f"Query exceeds {_MAX_QUERY_CHARS} characters.")
    return cleaned


@dataclass(frozen=True)
class GroundingAssessment:
    """Structured result of checking an answer against its retrieved context."""

    cited_chunk_ids: list[str] = field(default_factory=list)
    cited_doc_ids: list[str] = field(default_factory=list)
    dangling_citations: list[int] = field(default_factory=list)
    citation_count: int = 0

    @property
    def is_grounded(self) -> bool:
        """True when the answer cites a real chunk and no phantom ones.

        A dangling marker (``[7]`` when only four chunks were retrieved) is a
        fabricated reference, which is worse than no citation at all.
        """
        return bool(self.cited_chunk_ids) and not self.dangling_citations

    @property
    def reason(self) -> str:
        if self.citation_count == 0:
            return "Answer contains no citation markers."
        if self.dangling_citations and not self.cited_chunk_ids:
            return (
                "All citation markers are dangling "
                f"({self.dangling_citations}); no source actually supports the answer."
            )
        if self.dangling_citations:
            return f"Answer cites non-existent sources: {self.dangling_citations}."
        return f"Answer cites {len(self.cited_chunk_ids)} retrieved chunk(s)."


def assess_grounding(
    answer: str, chunks: Iterable[RetrievedChunk] | Mapping[int, RetrievedChunk]
) -> GroundingAssessment:
    """Check every citation marker in ``answer`` against the retrieved chunks.

    Markers are 1-indexed references into ``chunks``. Out-of-range markers are
    recorded as dangling rather than ignored (the previous behaviour) or raised
    (which would let one hallucinated marker crash a run).
    """
    citation_map = (
        dict(chunks) if isinstance(chunks, Mapping) else dict(enumerate(chunks, start=1))
    )
    cited_chunk_ids: list[str] = []
    cited_doc_ids: list[str] = []
    dangling: list[int] = []
    count = 0

    for match in _CITATION.finditer(answer):
        count += 1
        number = int(match.group(1))
        if number in citation_map:
            cited_chunk_ids.append(citation_map[number].chunk.chunk_id)
            cited_doc_ids.append(citation_map[number].chunk.doc_id)
        else:
            dangling.append(int(match.group(1)))

    def _dedupe(values: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for value in values:
            if value not in seen:
                seen.add(value)
                ordered.append(value)
        return ordered

    return GroundingAssessment(
        cited_chunk_ids=_dedupe(cited_chunk_ids),
        cited_doc_ids=_dedupe(cited_doc_ids),
        dangling_citations=sorted(set(dangling)),
        citation_count=count,
    )


def extract_citations(answer: str, chunks: Iterable[RetrievedChunk]) -> list[str]:
    """Return the source ``doc_id`` values actually cited in the answer."""
    return assess_grounding(answer, chunks).cited_doc_ids


def answer_is_grounded(answer: str, chunks: Iterable[RetrievedChunk]) -> bool:
    """True if the answer cites at least one retrieved chunk and no phantoms."""
    return assess_grounding(answer, chunks).is_grounded
