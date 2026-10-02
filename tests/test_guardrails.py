"""Tests for guardrails: PII redaction, query validation, grounding assessment."""

import pytest

from finsight.guardrails.validation import (
    answer_is_grounded,
    assess_grounding,
    extract_citations,
    redact_pii,
    validate_query,
)
from finsight.rag.models import Chunk, RetrievedChunk


def _chunk(chunk_id: str, doc_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(chunk_id=chunk_id, doc_id=doc_id, title="t", text="x", position=0),
        score=1.0,
        rank=1,
        method="rrf",
    )


def test_redact_pii_email():
    assert redact_pii("Contact user@example.com today") == "Contact [EMAIL] today"


def test_redact_pii_phone():
    assert "[PHONE]" in redact_pii("Call 91234567 now")


def test_redact_pii_nric():
    assert "[NRIC]" in redact_pii("My NRIC is S1234567D")


def test_validate_query_rejects_empty():
    with pytest.raises(ValueError):
        validate_query("   ")


def test_validate_query_rejects_overlong():
    with pytest.raises(ValueError, match="exceeds"):
        validate_query("x" * 2001)


def test_validate_query_strips_whitespace():
    assert validate_query("  hello  ") == "hello"


def test_extract_citations_maps_doc_ids():
    chunks = [_chunk("a:0", "airwallex"), _chunk("s:0", "stripe")]
    answer = "Airwallex was founded in 2015 [1], while Stripe is different [2]."
    assert extract_citations(answer, chunks) == ["airwallex", "stripe"]


def test_answer_is_grounded():
    chunks = [_chunk("a:0", "airwallex")]
    assert answer_is_grounded("Founded in 2015 [1].", chunks)
    assert not answer_is_grounded("No citations here.", chunks)


# --- grounding enforcement (new behaviour) ---------------------------------


def test_assess_grounding_flags_dangling_citations():
    """A marker pointing past the retrieved set is a fabricated reference.

    These used to be silently ignored, which made a hallucinated source look the
    same as no source at all.
    """
    chunks = [_chunk("a:0", "airwallex")]
    result = assess_grounding("Supported [1] but also invented [7].", chunks)
    assert result.dangling_citations == [7]
    assert not result.is_grounded
    assert result.cited_chunk_ids == ["a:0"]


def test_assess_grounding_all_dangling_is_ungrounded():
    chunks = [_chunk("a:0", "airwallex")]
    result = assess_grounding("Everything is invented [4].", chunks)
    assert result.dangling_citations == [4]
    assert result.cited_chunk_ids == []
    assert not result.is_grounded


def test_assess_grounding_clean_citation_is_grounded():
    chunks = [_chunk("a:0", "airwallex"), _chunk("s:0", "stripe")]
    result = assess_grounding("Airwallex [1] and Stripe [2].", chunks)
    assert result.is_grounded
    assert result.dangling_citations == []
    assert result.cited_chunk_ids == ["a:0", "s:0"]


def test_assess_grounding_dedupes_repeated_citations():
    chunks = [_chunk("a:0", "airwallex")]
    result = assess_grounding("Airwallex [1] is old [1] and large [1].", chunks)
    assert result.cited_chunk_ids == ["a:0"]
    assert result.citation_count == 3


def test_assess_grounding_reason_mentions_missing_sources():
    chunks = [_chunk("a:0", "airwallex")]
    result = assess_grounding("Invented [9].", chunks)
    assert "dangling" in result.reason.lower()


def test_ungrounded_answer_is_reported_when_no_chunks_retrieved():
    result = assess_grounding("A confident answer [1].", [])
    assert not result.is_grounded
    assert result.dangling_citations == [1]
