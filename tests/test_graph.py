"""Tests for the agent-graph nodes using deterministic fake LLMs."""

from types import SimpleNamespace
from typing import Any

from finsight.graph.nodes import (
    make_answer_node,
    make_finalize_node,
    make_reformulate_node,
    make_retrieve_node,
    make_verify_node,
)
from finsight.graph.state import AgentState
from finsight.rag.models import Chunk, RetrievedChunk


class FakeLLM:
    """Deterministic stand-in for a chat model.

    ``with_structured_output`` deliberately raises ``NotImplementedError`` so these
    tests exercise the JSON-extraction fallback — the path that has to work on
    providers without tool calling.
    """

    def __init__(self, *responses: str) -> None:
        self._responses = list(responses)
        self._index = 0

    def _next(self) -> str:
        if self._index < len(self._responses):
            response = self._responses[self._index]
            self._index += 1
            return response
        return self._responses[-1] if self._responses else ""

    def invoke(self, prompt: str) -> SimpleNamespace:
        return SimpleNamespace(content=self._next())

    def with_structured_output(self, schema: Any) -> Any:
        raise NotImplementedError


class StructuredFakeLLM(FakeLLM):
    """Fake that supports the native structured-output path."""

    def __init__(self, payload: Any) -> None:
        super().__init__()
        self._payload = payload

    def with_structured_output(self, schema: Any) -> Any:
        payload = self._payload
        inner = self

        class _Runner:
            def invoke(self, prompt: str) -> Any:
                return schema(**payload) if isinstance(payload, dict) else inner._payload

        return _Runner()


def _chunk(chunk_id: str, doc_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(chunk_id=chunk_id, doc_id=doc_id, title="t", text="some text", position=0),
        score=1.0,
        rank=1,
        method="rrf",
    )


# --- retrieve --------------------------------------------------------------


def test_retrieve_node_uses_current_query_when_present():
    seen: list[str] = []

    def retrieve(query: str) -> list[RetrievedChunk]:
        seen.append(query)
        return [_chunk("a:0", "a")]

    node = make_retrieve_node(retrieve)
    state: AgentState = {"question": "original", "current_query": "rewritten"}
    node(state)
    assert seen == ["rewritten"]


def test_retrieve_node_falls_back_to_question():
    seen: list[str] = []

    def retrieve(query: str) -> list[RetrievedChunk]:
        seen.append(query)
        return []

    node = make_retrieve_node(retrieve)
    node({"question": "original"})
    assert seen == ["original"]


# --- verify (structured) ---------------------------------------------------


def test_verify_node_parses_json_yes():
    llm = FakeLLM('{"sufficient": true, "reason": "context covers it"}')
    node = make_verify_node(llm)
    out = node({"question": "q", "retrieved": [_chunk("a:0", "a")]})
    assert out["sufficient"] is True
    assert "covers it" in out["verification_note"]


def test_verify_node_parses_json_no():
    llm = FakeLLM('{"sufficient": false, "reason": "missing revenue"}')
    node = make_verify_node(llm)
    out = node({"question": "q", "retrieved": [_chunk("a:0", "a")]})
    assert out["sufficient"] is False


def test_verify_node_tolerates_prose_around_json():
    """The old startswith("YES") check failed on exactly this shape of reply."""
    llm = FakeLLM('The context is sufficient — YES.\n\n{"sufficient": true, "reason": "ok"}')
    node = make_verify_node(llm)
    out = node({"question": "q", "retrieved": [_chunk("a:0", "a")]})
    assert out["sufficient"] is True


def test_verify_node_flags_unparseable_output_rather_than_silently_disagreeing():
    """Empty output used to silently mean 'insufficient' with no explanation."""
    llm = FakeLLM("", "")
    node = make_verify_node(llm)
    out = node({"question": "q", "retrieved": [_chunk("a:0", "a")]})
    assert out["sufficient"] is False
    assert "could not be parsed" in out["verification_note"].lower()


def test_verify_node_short_circuits_without_context():
    node = make_verify_node(FakeLLM("unused"))
    out = node({"question": "q", "retrieved": []})
    assert out["sufficient"] is False
    assert out["verification_note"] == "No context retrieved."


def test_verify_node_uses_native_structured_output_when_available():
    llm = StructuredFakeLLM({"sufficient": True, "reason": "native"})
    node = make_verify_node(llm)
    out = node({"question": "q", "retrieved": [_chunk("a:0", "a")]})
    assert out["sufficient"] is True
    assert out["verification_note"] == "native"


# --- reformulate -----------------------------------------------------------


def test_reformulate_node_increments_attempts():
    node = make_reformulate_node(FakeLLM("airwallex founding year"))
    out = node({"question": "q", "attempts": 0})
    assert out["attempts"] == 1
    assert out["current_query"] == "airwallex founding year"


def test_reformulate_node_falls_back_to_original_question_when_empty():
    """An empty rewrite would otherwise blank the query for the next retrieval."""
    node = make_reformulate_node(FakeLLM(""))
    out = node({"question": "original question", "attempts": 0})
    assert out["current_query"] == "original question"


# --- answer + grounding ----------------------------------------------------


def test_answer_node_extracts_citations_and_redacts():
    node = make_answer_node(FakeLLM("Founded in 2015 [1]. Contact user@example.com"))
    out = node({"question": "q", "retrieved": [_chunk("a:0", "airwallex")]})
    assert "[EMAIL]" in out["answer"]
    assert out["citations"] == ["airwallex"]
    assert out["grounded"] is True


def test_answer_node_reports_ungrounded_when_no_citations():
    node = make_answer_node(FakeLLM("It was founded at some point."))
    out = node({"question": "q", "retrieved": [_chunk("a:0", "airwallex")]})
    assert out["grounded"] is False
    assert out["dangling_citations"] == []


def test_answer_node_reports_dangling_citations():
    node = make_answer_node(FakeLLM("Claim [1] and another claim [5]."))
    out = node({"question": "q", "retrieved": [_chunk("a:0", "airwallex")]})
    assert out["dangling_citations"] == [5]
    assert out["grounded"] is False


# --- finalize --------------------------------------------------------------


def test_finalize_annotates_ungrounded_answer():
    node = make_finalize_node(enforce_grounding=True)
    out = node({"answer": "A confident claim.", "grounded": False, "grounding_note": "no cites"})
    assert "citation warning" in out["answer"]
    assert "no cites" in out["answer"]


def test_finalize_leaves_grounded_answer_alone():
    node = make_finalize_node(enforce_grounding=True)
    out = node({"answer": "Supported [1].", "grounded": True})
    assert out == {}


def test_finalize_is_inert_when_enforcement_disabled():
    node = make_finalize_node(enforce_grounding=False)
    out = node({"answer": "A confident claim.", "grounded": False})
    assert out == {}
