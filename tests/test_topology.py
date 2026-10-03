"""Topology tests for the compiled agent graph.

``graph/builder.py`` previously had no test at all: it constructed its own
retriever and LLM, so the graph could only be built against live services. That
left the routers, the conditional edges and the loop bound — the most bug-prone
parts — completely unexercised. These tests drive the real compiled graph with
fakes.
"""

from types import SimpleNamespace
from typing import Any, cast

import pytest

from finsight.config import Settings
from finsight.graph.builder import build_agent
from finsight.rag.models import Chunk, RetrievedChunk
from finsight.rag.retrieve import HybridRetriever


class ScriptedLLM:
    """Returns queued responses in order, and records every prompt it saw."""

    def __init__(self, *responses: str) -> None:
        self._responses = list(responses)
        self._index = 0
        self.prompts: list[str] = []

    def _next(self) -> str:
        if self._index < len(self._responses):
            response = self._responses[self._index]
            self._index += 1
            return response
        return self._responses[-1] if self._responses else ""

    def invoke(self, prompt: str) -> SimpleNamespace:
        self.prompts.append(prompt)
        return SimpleNamespace(content=self._next())

    def with_structured_output(self, schema: Any) -> Any:
        raise NotImplementedError


class FakeRetriever:
    """Records queries and returns a deterministic chunk set."""

    def __init__(self, chunk_ids: tuple[str, ...] = ("a:0",)) -> None:
        self.queries: list[str] = []
        self._chunk_ids = chunk_ids

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        self.queries.append(query)
        return [
            RetrievedChunk(
                chunk=Chunk(
                    chunk_id=chunk_id,
                    doc_id=chunk_id.split(":")[0],
                    title="t",
                    text="body text",
                    position=0,
                ),
                score=1.0,
                rank=index,
                method="rrf",
            )
            for index, chunk_id in enumerate(self._chunk_ids, start=1)
        ]


def _settings(**overrides: Any) -> Settings:
    return Settings(**overrides)


def _agent(settings: Settings, retriever: FakeRetriever, llm: ScriptedLLM) -> Any:
    # The fake satisfies the retriever protocol used by the graph.
    return build_agent(
        settings, retriever=cast(HybridRetriever, retriever), llm=cast(Any, llm)
    )


def _seed(question: str = "Who founded Airwallex?") -> dict[str, Any]:
    return {"question": question, "current_query": question, "attempts": 0}


# --- happy path ------------------------------------------------------------


def test_sufficient_grounded_answer_runs_straight_through():
    retriever = FakeRetriever()
    llm = ScriptedLLM(
        '{"sufficient": true, "reason": "ok"}',
        "Airwallex was founded in 2015 [1].",
    )
    final = _agent(_settings(), retriever, llm).invoke(_seed())

    assert final["sufficient"] is True
    assert final["grounded"] is True
    assert final["attempts"] == 0
    assert len(retriever.queries) == 1
    assert "citation warning" not in final["answer"]


# --- the verify branch -----------------------------------------------------


def test_insufficient_then_sufficient_reformulates_once():
    retriever = FakeRetriever()
    llm = ScriptedLLM(
        '{"sufficient": false, "reason": "missing"}',  # verify #1
        "airwallex founders 2015",                     # reformulate
        '{"sufficient": true, "reason": "ok"}',        # verify #2
        "Founded in 2015 [1].",                        # answer
    )
    final = _agent(_settings(), retriever, llm).invoke(_seed())

    assert final["attempts"] == 1
    assert retriever.queries[0] == "Who founded Airwallex?"  # original
    assert retriever.queries[1] == "airwallex founders 2015"  # reformulated
    assert len(retriever.queries) == 2
    assert final["grounded"] is True


def test_loop_bound_stops_reformulating_at_max_attempts():
    """The bound is the guard against an unbounded retry loop."""
    retriever = FakeRetriever()
    llm = ScriptedLLM(
        '{"sufficient": false, "reason": "no"}',
        "rewrite one",
        '{"sufficient": false, "reason": "still no"}',
        "rewrite two",
        '{"sufficient": false, "reason": "still no"}',
        "Founded in 2015 [1].",
    )
    settings = _settings(max_retrieval_attempts=2)
    final = _agent(settings, retriever, llm).invoke(_seed())

    # 1 initial retrieval + 2 reformulation retries.
    assert len(retriever.queries) == 3
    assert final["attempts"] == 2


# --- the grounding branch (new decision point) -----------------------------


def test_ungrounded_answer_triggers_another_retrieval():
    retriever = FakeRetriever()
    llm = ScriptedLLM(
        '{"sufficient": true, "reason": "ok"}',  # verify #1
        "It was founded at some point.",          # answer #1: no citations -> ungrounded
        "airwallex founding",                     # reformulate (grounding path)
        '{"sufficient": true, "reason": "ok"}',  # verify #2
        "Founded in 2015 [1].",                   # answer #2: grounded
    )
    final = _agent(_settings(), retriever, llm).invoke(_seed())

    assert len(retriever.queries) == 2
    assert final["grounded"] is True
    assert final["attempts"] == 1


def test_ungrounded_with_budget_exhausted_annotates_instead_of_looping():
    retriever = FakeRetriever()
    llm = ScriptedLLM(
        '{"sufficient": true, "reason": "ok"}',
        "Unsupported claim.",      # answer #1 ungrounded
        "rewrite",                 # reformulate
        '{"sufficient": true, "reason": "ok"}',
        "Unsupported claim again.",  # answer #2 ungrounded, budget exhausted
    )
    settings = _settings(max_retrieval_attempts=1)
    final = _agent(settings, retriever, llm).invoke(_seed())

    assert final["grounded"] is False
    assert "citation warning" in final["answer"]


def test_grounding_enforcement_can_be_disabled():
    retriever = FakeRetriever()
    llm = ScriptedLLM(
        '{"sufficient": true, "reason": "ok"}',
        "Unsupported claim.",
    )
    settings = _settings(enforce_grounding=False)
    final = _agent(settings, retriever, llm).invoke(_seed())

    # No extra retrieval, and no warning appended.
    assert len(retriever.queries) == 1
    assert "citation warning" not in final["answer"]
    assert final["grounded"] is False


# --- MCP transport ---------------------------------------------------------


def test_mcp_tool_path_is_used_when_enabled():
    retriever = FakeRetriever()
    llm = ScriptedLLM('{"sufficient": true, "reason": "ok"}', "Founded in 2015 [1].")
    settings = _settings(use_mcp_tools=True)
    final = _agent(settings, retriever, llm).invoke(_seed())
    # Routing through the tool layer must preserve typed results.
    assert final["retrieved"][0].chunk.chunk_id == "a:0"
    assert final["retrieved"][0].component_scores is None


def test_direct_path_is_used_when_mcp_disabled():
    retriever = FakeRetriever()
    llm = ScriptedLLM('{"sufficient": true, "reason": "ok"}', "Founded in 2015 [1].")
    settings = _settings(use_mcp_tools=False)
    final = _agent(settings, retriever, llm).invoke(_seed())
    assert final["grounded"] is True


# --- construction ----------------------------------------------------------


def test_build_agent_is_injectable_and_constructs_offline():
    """Building the graph must not require a live retriever or model service."""
    retriever = FakeRetriever()
    llm = ScriptedLLM('{"sufficient": true, "reason": "ok"}', "Answer [1].")
    graph = _agent(_settings(), retriever, llm)
    assert hasattr(graph, "invoke")


def test_settings_validation_requires_base_url_for_compatible_provider():
    with pytest.raises(Exception, match="llm_base_url is required"):
        _settings(llm_provider="openai_compatible")

    with pytest.raises(Exception, match="embed_base_url is required"):
        _settings(embed_provider="openai_compatible")


def test_reformulation_receives_failure_and_query_history():
    retriever = FakeRetriever()
    llm = ScriptedLLM(
        '{"sufficient": false, "reason": "missing founders"}',
        "airwallex founders",
        '{"sufficient": true, "reason": "ok"}',
        "Unsupported claim.",
        "airwallex founder biographies",
        '{"sufficient": true, "reason": "ok"}',
        "Founded [1].",
    )
    final = _agent(_settings(), retriever, llm).invoke(_seed())
    first_rewrite, second_rewrite = llm.prompts[1], llm.prompts[4]
    assert "missing founders" in first_rewrite
    assert "Who founded Airwallex?" in second_rewrite
    assert "airwallex founders" in second_rewrite
    assert "Answer contains no citation markers." in second_rewrite
    assert final["attempted_queries"] == retriever.queries


@pytest.mark.parametrize("rewrite", ["", "  WHO founded   AIRWALLEX?  "])
def test_empty_or_repeated_rewrite_does_not_repeat_retrieval(rewrite: str):
    retriever = FakeRetriever()
    llm = ScriptedLLM(
        '{"sufficient": false, "reason": "missing"}',
        rewrite,
        "The context does not contain the answer.",
    )
    final = _agent(_settings(max_retrieval_attempts=5), retriever, llm).invoke(_seed())
    assert len(retriever.queries) == 1
    assert final["reformulation_exhausted"] is True
    assert final["attempts"] == 1
    assert "citation warning" in final["answer"]


def test_rewrite_cannot_cycle_back_to_an_earlier_query():
    retriever = FakeRetriever()
    llm = ScriptedLLM(
        '{"sufficient": false, "reason": "missing"}',
        "founders",
        '{"sufficient": false, "reason": "still missing"}',
        "Who founded Airwallex?",
        "The context does not contain the answer.",
    )
    final = _agent(_settings(max_retrieval_attempts=5), retriever, llm).invoke(_seed())
    assert retriever.queries == ["Who founded Airwallex?", "founders"]
    assert final["reformulation_exhausted"] is True


def test_no_evidence_finishes_without_model_generation():
    retriever = FakeRetriever(chunk_ids=())
    llm = ScriptedLLM("This unsupported answer must never be generated [1].")
    final = _agent(_settings(max_retrieval_attempts=0), retriever, llm).invoke(_seed())
    assert final["no_evidence"] is True
    assert final["grounded"] is False
    assert "I cannot answer" in final["answer"]
    assert "citation warning" not in final["answer"]
    assert final["context_citations"] == {}
    assert llm.prompts == []


def test_all_context_omitted_does_not_generate_an_answer():
    retriever = FakeRetriever()
    llm = ScriptedLLM("Must not be used")
    final = _agent(
        _settings(max_retrieval_attempts=0, context_max_chars=1), retriever, llm
    ).invoke(_seed())
    assert final["no_evidence"] is True
    assert final["context_text"] == ""
    assert llm.prompts == []


@pytest.mark.parametrize("question", ["   ", "q" * 2001])
def test_graph_rejects_invalid_query_before_retrieval(question):
    retriever = FakeRetriever()
    with pytest.raises(ValueError):
        _agent(_settings(), retriever, ScriptedLLM()).invoke(_seed(question))
    assert retriever.queries == []


def test_overlong_rewrite_does_not_reach_retrieval():
    retriever = FakeRetriever()
    llm = ScriptedLLM(
        '{"sufficient": false}', "x" * 2001, "No answer is available."
    )
    final = _agent(_settings(), retriever, llm).invoke(_seed())
    assert len(retriever.queries) == 1
    assert final["reformulation_exhausted"] is True


def test_graph_retry_budget_has_enough_execution_steps():
    retriever = FakeRetriever()
    responses = []
    for attempt in range(6):
        responses += ['{"sufficient": true}', 'Unsupported answer.']
        if attempt < 5:
            responses.append(f"new query {attempt}")
    final = _agent(
        _settings(max_retrieval_attempts=5), retriever, ScriptedLLM(*responses)
    ).invoke(_seed())
    assert final["attempts"] == 5
    assert len(retriever.queries) == 6
    assert "citation warning" in final["answer"]
