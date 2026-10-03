"""Context budgets and citations must refer to exactly the evidence shown."""

from types import SimpleNamespace

import pytest

from finsight.graph.nodes import make_answer_node
from finsight.guardrails.validation import assess_grounding
from finsight.rag.format import render_context
from finsight.rag.models import Chunk, RetrievedChunk


def hit(number: int, text: str) -> RetrievedChunk:
    return RetrievedChunk(Chunk(f"d:{number}", "d", "D", text, number), 1.0, number + 1)


@pytest.mark.parametrize("budget", [0, 1, 10, 20, 50, 100, 12000])
def test_whole_render_never_exceeds_budget(budget):
    rendered = render_context([hit(0, "x" * 20000), hit(1, "short")], budget)
    assert len(rendered.text) <= budget
    assert 1 not in rendered.citations
    for number, result in rendered.citations.items():
        assert f"[{number}] (d) {result.chunk.text}" in rendered.text


def test_omitted_middle_chunk_is_not_a_valid_reference():
    rendered = render_context([hit(0, "short"), hit(1, "x" * 100), hit(2, "short")], 50)
    assert set(rendered.citations) == {1, 3}
    assert not assess_grounding("Claim [2].", rendered.citations).is_grounded
    assert assess_grounding("Claim [3].", rendered.citations).cited_chunk_ids == ["d:2"]


class AnswerLLM:
    def invoke(self, prompt):
        assert "[2] (d)" not in prompt
        return SimpleNamespace(content="Claim [2].")


def test_answer_node_records_only_visible_evidence():
    chunks = [hit(0, "short"), hit(1, "x" * 100), hit(2, "short")]
    final = make_answer_node(AnswerLLM(), context_max_chars=50)(
        {"question": "q", "retrieved": chunks}
    )
    assert not final["grounded"]
    assert final["dangling_citations"] == [2]
    assert set(final["context_citations"]) == {1, 3}
    assert len(final["context_text"]) <= 50


def test_negative_budget_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        render_context([], -1)


def test_evaluation_checks_the_same_visible_evidence(tmp_path):
    import json

    from finsight.config import Settings
    from finsight.eval.harness import run_eval

    chunks = [hit(0, "short"), hit(1, "x" * 100), hit(2, "short")]
    context = render_context(chunks, 50)
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps([{"id": "q", "question": "q", "expected_doc_ids": ["d"]}]))

    class Agent:
        def invoke(self, state):
            return {
                "retrieved": chunks,
                "answer": "Claim [2].",
                "context_citations": context.citations,
                "context_text": context.text,
            }

    class Judge:
        def with_structured_output(self, schema):
            raise NotImplementedError

        def invoke(self, prompt):
            assert "[3] (d) short" in prompt
            assert "[2] (d)" not in prompt
            assert "x" * 100 not in prompt
            return SimpleNamespace(content='{"score": 1}')

    result = run_eval(Settings(golden_file=golden), agent=Agent(), judge=Judge())[0]
    assert not result.grounded
    assert result.dangling_citations == [2]
    assert result.faithfulness == 0.0
