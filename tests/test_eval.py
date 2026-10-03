"""Tests for evaluation metrics and the harness.

Two behaviours worth pinning: faithfulness must be ``None`` (not ``0.0``) when
there is nothing to judge, and negative golden cases must be scored on abstention
rather than on retrieval.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from finsight.config import Settings
from finsight.eval.harness import EvalResult, load_golden, run_eval, summarize
from finsight.eval.metrics import (
    abstained,
    answer_matches_reference,
    faithfulness,
    semantic_abstention,
)
from finsight.rag.models import Chunk, RetrievedChunk


class JudgeLLM:
    """Returns queued JSON payloads; records prompts."""

    def __init__(self, *responses: str) -> None:
        self._responses = list(responses)
        self._index = 0
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> SimpleNamespace:
        self.prompts.append(prompt)
        if self._index < len(self._responses):
            response = self._responses[self._index]
            self._index += 1
        else:
            response = self._responses[-1] if self._responses else ""
        return SimpleNamespace(content=response)

    def with_structured_output(self, schema: Any) -> Any:
        raise NotImplementedError


def _chunk(chunk_id: str, doc_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(chunk_id=chunk_id, doc_id=doc_id, title="t", text="body", position=0),
        score=1.0,
        rank=1,
        method="rrf",
    )


# --- faithfulness ----------------------------------------------------------


def test_faithfulness_parses_typed_json():
    judge = JudgeLLM('{"score": 5, "rationale": "fully supported"}')
    assert faithfulness(judge, "q", "an answer", [_chunk("a:0", "a")]) == 1.0


def test_faithfulness_maps_the_scale_to_zero_one():
    judge = JudgeLLM('{"score": 1, "rationale": "fabricated"}')
    assert faithfulness(judge, "q", "an answer", [_chunk("a:0", "a")]) == 0.0


def test_faithfulness_is_not_fooled_by_prose_scores():
    """re.search(r"\\d+") read this as 4; the model is actually asking for 3."""
    judge = JudgeLLM('Considering it carefully, I would say {"score": 3, "rationale": "partial"} — '
                     'though 5 is arguable.')
    assert faithfulness(judge, "q", "an answer", [_chunk("a:0", "a")]) == pytest.approx(0.5)


def test_faithfulness_rejects_out_of_range_scores():
    judge = JudgeLLM('{"score": 10, "rationale": "too high"}')
    # Schema validation fails, so the metric is undefined rather than clamped to 5.
    assert faithfulness(judge, "q", "an answer", [_chunk("a:0", "a")]) is None


def test_faithfulness_is_none_without_retrieval():
    """Used to return 0.0, collapsing 'retrieval failed' with 'answer unfaithful'."""
    assert faithfulness(JudgeLLM("unused"), "q", "an answer", []) is None


def test_faithfulness_is_none_without_an_answer():
    assert faithfulness(JudgeLLM("unused"), "q", "", [_chunk("a:0", "a")]) is None


# --- correctness + abstention ---------------------------------------------


def test_correctness_judges_against_the_reference():
    judge = JudgeLLM('{"correct": true, "rationale": "matches"}')
    assert answer_matches_reference(judge, "q", "2015", "2015") == 1.0


def test_correctness_detects_a_mismatch():
    judge = JudgeLLM('{"correct": false, "rationale": "wrong year"}')
    assert answer_matches_reference(judge, "q", "2014", "2015") == 0.0


def test_correctness_is_none_without_a_reference():
    assert answer_matches_reference(JudgeLLM("unused"), "q", "a", "") is None


@pytest.mark.parametrize(
    "answer",
    [
        "I do not know based on the context provided.",
        "There is not enough information in the context.",
        "The context does not contain the answer.",
        "This cannot be determined from the documents.",
        "I'm unable to answer that from the retrieved sources.",
    ],
)
def test_abstention_is_detected(answer: str):
    assert abstained(answer)


def test_a_real_answer_is_not_an_abstention():
    assert not abstained("Airwallex was founded in 2015 by Jack Zhang [1].")


# --- harness ---------------------------------------------------------------


def _golden(tmp_path: Path) -> Path:
    path = tmp_path / "golden.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "a1",
                    "question": "Who founded Airwallex?",
                    "expected_doc_ids": ["a"],
                    "ground_truth": "Jack Zhang and others, 2015.",
                },
                {
                    "id": "n1",
                    "question": "What was the 2023 revenue?",
                    "answerable": False,
                    "expected_doc_ids": [],
                    "ground_truth": "",
                },
            ]
        ),
        encoding="utf-8",
    )
    return path


class StubAgent:
    def __init__(self, responses: dict[str, dict[str, Any]]) -> None:
        self._responses = responses

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        return self._responses[state["question"]]


def test_load_golden_missing_file_raises_with_a_pointer(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="FINSIGHT_GOLDEN_FILE"):
        load_golden(tmp_path / "nope.json")


def test_load_golden_rejects_an_empty_set(tmp_path: Path):
    path = tmp_path / "empty.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty"):
        load_golden(path)


def test_run_eval_scores_answerable_and_negative_cases(tmp_path: Path):
    golden = _golden(tmp_path)
    agent = StubAgent(
        {
            "Who founded Airwallex?": {
                "retrieved": [_chunk("a:0", "a")],
                "answer": "Founded in 2015 by Jack Zhang [1].",
            },
            "What was the 2023 revenue?": {
                "retrieved": [_chunk("a:0", "a")],
                "answer": "I do not know; the context does not contain that figure.",
            },
        }
    )
    judge = JudgeLLM(
        '{"score": 5, "rationale": "supported"}',
        '{"correct": true, "rationale": "matches"}',
        '{"declines_to_answer": true, "provides_answer": false}',
        '{"score": 5}',
    )
    settings = Settings(golden_file=golden)
    results = run_eval(settings, agent=cast(Any, agent), judge=cast(Any, judge))

    assert len(results) == 2
    answerable = next(r for r in results if r.answerable)
    negative = next(r for r in results if not r.answerable)

    assert answerable.grounded is True
    assert answerable.recall_at_k == 1.0
    assert answerable.correctness == 1.0
    assert answerable.abstained is None
    assert negative.grounded is False
    assert negative.abstained is True


def test_summarize_reports_negative_cases_separately(tmp_path: Path):
    golden = _golden(tmp_path)
    agent = StubAgent(
        {
            "Who founded Airwallex?": {
                "retrieved": [_chunk("a:0", "a")],
                "answer": "Founded in 2015 [1].",
            },
            "What was the 2023 revenue?": {
                "retrieved": [_chunk("a:0", "a")],
                "answer": "I do not know.",
            },
        }
    )
    judge = JudgeLLM(
        '{"score": 4, "rationale": "ok"}',
        '{"correct": true, "rationale": "match"}',
        '{"declines_to_answer": true, "provides_answer": false}',
        '{"score": 4}',
    )
    results = run_eval(Settings(golden_file=golden), agent=cast(Any, agent), judge=cast(Any, judge))
    summary = summarize(results)

    assert summary["n_answerable"] == 1
    assert summary["n_negative"] == 1
    assert summary["abstention_rate_on_negative"] == 1.0
    assert summary["recall_at_k"] == 1.0
    # Retrieval metrics are computed over answerable cases only.
    assert summary["retrieval_failures"] == 0


def test_summarize_flags_a_self_graded_judge():
    summary = summarize([], judge_is_same_model=True)
    assert summary["n"] == 0
    result = EvalResult(
        id="x",
        question="q",
        answerable=True,
        recall_at_k=1.0,
        reciprocal_rank=1.0,
        ndcg=1.0,
        grounded=True,
        abstained=False,
        answer="a [1]",
    )
    flagged = summarize([result], judge_is_same_model=True)
    assert "caveat" in flagged


def test_summarize_handles_an_empty_result_set():
    assert summarize([]) == {"n": 0}


def test_ndcg_counts_repeated_documents_only_once():
    from finsight.eval.metrics import ndcg_at_k

    assert ndcg_at_k(["a", "a", "a"], ["a"], 3) == 1.0
    assert 0 < ndcg_at_k(["a", "a", "b"], ["a", "b"], 3) < 1.0
    assert ndcg_at_k(["a", "b"], ["a", "b"], 2) == 1.0


def test_faithfulness_uses_only_rendered_context():
    judge = JudgeLLM('{"score": 5}')
    assert faithfulness(
        judge, "q", "answer", [_chunk("a:0", "a")], context_text="visible excerpt"
    ) == 1.0
    assert "visible excerpt" in judge.prompts[0]
    assert "body" not in judge.prompts[0]
    assert faithfulness(judge, "q", "answer", [_chunk("a:0", "a")], context_text="") is None


def test_semantic_abstention_rejects_refusal_followed_by_a_guess():
    answer = "I do not know the revenue, but it was probably $50 million."
    judge = JudgeLLM('{"declines_to_answer": true, "provides_answer": true}')
    assert semantic_abstention(judge, "What was revenue?", answer) is False
    assert answer in judge.prompts[0]
    assert "What was revenue?" in judge.prompts[0]
    assert "A refusal followed by a guessed answer is NOT successful abstention" in judge.prompts[0]


def test_semantic_abstention_accepts_refusal_without_legacy_keywords():
    answer = "The supplied sources leave this figure undisclosed; please provide audited accounts."
    assert abstained(answer) is False  # Legacy heuristic misses the meaning.
    judge = JudgeLLM('{"declines_to_answer": true, "provides_answer": false}')
    assert semantic_abstention(judge, "What was revenue?", answer) is True


@pytest.mark.parametrize("response", [
    "yes", '{}', '{"declines_to_answer": true}',
    '{"declines_to_answer": "true", "provides_answer": false}',
])
def test_semantic_abstention_malformed_verdict_is_unscored(response: str):
    assert semantic_abstention(JudgeLLM(response), "q", "I do not know.") is None


@pytest.mark.parametrize("answer", ["", " \n\t"])
def test_semantic_abstention_empty_answer_fails_without_judge_call(answer: str):
    judge = JudgeLLM('{"declines_to_answer": true, "provides_answer": false}')
    assert semantic_abstention(judge, "q", answer) is False
    assert judge.prompts == []


def test_harness_abstention_summary_exposes_scored_and_error_denominators(tmp_path: Path):
    path = tmp_path / "negative.json"
    questions = ["refusal", "refusal then guess", "broken verdict", "empty"]
    path.write_text(json.dumps([
        {"question": question, "answerable": False} for question in questions
    ]))
    agent = StubAgent({
        "refusal": {"answer": "The source leaves that undisclosed."},
        "refusal then guess": {"answer": "I do not know, but it was probably $50 million."},
        "broken verdict": {"answer": "I do not know."},
        "empty": {"answer": ""},
    })
    judge = JudgeLLM(
        '{"declines_to_answer": true, "provides_answer": false}',
        '{"declines_to_answer": true, "provides_answer": true}',
        "malformed", "malformed",
    )
    results = run_eval(Settings(golden_file=path), agent=agent, judge=judge)
    assert [result.abstained for result in results] == [True, False, None, False]
    summary = summarize(results)
    assert summary["n_negative"] == 4
    assert summary["n_abstention_scored"] == 3
    assert summary["n_abstention_errors"] == 1
    assert summary["abstention_rate_on_negative"] == 0.333
    all_failed = summarize([results[2]])
    assert all_failed["abstention_rate_on_negative"] is None
    assert all_failed["n_abstention_scored"] == 0
    assert all_failed["n_abstention_errors"] == 1


def test_answerable_cases_do_not_request_abstention_judgment(tmp_path: Path):
    path = tmp_path / "answerable.json"
    path.write_text(json.dumps([{"question": "q", "answerable": True}]))
    judge = JudgeLLM("unused")
    results = run_eval(
        Settings(golden_file=path), agent=StubAgent({"q": {"answer": "a"}}), judge=judge
    )
    assert results[0].abstained is None
    assert judge.prompts == []
    summary = summarize(results)
    assert summary["n_abstention_scored"] == 0
    assert summary["n_abstention_errors"] == 0
    assert summary["abstention_rate_on_negative"] is None
