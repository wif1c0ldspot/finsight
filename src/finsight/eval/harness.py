"""Evaluation harness: run a golden set end-to-end and aggregate metrics.

Changes from the original:

* The judge is a **separate model** where configured (``FINSIGHT_JUDGE_MODEL``),
  instead of the model that wrote the answer grading itself. When no separate
  judge is set the summary says so rather than reporting a self-graded score
  without qualification.
* **Negative cases** are supported: a case with ``"answerable": false`` expects an
  explicit abstention, not a citation.
* Metrics are **rank-aware** (recall@k, MRR, nDCG), and retrieval failures are
  reported separately from unfaithful answers.
* ``run_eval`` accepts an injected agent and judge so the harness itself is
  testable without any model service running.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from finsight.config import Settings
from finsight.eval.metrics import (
    answer_matches_reference,
    faithfulness,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    semantic_abstention,
)
from finsight.graph.builder import build_agent
from finsight.guardrails.validation import assess_grounding
from finsight.llm import build_judge
from finsight.rag.models import RetrievedChunk


@dataclass
class EvalResult:
    """Per-case outcome. ``None`` means unscored or not applicable, never zero."""

    id: str
    question: str
    answerable: bool
    recall_at_k: float
    reciprocal_rank: float
    ndcg: float
    grounded: bool
    abstained: bool | None
    answer: str
    faithfulness: float | None = None
    correctness: float | None = None
    dangling_citations: list[int] = field(default_factory=list)
    cited_doc_ids: list[str] = field(default_factory=list)
    retrieval_empty: bool = False


def load_golden(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Golden set not found at {path}. Create it, or point FINSIGHT_GOLDEN_FILE "
            "at the right location."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"Golden set at {path} must be a non-empty JSON list.")
    return cast(list[dict[str, Any]], data)


def run_eval(
    settings: Settings, *, agent: Any | None = None, judge: Any | None = None
) -> list[EvalResult]:
    """Run the full agent over every golden case and score the results."""
    golden = load_golden(settings.golden_file)
    active_agent = agent if agent is not None else build_agent(settings)
    active_judge = judge if judge is not None else build_judge(settings)
    k = settings.retrieval_top_k

    results: list[EvalResult] = []
    for case in golden:
        question = str(case["question"])
        # A negative case declares itself unanswerable and carries no expected docs.
        answerable = bool(case.get("answerable", True))
        expected = [str(x) for x in case.get("expected_doc_ids", [])]
        # The golden set stores reference text under "ground_truth".
        reference = str(case.get("ground_truth", "")) if answerable else ""

        final = active_agent.invoke(
            {"question": question, "current_query": question, "attempts": 0}
        )
        retrieved: list[RetrievedChunk] = final.get("retrieved", [])
        answer = str(final.get("answer", ""))
        retrieved_doc_ids = [c.chunk.doc_id for c in retrieved]
        evidence = final.get("context_citations", dict(enumerate(retrieved, start=1)))
        grounding = assess_grounding(answer, evidence)

        results.append(
            EvalResult(
                id=str(case.get("id", question[:32])),
                question=question,
                answerable=answerable,
                recall_at_k=recall_at_k(retrieved_doc_ids, expected, k),
                reciprocal_rank=reciprocal_rank(retrieved_doc_ids, expected),
                ndcg=ndcg_at_k(retrieved_doc_ids, expected, k),
                grounded=grounding.is_grounded,
                abstained=(
                    semantic_abstention(active_judge, question, answer)
                    if not answerable else None
                ),
                answer=answer,
                faithfulness=faithfulness(
                    active_judge, question, answer, list(evidence.values()),
                    context_text=final.get("context_text"),
                ),
                correctness=(
                    answer_matches_reference(active_judge, question, answer, reference)
                    if reference
                    else None
                ),
                dangling_citations=grounding.dangling_citations,
                cited_doc_ids=grounding.cited_doc_ids,
                retrieval_empty=not retrieved,
            )
        )
    return results


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def summarize(results: list[EvalResult], *, judge_is_same_model: bool = False) -> dict[str, Any]:
    """Aggregate per-case results into headline metrics.

    Retrieval metrics are computed over answerable cases only — negative cases have
    no expected documents, so including them would inflate recall.
    Abstention averages use scored negative cases only; the scored and error
    counts disclose coverage so malformed judge verdicts cannot earn credit.
    """
    if not results:
        return {"n": 0}

    answerable = [r for r in results if r.answerable]
    negative = [r for r in results if not r.answerable]
    faithful_scores = [r.faithfulness for r in results if r.faithfulness is not None]
    correctness_scores = [r.correctness for r in results if r.correctness is not None]
    abstention_scores = [r.abstained for r in negative if r.abstained is not None]

    summary: dict[str, Any] = {
        "n": len(results),
        "n_answerable": len(answerable),
        "n_negative": len(negative),
        "recall_at_k": _mean([r.recall_at_k for r in answerable]),
        "mrr": _mean([r.reciprocal_rank for r in answerable]),
        "ndcg_at_k": _mean([r.ndcg for r in answerable]),
        "grounded_rate": _mean([1.0 if r.grounded else 0.0 for r in results]),
        "dangling_citation_cases": sum(1 for r in results if r.dangling_citations),
        "retrieval_failures": sum(1 for r in answerable if r.retrieval_empty),
        "mean_faithfulness": _mean(faithful_scores),
        "n_faithfulness_scored": len(faithful_scores),
        "mean_correctness": _mean(correctness_scores),
        "abstention_rate_on_negative": _mean(
            [1.0 if abstention else 0.0 for abstention in abstention_scores]
        ),
        "n_abstention_scored": len(abstention_scores),
        "n_abstention_errors": len(negative) - len(abstention_scores),
    }
    if judge_is_same_model:
        summary["caveat"] = (
            "judge is the same model as the answering model — faithfulness is "
            "self-graded. Set FINSIGHT_JUDGE_MODEL to a different family."
        )
    return summary
