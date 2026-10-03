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

import hashlib
import json
import platform
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from functools import partial
from importlib.metadata import version
from pathlib import Path
from typing import Any

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
from finsight.guardrails.validation import assess_grounding, validate_query
from finsight.llm import build_judge
from finsight.rag.models import RetrievedChunk


@dataclass
class EvalResult:
    """Per-case outcome. ``None`` means unscored or not applicable, never zero."""

    id: str
    question: str
    answerable: bool
    recall_at_k: float | None
    reciprocal_rank: float | None
    ndcg: float | None
    grounded: bool | None
    abstained: bool | None
    answer: str
    faithfulness: float | None = None
    correctness: float | None = None
    dangling_citations: list[int] = field(default_factory=list)
    cited_doc_ids: list[str] = field(default_factory=list)
    retrieval_empty: bool = False
    agent_succeeded: bool = True
    reference_available: bool = False
    errors: dict[str, str] = field(default_factory=dict)
    operational_errors: list[str] = field(default_factory=list)


def load_golden(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Golden set not found at {path}. Create it, or point FINSIGHT_GOLDEN_FILE "
            "at the right location."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"Golden set at {path} must be a non-empty JSON list.")
    validated: list[dict[str, Any]] = []
    ids: set[str] = set()
    for index, case in enumerate(data, start=1):
        prefix = f"Invalid golden case {index}"
        if not isinstance(case, dict):
            raise ValueError(f"{prefix}: expected an object")
        question = case.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{prefix}: question must be a nonempty string")
        question = validate_query(question)
        case_id = case.get("id", f"case-{index}")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in ids:
            raise ValueError(f"{prefix}: id must be a unique nonempty string")
        answerable = case.get("answerable", True)
        if not isinstance(answerable, bool):
            raise ValueError(f"{prefix}: answerable must be a boolean")
        expected = case.get("expected_doc_ids", [])
        if (
            not isinstance(expected, list)
            or any(not isinstance(doc, str) or not doc.strip() for doc in expected)
            or len(set(expected)) != len(expected)
        ):
            raise ValueError(f"{prefix}: expected_doc_ids must be unique nonempty strings")
        if not answerable and expected:
            raise ValueError(f"{prefix}: negative cases cannot have expected documents")
        reference = case.get("ground_truth", "")
        if not isinstance(reference, str):
            raise ValueError(f"{prefix}: ground_truth must be a string")
        ids.add(case_id)
        validated.append({
            "id": case_id, "question": question, "answerable": answerable,
            "expected_doc_ids": expected, "ground_truth": reference,
        })
    return validated


def run_eval(
    settings: Settings, *, agent: Any | None = None, judge: Any | None = None
) -> list[EvalResult]:
    """Validate every case, then isolate operational failures without losing results."""
    golden = load_golden(settings.golden_file)
    agent_error: str | None = None
    judge_error: str | None = None
    try:
        active_agent = agent if agent is not None else build_agent(settings)
    except Exception as exc:
        active_agent = None
        agent_error = type(exc).__name__
    active_judge: Any
    try:
        active_judge = judge if judge is not None else build_judge(settings)
    except Exception as exc:
        active_judge = None
        judge_error = type(exc).__name__
    k = settings.retrieval_top_k

    results: list[EvalResult] = []
    for case in golden:
        question, answerable = case["question"], case["answerable"]
        expected = case["expected_doc_ids"]
        reference = case["ground_truth"] if answerable else ""
        result = EvalResult(
            id=case["id"], question=question, answerable=answerable,
            recall_at_k=None, reciprocal_rank=None, ndcg=None, grounded=None,
            abstained=None, answer="", agent_succeeded=False,
            reference_available=bool(reference.strip()),
        )
        results.append(result)
        if agent_error:
            result.errors["agent"] = agent_error
            result.operational_errors.append("agent")
            continue
        try:
            final = active_agent.invoke(
                {"question": question, "current_query": question, "attempts": 0}
            )
            retrieved: list[RetrievedChunk] = final.get("retrieved", [])
            answer = final.get("answer", "")
            if not isinstance(answer, str):
                raise TypeError("Agent answer must be text")
            retrieved_doc_ids = [c.chunk.doc_id for c in retrieved]
            evidence = final.get("context_citations", dict(enumerate(retrieved, start=1)))
            grounding = assess_grounding(answer, evidence)
        except Exception as exc:
            result.errors["agent"] = type(exc).__name__
            result.operational_errors.append("agent")
            continue
        result.agent_succeeded = True
        result.answer = answer
        if answerable and expected:
            result.recall_at_k = recall_at_k(retrieved_doc_ids, expected, k)
            result.reciprocal_rank = reciprocal_rank(retrieved_doc_ids, expected)
            result.ndcg = ndcg_at_k(retrieved_doc_ids, expected, k)
        result.grounded = grounding.is_grounded
        result.dangling_citations = grounding.dangling_citations
        result.cited_doc_ids = grounding.cited_doc_ids
        result.retrieval_empty = not retrieved

        if not answerable:
            result.abstained = (
                _score(result, "abstention", judge_error,
                       partial(semantic_abstention, active_judge, question, answer))
                if answer.strip() else False
            )
        context_text = final.get("context_text")
        if answer and evidence and context_text != "":
            result.faithfulness = _score(result, "faithfulness", judge_error, partial(faithfulness,
                active_judge, question, answer, list(evidence.values()), context_text=context_text
            ))
        if reference.strip():
            result.correctness = (
                _score(result, "correctness", judge_error, partial(
                    answer_matches_reference, active_judge, question, answer, reference
                )) if answer.strip() else 0.0
            )
    return results


def _score(
    result: EvalResult, stage: str, judge_error: str | None, operation: Callable[[], Any]
) -> Any:
    if judge_error:
        result.errors[stage] = judge_error
        result.operational_errors.append(stage)
        return None
    try:
        value = operation()
    except Exception as exc:
        result.errors[stage] = type(exc).__name__
        result.operational_errors.append(stage)
        return None
    if value is None:
        result.errors[stage] = "StructuredOutputError"
    return value


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
        "recall_at_k": _mean([r.recall_at_k for r in answerable if r.recall_at_k is not None]),
        "mrr": _mean([r.reciprocal_rank for r in answerable if r.reciprocal_rank is not None]),
        "ndcg_at_k": _mean([r.ndcg for r in answerable if r.ndcg is not None]),
        "grounded_rate": _mean([
            1.0 if r.grounded else 0.0 for r in results if r.grounded is not None
        ]),
        "n_agent_succeeded": sum(r.agent_succeeded for r in results),
        "n_agent_errors": sum(not r.agent_succeeded for r in results),
        "n_operational_failures": sum(bool(r.operational_errors) for r in results),
        "n_retrieval_scored": sum(r.recall_at_k is not None for r in answerable),
        "n_grounding_scored": sum(r.grounded is not None for r in results),
        "dangling_citation_cases": sum(1 for r in results if r.dangling_citations),
        "retrieval_failures": sum(1 for r in answerable if r.retrieval_empty),
        "mean_faithfulness": _mean(faithful_scores),
        "n_faithfulness_scored": len(faithful_scores),
        "mean_correctness": _mean(correctness_scores),
        "n_correctness_scored": len(correctness_scores),
        "n_correctness_eligible": sum(r.reference_available for r in results),
        "n_correctness_errors": sum("correctness" in r.errors for r in results),
        "n_faithfulness_errors": sum("faithfulness" in r.errors for r in results),
        "abstention_rate_on_negative": _mean(
            [1.0 if abstention else 0.0 for abstention in abstention_scores]
        ),
        "n_abstention_scored": len(abstention_scores),
        "n_abstention_errors": sum("abstention" in r.errors for r in negative),
        "n_abstention_unscored": len(negative) - len(abstention_scores),
    }
    if judge_is_same_model:
        summary["caveat"] = (
            "judge is the same model as the answering model — faithfulness is "
            "self-graded. Set FINSIGHT_JUDGE_MODEL to a different family."
        )
    return summary


def evaluation_provenance(settings: Settings) -> dict[str, Any]:
    """Allowlisted reproducibility data; never include credentials or raw endpoints."""
    provider, model, endpoint, _ = settings.resolved_judge()

    def endpoint_hash(value: str | None) -> str | None:
        return hashlib.sha256(value.encode()).hexdigest() if value else None

    golden_sha256: str | None = None
    golden_error: str | None = None
    try:
        golden_sha256 = hashlib.sha256(settings.golden_file.read_bytes()).hexdigest()
    except OSError as exc:
        golden_error = type(exc).__name__

    manifest = settings.index_dir / "manifest.json"
    index: dict[str, Any] = {"available": False}
    if manifest.exists():
        try:
            payload = manifest.read_bytes()
            data = json.loads(payload)
            index = {
                "available": True,
                "manifest_sha256": hashlib.sha256(payload).hexdigest(),
                "generation": data.get("generation"),
                "content_hash": data.get("content_hash"),
            }
        except Exception as exc:
            index = {"available": False, "error": type(exc).__name__}
    return {
        "captured_at": datetime.now(UTC).isoformat(),
        "finsight_version": version("finsight"),
        "python_version": platform.python_version(),
        "golden_sha256": golden_sha256,
        "golden_error": golden_error,
        "models": {
            "answer": {"provider": settings.llm_provider, "model": settings.llm_model,
                       "endpoint_sha256": endpoint_hash(settings.resolved_llm_base_url())},
            "judge": {"provider": provider, "model": model,
                      "endpoint_sha256": endpoint_hash(endpoint)},
            "embedding": {"provider": settings.embed_provider, "model": settings.embed_model,
                          "endpoint_sha256": endpoint_hash(settings.resolved_embed_base_url())},
        },
        "configuration": {name: getattr(settings, name) for name in (
            "llm_temperature", "llm_timeout_s", "llm_max_retries", "embed_timeout_s",
            "embed_max_retries", "chunk_size", "chunk_overlap", "retrieval_top_k",
            "retrieval_candidates", "context_max_chars", "max_retrieval_attempts",
            "enforce_grounding", "use_mcp_tools",
        )},
        "index": index,
    }


def write_evaluation_report(
    path: Path, results: list[EvalResult], summary: dict[str, Any],
    before: dict[str, Any], after: dict[str, Any],
) -> None:
    """Persist completed and failed cases alongside their scoring denominators."""
    report = {
        "report_version": 1,
        "summary": summary,
        "results": [asdict(result) for result in results],
        "provenance": {"before": before, "after": after,
                       "index_changed": before["index"] != after["index"],
                       "golden_changed": before["golden_sha256"] != after["golden_sha256"]},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
