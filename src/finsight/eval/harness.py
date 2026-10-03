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
import math
import os
import platform
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from functools import partial
from importlib.metadata import version
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

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

EVALUATION_CONFIG_FIELDS = (
    "llm_temperature", "llm_timeout_s", "llm_max_retries", "embed_timeout_s",
    "embed_max_retries", "chunk_size", "chunk_overlap", "chunk_max_tokens", "retrieval_top_k",
    "retrieval_candidates", "collection_name", "context_max_chars", "max_retrieval_attempts",
    "enforce_grounding", "use_mcp_tools", "run_timeout_s", "run_max_model_calls",
    "run_max_input_tokens", "run_max_output_tokens", "llm_max_output_tokens",
    "run_max_cost_usd", "input_cost_per_million", "output_cost_per_million",
    "rerank_enabled", "semantic_verification", "context_max_tokens", "semantic_max_segments",
)


class EvaluationCheckpointError(ValueError):
    """A fixed, safe explanation for refusing an incompatible resume."""


class EvalSplit(StrEnum):
    development = "development"
    held_out = "held_out"


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
    category: str = "general"
    split: str = "development"
    runtime: dict[str, Any] | None = None


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
        category = case.get("category", "general")
        if not isinstance(category, str) or not category.strip() or len(category) > 64:
            raise ValueError(f"{prefix}: category must be a nonempty string (max 64 characters)")
        split = case.get("split", "development")
        if split not in ("development", "held_out"):
            raise ValueError(f"{prefix}: split must be development or held_out")
        ids.add(case_id)
        validated.append({
            "id": case_id, "question": question, "answerable": answerable,
            "expected_doc_ids": expected, "ground_truth": reference,
            "category": category.strip(), "split": split,
        })
    return validated


def select_cases(
    cases: list[dict[str, Any]], *, categories: list[str] | None = None, split: str | None = None,
) -> list[dict[str, Any]]:
    """Select validated cases; reject typos rather than run an empty evaluation."""
    if split is not None and split not in ("development", "held_out"):
        raise ValueError("split must be development or held_out")
    selected = [case for case in cases if
                (not categories or case["category"] in categories)
                and (split is None or case["split"] == split)]
    if not selected or (categories and set(categories) - {case["category"] for case in selected}):
        raise ValueError("Evaluation selection contains no matching cases or unknown categories")
    return selected


def run_eval(
    settings: Settings, *, agent: Any | None = None, judge: Any | None = None,
    output: Path | None = None, resume: Path | None = None, retry_failures: bool = False,
    categories: list[str] | None = None, split: str | None = None,
) -> list[EvalResult]:
    """Evaluate selected cases with atomic, fingerprint-checked optional checkpoints."""
    before = evaluation_provenance(settings)
    golden = select_cases(load_golden(settings.golden_file), categories=categories, split=split)
    if before["golden_sha256"] != evaluation_provenance(settings)["golden_sha256"]:
        raise ValueError("Golden set changed during validation")
    if retry_failures and resume is None:
        raise ValueError("retry_failures requires a resume report")
    destination = output or resume
    if destination is not None and destination.resolve() in {
        settings.golden_file.resolve(), (settings.index_dir / "manifest.json").resolve(),
    }:
        raise ValueError("Report destination cannot overwrite evaluation inputs")
    selection = {"categories": sorted(set(categories or [])), "split": split,
                 "case_ids": [case["id"] for case in golden]}
    previous = _load_checkpoint(resume, before, selection, golden) if resume else []
    retained = {r.id: r for r in previous if not (retry_failures and r.errors)}
    # Keep all retained results in the initial checkpoint, including successes
    # after a failed case that is about to be retried.
    results = [retained[case["id"]] for case in golden if case["id"] in retained]

    def checkpoint(complete: bool) -> None:
        if destination is not None:
            write_evaluation_report(
                destination, results,
                summarize(results, judge_is_same_model=settings.judge_is_same_model,
                          selected_cases=golden),
                before, evaluation_provenance(settings),
                checkpoint={"version": 1, "selection": selection, "complete": complete},
            )

    checkpoint(len(results) == len(golden))
    if len(results) == len(golden):
        return results
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

    for case in golden:
        if case["id"] in retained:
            continue
        result = _evaluate_case(case, settings.retrieval_top_k, active_agent, active_judge,
                                agent_error, judge_error)
        retained[result.id] = result
        results = [retained[item["id"]] for item in golden if item["id"] in retained]
        checkpoint(len(results) == len(golden))
    return results


def _evaluate_case(
    case: dict[str, Any], k: int, active_agent: Any, active_judge: Any,
    agent_error: str | None, judge_error: str | None,
) -> EvalResult:
    question, answerable = case["question"], case["answerable"]
    expected = case["expected_doc_ids"]
    reference = case["ground_truth"] if answerable else ""
    result = EvalResult(
        id=case["id"], question=question, answerable=answerable,
        recall_at_k=None, reciprocal_rank=None, ndcg=None, grounded=None,
        abstained=None, answer="", agent_succeeded=False,
        reference_available=bool(reference.strip()),
        category=case["category"], split=case["split"],
    )
    if agent_error:
        result.errors["agent"] = agent_error
        result.operational_errors.append("agent")
        return result
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
        result.runtime = sanitize_runtime(
            getattr(exc, "summary", None) or getattr(exc, "runtime_summary", None)
        )
        result.errors["agent"] = type(exc).__name__
        result.operational_errors.append("agent")
        return result
    result.runtime = sanitize_runtime(final.get("runtime"))
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
    return result



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


def summarize(
    results: list[EvalResult], *, judge_is_same_model: bool = False,
    selected_cases: list[dict[str, Any]] | None = None, _grouped: bool = True,
) -> dict[str, Any]:
    """Aggregate per-case results into headline metrics.

    Retrieval metrics are computed over answerable cases only — negative cases have
    no expected documents, so including them would inflate recall.
    Abstention averages use scored negative cases only; the scored and error
    counts disclose coverage so malformed judge verdicts cannot earn credit.
    """
    if not results and selected_cases is None:
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
    if _grouped:
        selected_count = len(selected_cases) if selected_cases is not None else len(results)
        summary["n_selected"] = selected_count
        summary["n_completed"] = len(results)
        summary["completion_rate"] = (
            round(len(results) / selected_count, 3) if selected_count else None
        )
        names = sorted({case["category"] for case in selected_cases} if selected_cases is not None
                       else {result.category for result in results})
        summary["by_category"] = {}
        for name in names:
            group = [result for result in results if result.category == name]
            eligible = (sum(case["category"] == name for case in selected_cases)
                        if selected_cases is not None else len(group))
            metrics = summarize(group, selected_cases=[], _grouped=False)
            metrics.update(n_selected=eligible, n_completed=len(group),
                           completion_rate=round(len(group) / eligible, 3) if eligible else None)
            summary["by_category"][name] = metrics
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
                          "revision": getattr(settings, "embed_revision", None),
                          "endpoint_sha256": endpoint_hash(settings.resolved_embed_base_url())},
        },
        "configuration": {name: getattr(settings, name, None)
                          for name in EVALUATION_CONFIG_FIELDS},
        "index": index,
    }


def write_evaluation_report(
    path: Path, results: list[EvalResult], summary: dict[str, Any],
    before: dict[str, Any], after: dict[str, Any],
    *, checkpoint: dict[str, Any] | None = None,
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
    if checkpoint is not None:
        report["checkpoint"] = checkpoint
    path.parent.mkdir(parents=True, exist_ok=True)
    # A killed process leaves either the previous checkpoint or the next complete
    # one; it cannot replace the destination with a half-written JSON document.
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as stream:
            temporary = stream.name
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def _fingerprint(provenance: dict[str, Any]) -> str:
    stable = {key: provenance[key] for key in (
        "golden_sha256", "models", "configuration", "index", "finsight_version", "python_version"
    )}
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()


def _load_checkpoint(
    path: Path, current: dict[str, Any], selection: dict[str, Any], cases: list[dict[str, Any]],
) -> list[EvalResult]:
    """Fail closed before model construction if results no longer describe this run."""
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        checkpoint = report["checkpoint"]
        if report["report_version"] != 1 or checkpoint["version"] != 1:
            raise EvaluationCheckpointError("Unsupported evaluation checkpoint format.")
        if checkpoint["selection"] != selection:
            raise EvaluationCheckpointError("Resume refused: category/split selection differs.")
        if any(_fingerprint(report["provenance"][phase]) != _fingerprint(current)
               for phase in ("before", "after")):
            raise EvaluationCheckpointError(
                "Resume refused: golden, model, configuration, index, or runtime version changed."
            )
        loaded = TypeAdapter(list[EvalResult]).validate_python(report["results"])
        expected = {case["id"]: case for case in cases}
        seen: set[str] = set()
        for result in loaded:
            if result.id in seen or result.id not in expected:
                raise EvaluationCheckpointError("Checkpoint has duplicate or unexpected cases.")
            case = expected[result.id]
            if any(getattr(result, key) != case[key]
                   for key in ("question", "answerable", "category", "split")):
                raise EvaluationCheckpointError("Checkpoint case metadata differs from golden set.")
            result.runtime = sanitize_runtime(result.runtime)
            seen.add(result.id)
        return loaded
    except EvaluationCheckpointError:
        raise
    except (ValueError, KeyError, TypeError, OSError) as exc:
        raise EvaluationCheckpointError("Cannot read a valid evaluation checkpoint.") from exc


def sanitize_runtime(value: Any) -> dict[str, Any] | None:
    """Persist only bounded numeric telemetry, never provider metadata or inputs."""
    if not isinstance(value, dict):
        return None
    numeric = {"elapsed_ms", "model_calls", "input_budget_tokens", "output_budget_tokens",
               "reported_input_tokens", "reported_output_tokens", "reported_usage_calls",
               "estimated_cost_usd", "budget_cost_usd"}
    result: dict[str, Any] = {}
    for key in numeric:
        number = value.get(key)
        if isinstance(number, (int, float)) and not isinstance(number, bool):
            if math.isfinite(number) and number >= 0:
                result[key] = number
    for key in ("estimated_cost_usd", "budget_cost_usd"):
        if value.get(key) is None:
            result[key] = None
    statuses = {"completed", "error", "cancelled", "deadline_exceeded", "ok",
                "model_call_budget_exceeded", "input_token_budget_exceeded",
                "output_token_budget_exceeded", "cost_budget_exceeded"}
    for key, allowed in {
        "status": statuses,
        "token_count_basis": {
            "injected_content_counter", "utf8_content_bytes_excludes_provider_overhead"
        },
    }.items():
        if isinstance(value.get(key), str) and value[key] in allowed:
            result[key] = value[key]
    node_names = {"retrieve", "rerank", "verify", "reformulate", "answer", "grade", "finalize"}
    for key in ("nodes", "model_events"):
        events = value.get(key)
        if not isinstance(events, list):
            continue
        clean = []
        for event in events[:1000]:
            if not isinstance(event, dict):
                continue
            elapsed = event.get("elapsed_ms")
            status = event.get("status")
            if (not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool)
                    or not math.isfinite(elapsed) or elapsed < 0
                    or not isinstance(status, str) or status not in statuses):
                continue
            item = {"elapsed_ms": elapsed, "status": status}
            if key == "nodes":
                name = event.get("name")
                if not isinstance(name, str) or name not in node_names:
                    continue
                item["name"] = name
            clean.append(item)
        result[key] = clean
    return result
