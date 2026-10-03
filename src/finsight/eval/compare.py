"""Descriptive comparisons of reports evaluated against the same evidence and cases."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from finsight.eval.harness import EVALUATION_CONFIG_FIELDS, EvalResult, summarize


class ReportComparisonError(ValueError):
    """Safe, fixed explanation of why reports cannot be compared."""


_METRICS = (
    "recall_at_k", "mrr", "ndcg_at_k", "grounded_rate", "mean_faithfulness",
    "mean_correctness", "abstention_rate_on_negative",
)
_COUNTS = (
    "n", "n_answerable", "n_negative", "n_agent_succeeded", "n_agent_errors",
    "n_operational_failures", "n_retrieval_scored", "n_grounding_scored",
    "n_faithfulness_scored", "n_faithfulness_errors", "n_correctness_scored",
    "n_correctness_eligible", "n_correctness_errors", "n_abstention_scored",
    "n_abstention_errors", "n_abstention_unscored",
)
_METRIC_FIELDS = {
    "recall_at_k": "recall_at_k", "mrr": "reciprocal_rank", "ndcg_at_k": "ndcg",
    "grounded_rate": "grounded", "mean_faithfulness": "faithfulness",
    "mean_correctness": "correctness", "abstention_rate_on_negative": "abstained",
}


def _scored_cases(results: list[EvalResult], metric: str) -> dict[str, float]:
    scores = {}
    for result in results:
        if metric in {"recall_at_k", "mrr", "ndcg_at_k"} and not result.answerable:
            continue
        if metric == "abstention_rate_on_negative" and result.answerable:
            continue
        value = getattr(result, _METRIC_FIELDS[metric])
        if value is not None:
            scores[result.id] = float(value)
    return scores


def _load(path: Path) -> tuple[dict[str, Any], list[EvalResult]]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["report_version"] != 1 or report["checkpoint"]["version"] != 1:
            raise ReportComparisonError("Comparison requires versioned evaluation checkpoints.")
        selected = report["checkpoint"]["selection"]["case_ids"]
        if (not isinstance(selected, list) or not selected
                or any(not isinstance(case_id, str) for case_id in selected)
                or len(set(selected)) != len(selected)):
            raise ReportComparisonError("Report has invalid selected case IDs.")
        results = TypeAdapter(list[EvalResult]).validate_python(report["results"])
        ids = [result.id for result in results]
        if len(ids) != len(set(ids)) or set(ids) - set(selected):
            raise ReportComparisonError("Report has duplicate or unexpected results.")
        return report, results
    except ReportComparisonError:
        raise
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise ReportComparisonError("Cannot read valid evaluation reports.") from exc


def _identity(report: dict[str, Any]) -> tuple[str, str, set[str]]:
    try:
        before, after = (report["provenance"][phase] for phase in ("before", "after"))
        golden = before["golden_sha256"]
        content = before["index"].get("content_hash")
        if not isinstance(golden, str) or not golden or not isinstance(content, str) or not content:
            raise ReportComparisonError("Reports require known golden and index content hashes.")
        if golden != after["golden_sha256"] or content != after["index"].get("content_hash"):
            raise ReportComparisonError("Evidence changed during evaluation; cannot compare.")
        if any(before[key] != after[key] for key in ("models", "configuration")):
            raise ReportComparisonError("Model or configuration changed during evaluation.")
        return golden, content, set(report["checkpoint"]["selection"]["case_ids"])
    except (KeyError, TypeError, AttributeError) as exc:
        raise ReportComparisonError("Reports are missing comparison provenance.") from exc


def _cohort(
    baseline: list[EvalResult], candidate: list[EvalResult], selected: int,
) -> dict[str, Any]:
    summaries = [summarize(results) for results in (baseline, candidate)]
    views: list[dict[str, Any]] = []
    for results, summary in zip((baseline, candidate), summaries, strict=True):
        view: dict[str, Any] = {key: summary.get(key) for key in _METRICS}
        view.update({key: summary.get(key, 0) for key in _COUNTS})
        view.update(n_selected=selected, n_completed=len(results),
                    completion_rate=round(len(results) / selected, 3) if selected else None)
        views.append(view)
    paired = {}
    for key in _METRICS:
        left, right = (_scored_cases(results, key) for results in (baseline, candidate))
        ids = sorted(left.keys() & right.keys())
        left_mean = sum(left[case_id] for case_id in ids) / len(ids) if ids else None
        right_mean = sum(right[case_id] for case_id in ids) / len(ids) if ids else None
        paired[key] = {
            "paired_count": len(ids), "case_ids": ids,
            "baseline": round(left_mean, 3) if left_mean is not None else None,
            "candidate": round(right_mean, 3) if right_mean is not None else None,
            "delta": round(right_mean - left_mean, 3)
            if left_mean is not None and right_mean is not None else None,
        }
    return {"baseline": views[0], "candidate": views[1], "paired": paired,
            "deltas": {key: paired[key]["delta"] for key in _METRICS}}


def compare_reports(baseline_path: Path, candidate_path: Path) -> dict[str, Any]:
    """Compare compatible reports without asserting a quality or significance threshold."""
    baseline, left = _load(baseline_path)
    candidate, right = _load(candidate_path)
    identity = _identity(baseline)
    if identity != _identity(candidate):
        raise ReportComparisonError("Golden, index content, or selected case IDs differ.")
    left_cases = {result.id: result for result in left}
    for result in right:
        previous = left_cases.get(result.id)
        if previous and (previous.category, previous.split) != (result.category, result.split):
            raise ReportComparisonError("Case category or split differs between reports.")
    try:
        names = sorted(set(baseline["summary"]["by_category"])
                       | set(candidate["summary"]["by_category"]))
        by_category = {}
        if any(result.category not in names for result in left + right):
            raise ReportComparisonError("Result category is absent from selected coverage.")
        category_totals = 0
        for name in names:
            counts = [report["summary"]["by_category"][name]["n_selected"]
                      for report in (baseline, candidate)]
            if (any(type(count) is not int or count < 1 for count in counts)
                    or counts[0] != counts[1]):
                raise ReportComparisonError("Selected category coverage differs between reports.")
            if any(sum(result.category == name for result in group) > counts[0]
                   for group in (left, right)):
                raise ReportComparisonError("Category results exceed selected coverage.")
            category_totals += counts[0]
            by_category[name] = _cohort(
                [r for r in left if r.category == name],
                [r for r in right if r.category == name], counts[0],
            )
        if category_totals != len(identity[2]):
            raise ReportComparisonError("Report category counts do not match selected cases.")
        model_changes = {}
        for role in ("answer", "judge", "embedding"):
            values = [{key: report["provenance"]["before"]["models"][role].get(key)
                       for key in ("provider", "model", "endpoint_sha256", "revision")}
                      for report in (baseline, candidate)]
            if values[0] != values[1]:
                model_changes[role] = {"baseline": values[0], "candidate": values[1]}
        configs = [report["provenance"]["before"]["configuration"]
                   for report in (baseline, candidate)]
        configuration_changes = {
            key: {"baseline": configs[0].get(key), "candidate": configs[1].get(key)}
            for key in EVALUATION_CONFIG_FIELDS
            if configs[0].get(key) != configs[1].get(key)
        }
    except (KeyError, TypeError) as exc:
        raise ReportComparisonError("Report coverage/configuration metadata is invalid.") from exc
    return {
        "comparison_version": 1,
        "golden_sha256": identity[0], "index_content_hash": identity[1],
        "overall": _cohort(left, right, len(identity[2])),
        "by_category": by_category,
        "model_changes": model_changes,
        "configuration_changes": configuration_changes,
        "caveat": "Deltas use only case IDs scored in both reports for each metric. "
                  "Independent aggregates retain all scored cases; inspect coverage and errors. "
                  "No statistical significance or release-quality threshold is inferred.",
    }
