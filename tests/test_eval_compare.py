"""Only compare reports with matching evidence and cohorts; always expose coverage."""
import json

import pytest
from test_eval_resume import Agent, Judge, fixture
from typer.testing import CliRunner

from finsight import cli
from finsight.eval import harness
from finsight.eval.compare import ReportComparisonError, compare_reports


def reports(tmp_path):
    settings, baseline = fixture(tmp_path)
    settings.index_dir.mkdir()
    (settings.index_dir / "manifest.json").write_text('{"content_hash":"same-content"}')
    harness.run_eval(settings, output=baseline, agent=Agent([{"answer": ""}] * 3), judge=Judge())
    candidate = tmp_path / "candidate.json"
    candidate.write_bytes(baseline.read_bytes())
    return baseline, candidate


def test_comparison_reports_category_deltas_errors_and_model_changes(tmp_path):
    baseline, candidate = reports(tmp_path)
    payload = json.loads(candidate.read_text())
    payload["results"][0]["abstained"] = True
    payload["results"][1]["abstained"] = None
    payload["results"][1]["errors"] = {"abstention": "TimeoutError"}
    payload["results"][1]["operational_errors"] = ["abstention"]
    for phase in ("before", "after"):
        payload["provenance"][phase]["models"]["answer"]["model"] = "candidate-model"
        payload["provenance"][phase]["configuration"]["retrieval_top_k"] = 7
    candidate.write_text(json.dumps(payload))
    comparison = compare_reports(baseline, candidate)
    negative = comparison["by_category"]["missing"]
    assert negative["deltas"]["abstention_rate_on_negative"] == 1.0
    assert negative["baseline"]["n_abstention_scored"] == 2
    assert negative["candidate"]["n_abstention_scored"] == 1
    assert negative["candidate"]["n_abstention_errors"] == 1
    assert comparison["overall"]["candidate"]["n_operational_failures"] == 1
    assert comparison["model_changes"]["answer"]["candidate"]["model"] == "candidate-model"
    assert comparison["configuration_changes"]["retrieval_top_k"]["candidate"] == 7
    assert "No statistical significance" in comparison["caveat"]


@pytest.mark.parametrize("change", ["golden", "index", "cases", "unstable", "missing_hash"])
def test_comparison_refuses_incompatible_reports(tmp_path, change):
    baseline, candidate = reports(tmp_path)
    payload = json.loads(candidate.read_text())
    if change == "golden":
        for phase in ("before", "after"):
            payload["provenance"][phase]["golden_sha256"] = "different"
    elif change == "index":
        for phase in ("before", "after"):
            payload["provenance"][phase]["index"]["content_hash"] = "different"
    elif change == "cases":
        payload["checkpoint"]["selection"]["case_ids"].append("extra")
    elif change == "unstable":
        payload["provenance"]["after"]["golden_sha256"] = "different"
    else:
        payload["provenance"]["before"]["index"].pop("content_hash")
    candidate.write_text(json.dumps(payload))
    with pytest.raises(ReportComparisonError):
        compare_reports(baseline, candidate)


def test_comparison_exposes_incomplete_category_coverage(tmp_path):
    baseline, candidate = reports(tmp_path)
    payload = json.loads(candidate.read_text())
    payload["results"] = payload["results"][:1]
    payload["checkpoint"]["complete"] = False
    candidate.write_text(json.dumps(payload))
    comparison = compare_reports(baseline, candidate)
    assert comparison["overall"]["candidate"]["n_selected"] == 3
    assert comparison["overall"]["candidate"]["n_completed"] == 1
    assert comparison["by_category"]["missing"]["candidate"]["completion_rate"] == .5
    assert comparison["by_category"]["lookup"]["candidate"]["n_completed"] == 0
    assert comparison["by_category"]["lookup"]["deltas"]["grounded_rate"] is None


def test_compare_cli_reports_deltas_and_refuses_unrelated_files(tmp_path):
    baseline, candidate = reports(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli.app, ["compare", str(baseline), str(candidate)])
    assert result.exit_code == 0, result.output
    assert '"deltas"' in result.output
    candidate.write_text("[]")
    result = runner.invoke(cli.app, ["compare", str(baseline), str(candidate)])
    assert result.exit_code == 1
    assert "valid evaluation reports" in result.output


@pytest.mark.parametrize("unscored", [False, True])
def test_deltas_require_matching_scored_cases(tmp_path, unscored):
    baseline, candidate = reports(tmp_path)
    left = json.loads(baseline.read_text())
    right = json.loads(candidate.read_text())
    right["results"][1]["abstained"] = True
    if unscored:
        left["results"][1]["abstained"] = None
        left["results"][1]["errors"] = {"abstention": "StructuredOutputError"}
        right["results"][0]["abstained"] = None
        right["results"][0]["errors"] = {"abstention": "StructuredOutputError"}
    else:
        left["results"] = left["results"][:1]
        right["results"] = right["results"][1:2]
    baseline.write_text(json.dumps(left))
    candidate.write_text(json.dumps(right))
    comparison = compare_reports(baseline, candidate)
    for cohort in (comparison["overall"], comparison["by_category"]["missing"]):
        assert cohort["baseline"]["abstention_rate_on_negative"] == 0.0
        assert cohort["candidate"]["abstention_rate_on_negative"] == 1.0
        assert cohort["deltas"]["abstention_rate_on_negative"] is None
        assert cohort["paired"]["abstention_rate_on_negative"] == {
            "paired_count": 0, "case_ids": [], "baseline": None, "candidate": None, "delta": None,
        }


def test_deltas_use_paired_means_while_retaining_independent_aggregates(tmp_path):
    baseline, candidate = reports(tmp_path)
    left = json.loads(baseline.read_text())
    right = json.loads(candidate.read_text())
    left["results"][1]["abstained"] = True
    right["results"][0]["abstained"] = True
    right["results"][1]["abstained"] = None
    baseline.write_text(json.dumps(left))
    candidate.write_text(json.dumps(right))
    cohort = compare_reports(baseline, candidate)["by_category"]["missing"]
    assert cohort["baseline"]["abstention_rate_on_negative"] == .5
    assert cohort["candidate"]["abstention_rate_on_negative"] == 1.0
    assert cohort["deltas"]["abstention_rate_on_negative"] == 1.0
    assert cohort["paired"]["abstention_rate_on_negative"] == {
        "paired_count": 1, "case_ids": ["a"], "baseline": 0.0, "candidate": 1.0, "delta": 1.0,
    }
