"""Evaluation failures remain visible without discarding successful cases."""
import json
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from finsight import cli
from finsight.config import Settings
from finsight.eval import harness


class Agent:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def invoke(self, state):
        self.calls += 1
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


class Judge:
    def __init__(self, responses):
        self.responses = iter(responses)

    def with_structured_output(self, schema):
        raise NotImplementedError

    def invoke(self, prompt):
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(content=response)


def settings_for(tmp_path, cases):
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps(cases))
    return Settings(golden_file=golden, index_dir=tmp_path / "index")


@pytest.mark.parametrize("bad_case", [
    1, {}, {"question": ""}, {"question": 2}, {"question": "x" * 2001},
    {"question": "q", "answerable": "false"},
    {"question": "q", "expected_doc_ids": "a"},
    {"question": "q", "expected_doc_ids": ["a", "a"]},
    {"question": "q", "expected_doc_ids": [[]]},
    {"question": "q", "ground_truth": 3},
    {"question": "q", "answerable": False, "expected_doc_ids": ["a"]},
])
def test_entire_golden_set_validated_before_model_work(tmp_path, bad_case):
    settings = settings_for(tmp_path, [{"question": "valid"}, bad_case])
    agent = Agent([])
    with pytest.raises(ValueError):
        harness.run_eval(settings, agent=agent, judge=Judge([]))
    assert agent.calls == 0


def test_duplicate_ids_rejected(tmp_path):
    settings = settings_for(tmp_path, [{"id": "x", "question": "q"}] * 2)
    with pytest.raises(ValueError, match="unique"):
        harness.load_golden(settings.golden_file)


def test_agent_failure_does_not_discard_later_cases_or_inflate_scores(tmp_path):
    settings = settings_for(tmp_path, [
        {"question": "one", "expected_doc_ids": ["a"]},
        {"question": "two", "expected_doc_ids": ["a"]},
    ])
    agent = Agent([RuntimeError("secret-api-key"), {"answer": "I do not know"}])
    results = harness.run_eval(settings, agent=agent, judge=Judge([]))
    assert results[0].errors == {"agent": "RuntimeError"}
    assert results[0].grounded is None
    assert results[0].recall_at_k is None
    assert results[1].recall_at_k == 0
    summary = harness.summarize(results)
    assert summary["n_agent_errors"] == 1
    assert summary["n_operational_failures"] == 1
    assert summary["n_retrieval_scored"] == 1
    assert summary["recall_at_k"] == 0
    assert summary["n_grounding_scored"] == 1


def test_judge_failure_isolated_and_malformed_verdict_explicit(tmp_path):
    settings = settings_for(tmp_path, [
        {"question": "one", "answerable": False},
        {"question": "two", "answerable": False},
        {"question": "three", "answerable": False},
    ])
    agent = Agent([{"answer": "I cannot answer"}] * 3)
    judge = Judge([
        TimeoutError("secret-api-key"), "not-json", "still-not-json",
        '{"declines_to_answer": true, "provides_answer": false}',
    ])
    results = harness.run_eval(settings, agent=agent, judge=judge)
    assert [r.abstained for r in results] == [None, None, True]
    assert results[0].errors == {"abstention": "TimeoutError"}
    assert results[1].errors == {"abstention": "StructuredOutputError"}
    assert results[1].operational_errors == []
    summary = harness.summarize(results)
    assert summary["n_operational_failures"] == 1
    assert summary["n_abstention_scored"] == 1
    assert summary["n_abstention_errors"] == 2
    assert summary["n_abstention_unscored"] == 2


def test_initialization_failures_recorded_for_every_case(tmp_path, monkeypatch):
    settings = settings_for(tmp_path, [{"question": "q"}, {"question": "r"}])

    def fail(settings):
        raise ConnectionError("secret-api-key")

    monkeypatch.setattr(harness, "build_agent", fail)
    results = harness.run_eval(settings, judge=Judge([]))
    assert len(results) == 2
    assert all(r.errors == {"agent": "ConnectionError"} for r in results)


def test_missing_retrieval_labels_are_unscored(tmp_path):
    settings = settings_for(tmp_path, [{"question": "q"}])
    results = harness.run_eval(settings, agent=Agent([{"answer": ""}]), judge=Judge([]))
    summary = harness.summarize(results)
    assert summary["recall_at_k"] is None
    assert summary["n_retrieval_scored"] == 0


def test_cli_writes_failure_report_then_exits_nonzero(tmp_path, monkeypatch):
    settings = settings_for(tmp_path, [{"question": "q", "answerable": False}])
    settings.llm_api_key = "secret-api-key"
    settings.llm_base_url = "https://user:secret-api-key@example.com/v1?token=secret"
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(harness, "build_agent", lambda _: Agent([TimeoutError("secret-api-key")]))
    monkeypatch.setattr(harness, "build_judge", lambda _: Judge([]))
    output = tmp_path / "reports" / "eval.json"
    command = CliRunner().invoke(cli.app, ["evaluate", "--output", str(output)])
    assert command.exit_code == 1, command.output
    assert "Evaluation report saved:" in command.output
    report = json.loads(output.read_text())
    assert report["report_version"] == 1
    assert report["summary"]["n_operational_failures"] == 1
    assert report["results"][0]["errors"] == {"agent": "TimeoutError"}
    assert "secret-api-key" not in output.read_text() + command.output
    assert "example.com" not in output.read_text()
    assert report["provenance"]["before"]["golden_sha256"]
    assert report["provenance"]["index_changed"] is False
    assert report["provenance"]["golden_changed"] is False


def test_cli_success_report_and_index_change_provenance(tmp_path, monkeypatch):
    settings = settings_for(tmp_path, [{"question": "q", "answerable": False}])
    settings.index_dir.mkdir()
    manifest = settings.index_dir / "manifest.json"
    manifest.write_text('{"generation":"before","content_hash":"hash"}')

    class ChangingAgent:
        def invoke(self, state):
            manifest.write_text('{"generation":"after","content_hash":"hash2"}')
            settings.golden_file.write_text('[{"question":"changed", "answerable":false}]')
            return {"answer": ""}

    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(harness, "build_agent", lambda _: ChangingAgent())
    monkeypatch.setattr(harness, "build_judge", lambda _: Judge([]))
    output = tmp_path / "eval.json"
    command = CliRunner().invoke(cli.app, ["evaluate", "--output", str(output)])
    assert command.exit_code == 0, command.output
    report = json.loads(output.read_text())
    assert report["provenance"]["index_changed"] is True
    assert report["provenance"]["golden_changed"] is True
    assert report["provenance"]["before"]["index"]["generation"] == "before"
    assert report["provenance"]["after"]["index"]["generation"] == "after"
    assert report["summary"]["n_abstention_scored"] == 1
    assert report["results"][0]["abstained"] is False


def test_empty_answer_fails_correctness_without_a_judge_call(tmp_path):
    settings = settings_for(tmp_path, [{"question": "q", "ground_truth": "reference"}])
    results = harness.run_eval(settings, agent=Agent([{"answer": " "}]), judge=Judge([]))
    summary = harness.summarize(results)
    assert results[0].correctness == 0.0
    assert summary["n_correctness_eligible"] == 1
    assert summary["n_correctness_scored"] == 1
    assert summary["mean_correctness"] == 0.0


def test_judge_initialization_failure_preserves_agent_results(tmp_path, monkeypatch):
    settings = settings_for(tmp_path, [{"question": "q", "answerable": False}])

    def fail(settings):
        raise OSError("secret-api-key")

    monkeypatch.setattr(harness, "build_judge", fail)
    results = harness.run_eval(settings, agent=Agent([{"answer": "I cannot answer"}]))
    assert results[0].agent_succeeded
    assert results[0].answer == "I cannot answer"
    assert results[0].errors == {"abstention": "OSError"}
    assert results[0].operational_errors == ["abstention"]


def test_cli_preserves_completed_results_if_golden_file_disappears(tmp_path, monkeypatch):
    settings = settings_for(tmp_path, [{"question": "q", "answerable": False}])

    class DeletingAgent:
        def invoke(self, state):
            settings.golden_file.unlink()
            return {"answer": ""}

    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(harness, "build_agent", lambda _: DeletingAgent())
    monkeypatch.setattr(harness, "build_judge", lambda _: Judge([]))
    output = tmp_path / "eval.json"
    command = CliRunner().invoke(cli.app, ["evaluate", "--output", str(output)])
    assert command.exit_code == 0, command.output
    report = json.loads(output.read_text())
    assert report["results"][0]["agent_succeeded"] is True
    assert report["results"][0]["question"] == "q"
    assert report["summary"]["n_abstention_scored"] == 1
    assert report["provenance"]["before"]["golden_sha256"] is not None
    assert report["provenance"]["after"]["golden_sha256"] is None
    assert report["provenance"]["after"]["golden_error"] == "FileNotFoundError"
    assert report["provenance"]["golden_changed"] is True
