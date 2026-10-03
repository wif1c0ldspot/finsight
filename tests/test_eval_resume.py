"""Atomic checkpoints, strict resume identity, and cohort selection."""
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
        self.queries = []

    def invoke(self, state):
        self.queries.append(state["question"])
        value = next(self.responses)
        if isinstance(value, BaseException):
            raise value
        return value


class Judge:
    def with_structured_output(self, schema):
        raise NotImplementedError

    def invoke(self, prompt):
        return SimpleNamespace(content="malformed")


def fixture(tmp_path):
    cases = [
        {"id": "a", "question": "first", "answerable": False,
         "category": "missing", "split": "development"},
        {"id": "b", "question": "second", "answerable": False,
         "category": "missing", "split": "held_out"},
        {"id": "c", "question": "third", "category": "lookup", "split": "held_out"},
    ]
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps(cases))
    return Settings(golden_file=golden, index_dir=tmp_path / "index"), tmp_path / "report.json"


@pytest.mark.parametrize("bad", [
    {"category": ""}, {"category": 3}, {"category": "x" * 65},
    {"split": "test"}, {"split": False},
])
def test_category_and_split_validate_before_agent_work(tmp_path, bad):
    settings, _ = fixture(tmp_path)
    settings.golden_file.write_text(json.dumps([{"question": "q", **bad}]))
    agent = Agent([])
    with pytest.raises(ValueError):
        harness.run_eval(settings, agent=agent, judge=Judge())
    assert agent.queries == []


def test_filtered_run_reports_selected_category_coverage(tmp_path):
    settings, output = fixture(tmp_path)
    agent = Agent([{"answer": ""}])
    results = harness.run_eval(settings, output=output, agent=agent, judge=Judge(),
                               categories=["missing"], split="held_out")
    assert [r.id for r in results] == ["b"]
    assert agent.queries == ["second"]
    report = json.loads(output.read_text())
    assert report["checkpoint"]["complete"] is True
    assert report["checkpoint"]["selection"]["case_ids"] == ["b"]
    category = report["summary"]["by_category"]["missing"]
    assert category["n_selected"] == 1
    assert category["n_abstention_scored"] == 1
    assert category["completion_rate"] == 1


def test_keyboard_interrupt_retains_completed_cases_and_resume_skips_them(tmp_path):
    settings, output = fixture(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        harness.run_eval(settings, output=output, judge=Judge(),
                         agent=Agent([{"answer": ""}, KeyboardInterrupt()]))
    report = json.loads(output.read_text())
    assert [r["id"] for r in report["results"]] == ["a"]
    assert report["checkpoint"]["complete"] is False
    assert report["summary"]["n_selected"] == 3
    assert report["summary"]["n_completed"] == 1
    assert report["summary"]["by_category"]["missing"]["completion_rate"] == .5
    assert report["summary"]["by_category"]["lookup"]["completion_rate"] == 0
    resumed = Agent([{"answer": ""}, {"answer": ""}])
    results = harness.run_eval(settings, resume=output, judge=Judge(), agent=resumed)
    assert resumed.queries == ["second", "third"]
    assert [r.id for r in results] == ["a", "b", "c"]
    assert json.loads(output.read_text())["checkpoint"]["complete"] is True


@pytest.mark.parametrize("change", ["golden", "model", "config", "endpoint", "index", "selection"])
def test_resume_rejects_changed_identity_before_model_work(tmp_path, change, monkeypatch):
    settings, output = fixture(tmp_path)
    harness.run_eval(settings, output=output, agent=Agent([{"answer": ""}] * 3), judge=Judge())
    original = output.read_bytes()
    kwargs = {}
    if change == "golden":
        settings.golden_file.write_text(settings.golden_file.read_text() + " ")
    elif change == "model":
        settings.llm_model = "other"
    elif change == "config":
        settings.retrieval_top_k += 1
    elif change == "endpoint":
        settings.llm_base_url = "https://new.example.com"
    elif change == "index":
        settings.index_dir.mkdir()
        (settings.index_dir / "manifest.json").write_text('{"generation":"new"}')
    else:
        kwargs["split"] = "held_out"
    constructors = []
    monkeypatch.setattr(harness, "build_agent", lambda _: constructors.append("agent"))
    with pytest.raises(ValueError):
        harness.run_eval(settings, resume=output, **kwargs)
    assert constructors == []
    assert output.read_bytes() == original


def test_completed_resume_skips_models_and_retry_only_retries_errors(tmp_path, monkeypatch):
    settings, output = fixture(tmp_path)
    harness.run_eval(settings, output=output, judge=Judge(), agent=Agent([
        {"answer": ""}, RuntimeError("secret"), {"answer": ""},
    ]))
    constructors = []
    monkeypatch.setattr(harness, "build_agent", lambda _: constructors.append("agent"))
    harness.run_eval(settings, resume=output)
    assert constructors == []
    agent = Agent([{"answer": ""}])
    results = harness.run_eval(
        settings, resume=output, retry_failures=True, agent=agent, judge=Judge()
    )
    assert agent.queries == ["second"]
    assert [r.id for r in results] == ["a", "b", "c"]
    assert all(not r.errors for r in results)
    # Successful execution with low quality (empty abstention) is not an error.
    assert results[0].abstained is False


def test_malformed_judge_verdict_can_be_explicitly_retried(tmp_path):
    settings, output = fixture(tmp_path)
    harness.run_eval(settings, output=output, categories=["missing"], split="development",
                     agent=Agent([{"answer": "I do not know"}]), judge=Judge())
    agent = Agent([{"answer": ""}])
    results = harness.run_eval(settings, resume=output, retry_failures=True, categories=["missing"],
                               split="development", agent=agent, judge=Judge())
    assert agent.queries == ["first"]
    assert results[0].errors == {}


def test_failed_atomic_replace_leaves_previous_checkpoint_valid(tmp_path, monkeypatch):
    settings, output = fixture(tmp_path)
    harness.run_eval(settings, output=output, agent=Agent([{"answer": ""}] * 3), judge=Judge())
    previous = output.read_bytes()

    def fail_replace(source, destination):
        raise OSError("interrupted storage")

    monkeypatch.setattr(harness.os, "replace", fail_replace)
    with pytest.raises(OSError):
        harness.run_eval(settings, resume=output)
    assert output.read_bytes() == previous
    assert not list(tmp_path.glob(".report.json.*"))


def test_cli_interrupt_and_resume(tmp_path, monkeypatch):
    settings, output = fixture(tmp_path)
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(harness, "build_judge", lambda _: Judge())
    monkeypatch.setattr(
        harness, "build_agent", lambda _: Agent([{"answer": ""}, KeyboardInterrupt()])
    )
    runner = CliRunner()
    first = runner.invoke(cli.app, ["evaluate", "--output", str(output), "--split", "held_out"])
    assert first.exit_code == 130, first.output
    assert len(json.loads(output.read_text())["results"]) == 1
    agent = Agent([{"answer": ""}])
    monkeypatch.setattr(harness, "build_agent", lambda _: agent)
    second = runner.invoke(cli.app, ["evaluate", "--resume", str(output), "--split", "held_out"])
    assert second.exit_code == 0, second.output
    assert agent.queries == ["third"]


def test_rerank_telemetry_survives_checkpoint_and_resume(tmp_path):
    settings, output = fixture(tmp_path)
    event = {"name": "rerank", "elapsed_ms": 2.5, "status": "error"}
    runtime = {"nodes": [{**event, "prompt": "private content"}]}
    harness.run_eval(settings, output=output, judge=Judge(),
                     agent=Agent([{"answer": "", "runtime": runtime}] * 3))
    assert json.loads(output.read_text())["results"][0]["runtime"]["nodes"] == [event]
    results = harness.run_eval(settings, resume=output, agent=Agent([]), judge=Judge())
    assert results[0].runtime["nodes"] == [event]
    assert "private content" not in output.read_text()


def test_runtime_report_uses_numeric_allowlist(tmp_path):
    settings, output = fixture(tmp_path)
    runtime = {"elapsed_ms": 3.0, "model_calls": 2, "api_key": "secret", "prompt": "secret",
               "reported_input_tokens": float("nan"), "status": "completed"}
    harness.run_eval(settings, output=output, agent=Agent([{"answer": "", "runtime": runtime}] * 3),
                     judge=Judge())
    report = json.loads(output.read_text())
    assert "secret" not in output.read_text()
    assert report["results"][0]["runtime"]["model_calls"] == 2
    assert "reported_input_tokens" not in report["results"][0]["runtime"]


def test_retry_interruption_preserves_later_successes(tmp_path):
    settings, output = fixture(tmp_path)
    harness.run_eval(settings, output=output, agent=Agent([
        RuntimeError("failed"), {"answer": ""}, {"answer": ""},
    ]), judge=Judge())
    with pytest.raises(KeyboardInterrupt):
        harness.run_eval(settings, resume=output, retry_failures=True,
                         agent=Agent([KeyboardInterrupt()]), judge=Judge())
    report = json.loads(output.read_text())
    assert [r["id"] for r in report["results"]] == ["b", "c"]
    agent = Agent([{"answer": ""}])
    results = harness.run_eval(settings, resume=output, agent=agent, judge=Judge())
    assert agent.queries == ["first"]
    assert [r.id for r in results] == ["a", "b", "c"]


def test_runtime_limits_participate_in_resume_identity(tmp_path):
    settings, output = fixture(tmp_path)
    harness.run_eval(settings, output=output, agent=Agent([{"answer": ""}] * 3), judge=Judge())
    settings.run_max_model_calls = 1
    with pytest.raises(harness.EvaluationCheckpointError, match="configuration"):
        harness.run_eval(settings, resume=output, agent=Agent([]), judge=Judge())


def test_runtime_failure_summary_and_node_events_are_sanitized(tmp_path):
    settings, output = fixture(tmp_path)
    failure = RuntimeError("api-key-secret")
    failure.runtime_summary = {
        "status": "deadline_exceeded", "elapsed_ms": 42,
        "estimated_cost_usd": None, "budget_cost_usd": .001,
        "token_count_basis": "utf8_content_bytes_excludes_provider_overhead",
        "nodes": [{"name": "retrieve", "elapsed_ms": 41, "status": "error", "prompt": "secret"},
                  {"name": "secret", "elapsed_ms": 1, "status": "error"}],
        "model_events": [{"elapsed_ms": 30, "status": "ok", "api_key": "secret"}],
    }
    harness.run_eval(settings, output=output,
                     agent=Agent([failure, {"answer": ""}, {"answer": ""}]), judge=Judge())
    report = json.loads(output.read_text())
    runtime = report["results"][0]["runtime"]
    assert runtime["status"] == "deadline_exceeded"
    assert runtime["budget_cost_usd"] == .001
    assert runtime["estimated_cost_usd"] is None
    assert runtime["nodes"] == [{"name": "retrieve", "elapsed_ms": 41, "status": "error"}]
    assert runtime["model_events"] == [{"elapsed_ms": 30, "status": "ok"}]
    assert "secret" not in output.read_text()


def test_cli_resume_refusal_has_safe_actionable_reason(tmp_path, monkeypatch):
    settings, output = fixture(tmp_path)
    harness.run_eval(settings, output=output, agent=Agent([{"answer": ""}] * 3), judge=Judge())
    settings.llm_model = "changed-secret-model-name"
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    command = CliRunner().invoke(cli.app, ["evaluate", "--resume", str(output)])
    assert command.exit_code == 1
    assert "Resume refused" in command.output
    assert "configuration" in command.output
    assert "changed-secret-model-name" not in command.output
