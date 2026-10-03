"""Offline CLI workflows against real temporary Chroma generations and provenance."""

import json
from types import SimpleNamespace

import chromadb
import pytest
from langchain_core.embeddings import Embeddings
from rich.console import Console
from typer.testing import CliRunner

from finsight import cli
from finsight.config import Settings
from finsight.eval import harness
from finsight.graph import builder
from finsight.rag import ingest, retrieve
from finsight.rag.index import read_manifest


class OfflineEmbeddings(Embeddings):
    def __init__(self):
        self.document_batches = []
        self.queries = []

    def embed_documents(self, texts):
        self.document_batches.append(list(texts))
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text):
        self.queries.append(text)
        return [1.0, 0.0, 0.0]


class WorkflowModel:
    def __init__(self):
        self.prompts = []
        self.error = None

    def with_structured_output(self, schema):
        raise NotImplementedError

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if self.error is not None:
            raise self.error
        if "Assess factual support for EVERY" in prompt:
            response = '{"segments":[{"segment":1,"supported":true}]}'
        elif "verifying whether retrieved context" in prompt:
            response = '{"sufficient":true,"reason":"The date is in the source."}'
        elif "evaluating whether an answer is faithful" in prompt:
            response = '{"score":5,"rationale":"Supported."}'
        elif "same factual content as the reference" in prompt:
            response = '{"correct":true,"rationale":"Same date."}'
        else:
            response = "Alpha was founded in 2024 [1]."
        return SimpleNamespace(content=response)


@pytest.fixture
def workflow(monkeypatch, tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for doc_id, body, date in [
        ("alpha", "Alpha was founded in 2024. PRIVATE-ALPHA-SOURCE", "2024-06-01"),
        ("archive", "Archive was founded in 1999. PRIVATE-ARCHIVE-SOURCE", "1999-01-01"),
    ]:
        (corpus / f"{doc_id}.md").write_text(f"# {doc_id.title()}\n\n{body}\n")
        (corpus / f"{doc_id}.metadata.json").write_text(json.dumps({
            "source_url": f"https://sources.example/{doc_id}", "published_at": date,
            "retrieved_at": "2026-10-03", "revision": "edition-1",
        }))
    settings = Settings(
        _env_file=None, corpus_dir=corpus, index_dir=tmp_path / "index",
        chroma_dir=tmp_path / "chroma", golden_file=tmp_path / "golden.json",
        max_retrieval_attempts=0, retrieval_top_k=2, embed_model="offline-fixture-v1",
        llm_api_key="FAKE-CREDENTIAL-MUST-NOT-LEAK",
    )
    embeddings, answer, judge = OfflineEmbeddings(), WorkflowModel(), WorkflowModel()
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(cli, "console", Console(width=200, color_system=None))
    monkeypatch.setattr(ingest, "build_embeddings", lambda _: embeddings)
    monkeypatch.setattr(retrieve, "build_embeddings", lambda _: embeddings)
    monkeypatch.setattr(builder, "build_llm", lambda _: answer)
    monkeypatch.setattr(harness, "build_judge", lambda _: judge)
    runner = CliRunner()

    def invoke(*arguments):
        return runner.invoke(cli.app, list(arguments))

    def indexed():
        result = invoke("ingest")
        assert result.exit_code == 0, result.output
        return result

    return SimpleNamespace(settings=settings, embeddings=embeddings, answer=answer, judge=judge,
                           invoke=invoke, indexed=indexed, root=tmp_path)


def test_ingest_reuses_vectors_full_rebuild_reembeds_and_cleanup_is_dry_run(workflow):
    first = workflow.indexed()
    initial_generation = read_manifest(workflow.settings.index_dir).generation
    assert "Embedded 2; reused 0." in first.output
    second = workflow.invoke("ingest")
    assert second.exit_code == 0, second.output
    assert "Embedded 0; reused 2." in second.output
    assert sum(map(len, workflow.embeddings.document_batches)) == 2
    assert read_manifest(workflow.settings.index_dir).generation != initial_generation

    full = workflow.invoke("ingest", "--full")
    assert full.exit_code == 0, full.output
    assert "Embedded 2; reused 0." in full.output
    assert sum(map(len, workflow.embeddings.document_batches)) == 4
    client = chromadb.PersistentClient(path=str(workflow.settings.chroma_dir))
    collections_before = {collection.name for collection in client.list_collections()}
    files_before = set(workflow.settings.index_dir.rglob("*"))
    current_before = read_manifest(workflow.settings.index_dir).generation
    preview = workflow.invoke("index-cleanup", "--keep", "0", "--min-age-hours", "0")
    assert preview.exit_code == 0, preview.output
    assert "Dry run: no generations deleted." in preview.output
    assert "would_delete" in preview.output
    assert {collection.name for collection in client.list_collections()} == collections_before
    assert set(workflow.settings.index_dir.rglob("*")) == files_before
    assert read_manifest(workflow.settings.index_dir).generation == current_before


def test_filtered_claim_verified_answer_displays_provenance_but_trace_is_content_free(workflow):
    workflow.indexed()
    trace_path = workflow.root / "ask-trace.json"
    question = "When was Alpha founded? PRIVATE-QUESTION"
    result = workflow.invoke(
        "ask", question, "--doc", "alpha", "--published-after", "2024-01-01",
        "--published-before", "2024-12-31", "--verify-claims", "--trace", str(trace_path),
    )
    assert result.exit_code == 0, result.output
    assert "Alpha was founded in 2024" in result.output
    assert "Semantic support passed: True" in result.output
    assert "2024-06-01" in result.output
    assert "https://sources.example/alpha" in result.output
    assert len(workflow.answer.prompts) == 3
    assert all("PRIVATE-ARCHIVE-SOURCE" not in prompt for prompt in workflow.answer.prompts)
    assert any("PRIVATE-ALPHA-SOURCE" in prompt for prompt in workflow.answer.prompts)
    saved = trace_path.read_text()
    trace = json.loads(saved)
    assert trace["trace_version"] == 1
    assert trace["runtime"]["status"] == "completed"
    assert trace["runtime"]["model_calls"] == 3
    for private in (question, "PRIVATE-ALPHA-SOURCE", "PRIVATE-ARCHIVE-SOURCE",
                    "FAKE-CREDENTIAL-MUST-NOT-LEAK", "https://sources.example/alpha"):
        assert private not in saved

    workflow.answer.prompts.clear()
    outside_date = workflow.invoke("ask", "When was Alpha founded?", "--doc", "alpha",
                                   "--published-before", "2020-01-01")
    assert outside_date.exit_code == 0, outside_date.output
    assert "cannot answer" in outside_date.output
    assert workflow.answer.prompts == []  # the date filter excludes the otherwise matching doc


def test_exhausted_model_budget_exits_nonzero_and_saves_failure_trace(workflow):
    workflow.indexed()
    path = workflow.root / "budget-trace.json"
    result = workflow.invoke("ask", "When was Alpha founded?", "--max-model-calls", "1",
                             "--trace", str(path))
    assert result.exit_code == 1
    assert "model_call_budget_exceeded" in result.output
    saved = json.loads(path.read_text())["runtime"]
    assert saved["status"] == "model_call_budget_exceeded"
    assert saved["model_calls"] == 1
    assert len(workflow.answer.prompts) == 1
    assert saved["nodes"][-1]["status"] == "model_call_budget_exceeded"


def test_interrupted_model_exits_130_and_saves_cancelled_trace(workflow):
    workflow.indexed()
    workflow.answer.error = KeyboardInterrupt("PRIVATE-INTERRUPT-PAYLOAD")
    path = workflow.root / "cancelled-trace.json"
    result = workflow.invoke("ask", "When was Alpha founded?", "--trace", str(path))
    assert result.exit_code == 130, result.output
    assert "Run cancelled." in result.output
    saved = path.read_text()
    runtime = json.loads(saved)["runtime"]
    assert runtime["status"] == "cancelled"
    assert runtime["model_calls"] == 1
    assert runtime["nodes"][-1]["status"] == "cancelled"
    assert runtime["model_events"][-1]["status"] == "cancelled"
    for private in ("PRIVATE-INTERRUPT-PAYLOAD", "PRIVATE-ALPHA-SOURCE",
                    "FAKE-CREDENTIAL-MUST-NOT-LEAK"):
        assert private not in saved
        assert private not in result.output


def test_evaluate_filters_resume_without_calls_and_compare_compatible_reports(workflow):
    workflow.indexed()
    cases = [
        {"id": "held-profile", "category": "profile", "split": "held_out"},
        {"id": "held-history", "category": "history", "split": "held_out"},
        {"id": "dev-profile", "category": "profile", "split": "development"},
        {"id": "held-other", "category": "other", "split": "held_out"},
    ]
    workflow.settings.golden_file.write_text(json.dumps([
        {**case, "question": "When was Alpha founded?", "expected_doc_ids": ["alpha"],
         "ground_truth": "Alpha was founded in 2024."} for case in cases
    ]))
    baseline, resumed = workflow.root / "baseline.json", workflow.root / "resumed.json"
    selection = ["--split", "held_out", "--category", "profile", "--category", "history"]
    result = workflow.invoke("evaluate", "--output", str(baseline), *selection)
    assert result.exit_code == 0, result.output
    report = json.loads(baseline.read_text())
    assert {case["id"] for case in report["results"]} == {"held-profile", "held-history"}
    assert report["summary"]["n"] == 2
    assert set(report["summary"]["by_category"]) == {"profile", "history"}
    assert all(case["correctness"] == 1 and case["agent_succeeded"] for case in report["results"])
    prior = (len(workflow.answer.prompts), len(workflow.judge.prompts),
             len(workflow.embeddings.queries))
    assert all(count > 0 for count in prior)

    continuation = workflow.invoke("evaluate", "--resume", str(baseline),
                                   "--output", str(resumed), *selection)
    assert continuation.exit_code == 0, continuation.output
    assert (len(workflow.answer.prompts), len(workflow.judge.prompts),
            len(workflow.embeddings.queries)) == prior
    assert json.loads(resumed.read_text())["results"] == report["results"]
    compared = workflow.invoke("compare", str(baseline), str(resumed))
    assert compared.exit_code == 0, compared.output
    comparison = json.loads(compared.output)
    assert comparison["overall"]["paired"]["mean_correctness"]["paired_count"] == 2
    assert comparison["overall"]["deltas"]["mean_correctness"] == 0


def test_generic_provider_error_does_not_leak_credentials_to_cli_or_trace(workflow):
    workflow.indexed()
    workflow.answer.error = RuntimeError("provider echoed FAKE-CREDENTIAL-MUST-NOT-LEAK")
    path = workflow.root / "provider-error-trace.json"
    result = workflow.invoke("ask", "When was Alpha founded?", "--trace", str(path))
    assert result.exit_code == 1
    assert "RuntimeError" in result.output
    assert "FAKE-CREDENTIAL-MUST-NOT-LEAK" not in result.output
    assert "FAKE-CREDENTIAL-MUST-NOT-LEAK" not in path.read_text()
    assert json.loads(path.read_text())["runtime"]["status"] == "error"
