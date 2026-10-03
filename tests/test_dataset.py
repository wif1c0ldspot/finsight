"""The public benchmark labels must resolve to dated, inspectable source snapshots."""
from pathlib import Path

from finsight.eval.harness import load_golden
from finsight.rag.ingest import load_documents

ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_sources_and_labels_are_consistent():
    documents = load_documents(ROOT / "data/corpus")
    by_id = {document.doc_id: document for document in documents}
    cases = load_golden(ROOT / "data/golden/golden.json")
    assert len(documents) >= 10
    assert len(cases) >= 40
    for document in documents:
        assert document.source_url and document.source_url.startswith("https://")
        assert document.published_at
        assert document.retrieved_at
        assert document.revision
    for case in cases:
        assert set(case["expected_doc_ids"]) <= by_id.keys(), case["id"]
        if case["answerable"]:
            assert case["expected_doc_ids"] and case["ground_truth"].strip(), case["id"]
    assert {case["category"] for case in cases} >= {
        "single_fact", "numerical", "multi_document", "temporal", "misleading",
        "unanswerable", "adversarial",
    }
    development = {case["question"].casefold() for case in cases if case["split"] == "development"}
    held_out = {case["question"].casefold() for case in cases if case["split"] == "held_out"}
    assert development and held_out and not development.intersection(held_out)
