"""Tests for retrieval logic: chunking, tokenisation, RRF fusion, ranking metrics."""

from finsight.eval.metrics import (
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    retrieval_recall,
)
from finsight.rag.ingest import _split_text, chunk_documents, tokenize
from finsight.rag.models import Chunk, Document
from finsight.rag.retrieve import HybridRetriever


def test_tokenize_lowercases_and_splits():
    assert tokenize("Hello World 123") == ["hello", "world", "123"]


def test_split_text_respects_chunk_size():
    text = "one\n\ntwo\n\nthree\n\nfour"
    chunks = _split_text(text, chunk_size=20, overlap=0)
    assert all(len(c) <= 20 for c in chunks)


def test_chunk_documents_assigns_ids_and_positions():
    doc = Document(doc_id="a", title="A", text="para one\n\npara two", source="a.md")
    chunks = chunk_documents([doc], chunk_size=100, overlap=0)
    assert chunks[0].chunk_id == "a:0"
    assert chunks[0].doc_id == "a"


# --- ranking metrics -------------------------------------------------------


def test_retrieval_recall_is_rank_blind():
    """Kept for backwards compatibility; documents the reason recall_at_k exists."""
    assert retrieval_recall(["a", "b"], ["a"]) == 1.0
    assert retrieval_recall(["b", "c"], ["a"]) == 0.0
    assert retrieval_recall(["a"], ["a", "d"]) == 0.5


def test_recall_at_k_respects_the_cutoff():
    # The relevant document sits at position 3, so it counts at k=3 but not k=2.
    retrieved = ["x", "y", "a"]
    assert recall_at_k(retrieved, ["a"], 2) == 0.0
    assert recall_at_k(retrieved, ["a"], 3) == 1.0


def test_reciprocal_rank_rewards_early_hits():
    assert reciprocal_rank(["a", "b"], ["a"]) == 1.0
    assert reciprocal_rank(["x", "a"], ["a"]) == 0.5
    assert reciprocal_rank(["x", "y"], ["a"]) == 0.0


def test_ndcg_prefers_relevant_documents_ranked_higher():
    early = ndcg_at_k(["a", "x", "y"], ["a"], 3)
    late = ndcg_at_k(["x", "y", "a"], ["a"], 3)
    assert early > late
    assert ndcg_at_k(["a"], ["a"], 1) == 1.0


# --- fusion ----------------------------------------------------------------


def test_rrf_fusion_combines_ranks():
    retriever = object.__new__(HybridRetriever)
    retriever._chunks = [
        Chunk(chunk_id="a:0", doc_id="a", title="t", text="x", position=0),
        Chunk(chunk_id="b:0", doc_id="b", title="t", text="y", position=0),
    ]
    vector = [("a:0", 0.9), ("b:0", 0.5)]
    bm25 = [("b:0", 10.0), ("a:0", 5.0)]
    fused = retriever._rrf_fuse(vector, bm25, top_k=2)
    assert [r.chunk.chunk_id for r in fused] == ["a:0", "b:0"]


def test_rrf_fusion_retains_component_scores():
    """The similarity scores used to be computed and thrown away."""
    retriever = object.__new__(HybridRetriever)
    retriever._chunks = [
        Chunk(chunk_id="a:0", doc_id="a", title="t", text="x", position=0),
    ]
    fused = retriever._rrf_fuse([("a:0", 0.75)], [("a:0", 3.5)], top_k=1)
    assert fused[0].component_scores == {"vector": 0.75, "bm25": 3.5}


def test_rrf_fusion_rejects_unknown_chunk_ids():
    """An unguarded by_id[cid] used to raise a bare KeyError here."""
    import pytest

    from finsight.rag.index import IndexIntegrityError

    retriever = object.__new__(HybridRetriever)
    retriever._chunks = [
        Chunk(chunk_id="a:0", doc_id="a", title="t", text="x", position=0),
    ]
    with pytest.raises(IndexIntegrityError, match="unknown chunk id"):
        retriever._rrf_fuse([("ghost:9", 0.9)], [], top_k=1)
