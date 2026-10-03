"""Bounded chunk content with multibyte text, overlap, and injected tokenizers."""

import pytest
from langchain_core.embeddings import Embeddings

from finsight.config import Settings
from finsight.rag import ingest
from finsight.rag.models import Document


def _assert_no_lost_text(text, chunks, overlap):
    start = 0
    covered = set()
    for chunk in chunks:
        assert chunk
        assert text[start : start + len(chunk)] == chunk
        covered.update(range(start, start + len(chunk)))
        start = max(start + 1, start + len(chunk) - overlap)
    assert covered == set(range(len(text)))


@pytest.mark.parametrize(
    "text,budget,overlap",
    [
        ("abcdefghijklmno", 3, 8),
        ("甲乙丙丁戊己庚辛", 6, 8),
        ("a甲b乙c丙tail", 4, 6),
        ("abc\n\n甲乙tail", 5, 2),
    ],
)
def test_utf8_budget_preserves_tail_and_clips_overlap(text, budget, overlap):
    chunks = ingest._split_text(text, 12, overlap, max_tokens=budget)
    assert all(len(chunk) <= 12 and len(chunk.encode("utf-8")) <= budget for chunk in chunks)
    _assert_no_lost_text(text, chunks, overlap)
    assert len(chunks) <= len(text)


def test_injected_tokenizer_counts_unicode_content_exactly():
    text = "甲乙丙丁戊己庚辛"
    chunks = ingest._split_text(text, 6, 4, max_tokens=2, token_counter=len)
    assert all(len(chunk) == 2 for chunk in chunks)
    _assert_no_lost_text(text, chunks, 4)


def test_nonmonotonic_prefix_counter_is_supported():
    counts = {"abcd": 3, "abc": 1, "ab": 2, "a": 1, "d": 1}
    chunks = ingest._split_text("abcd", 10, 0, max_tokens=1, token_counter=counts.__getitem__)
    assert chunks == ["abc", "d"]


@pytest.mark.parametrize("count", [-1, True, 1.5, "1"])
def test_invalid_token_counter_results_are_rejected(count):
    with pytest.raises(ValueError, match="nonnegative integer"):
        ingest._split_text("text", 10, 0, max_tokens=2, token_counter=lambda _: count)


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_invalid_token_budget_is_rejected(budget):
    with pytest.raises(ValueError, match="positive integer"):
        ingest._split_text("text", 10, 0, max_tokens=budget)


def test_unrepresentable_character_has_actionable_error():
    with pytest.raises(ValueError, match="increase max_tokens"):
        ingest._split_text("甲乙", 10, 4, max_tokens=2)
    with pytest.raises(ValueError, match="increase max_tokens"):
        ingest._split_text("abc", 10, 4, max_tokens=1, token_counter=lambda _: 2)


def test_zero_token_count_is_valid_and_character_budget_still_applies():
    chunks = ingest._split_text("abcdefgh", 4, 0, max_tokens=1, token_counter=lambda _: 0)
    assert chunks == ["abcd", "efgh"]


def test_document_chunking_preserves_provenance_and_sequential_ids():
    doc = Document("doc", "Title", "甲乙丙丁", "doc.md", published_at="2025-01-01")
    chunks = ingest.chunk_documents([doc], 10, 8, max_tokens=1, token_counter=len)
    assert [chunk.text for chunk in chunks] == list(doc.text)
    assert [chunk.chunk_id for chunk in chunks] == [f"doc:{index}" for index in range(4)]
    assert all(chunk.published_at == "2025-01-01" for chunk in chunks)


def test_empty_document_still_produces_no_chunks():
    assert ingest._split_text(" \n\n ", 10, 0, max_tokens=2) == []


class TrackingEmbeddings(Embeddings):
    def __init__(self):
        self.texts = []

    def embed_documents(self, texts):
        self.texts.extend(texts)
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


def test_build_uses_token_setting_and_reuses_only_identical_chunks(tmp_path, monkeypatch):
    settings = Settings(
        corpus_dir=tmp_path / "corpus",
        index_dir=tmp_path / "index",
        chroma_dir=tmp_path / "chroma",
        chunk_size=20,
        chunk_overlap=4,
        chunk_max_tokens=6,
    )
    settings.corpus_dir.mkdir()
    (settings.corpus_dir / "a.md").write_text("abcdefghijkl")
    embeddings = TrackingEmbeddings()
    monkeypatch.setattr(ingest, "build_embeddings", lambda _: embeddings)
    first_count = ingest.build_index(settings)
    assert first_count == len(embeddings.texts)
    assert all(len(text.encode("utf-8")) <= 6 for text in embeddings.texts)
    embeddings.texts.clear()
    tighter = settings.model_copy(update={"chunk_max_tokens": 3})
    stats = []
    second_count = ingest.build_index(tighter, on_stats=stats.append)
    assert second_count == len(embeddings.texts)
    assert stats[-1].reused_chunks == 0
    assert all(len(text.encode("utf-8")) <= 3 for text in embeddings.texts)
    embeddings.texts.clear()
    ingest.build_index(tighter, on_stats=stats.append)
    assert embeddings.texts == []
    assert stats[-1].reused_chunks == second_count
