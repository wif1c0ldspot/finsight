"""Core data models for the RAG pipeline."""

from dataclasses import dataclass
from typing import Literal

#: How a chunk was surfaced. ``rrf`` means it came out of rank fusion.
RetrievalMethod = Literal["vector", "bm25", "rrf"]


@dataclass(frozen=True)
class Document:
    """A single source document loaded from the corpus."""

    doc_id: str
    title: str
    text: str
    source: str


@dataclass(frozen=True)
class Chunk:
    """A contiguous slice of a document, with its provenance."""

    chunk_id: str
    doc_id: str
    title: str
    text: str
    position: int


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by the retriever, with its fusion score and provenance.

    ``component_scores`` retains the per-retriever similarity scores that the
    fusion step discards. They are not used for ranking (RRF is rank-based), but
    they make a result explainable: you can see whether a chunk won on dense or
    sparse agreement.
    """

    chunk: Chunk
    score: float
    rank: int
    method: RetrievalMethod = "rrf"
    component_scores: dict[str, float] | None = None
