"""Core data models for the RAG pipeline."""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

#: How a chunk was surfaced. ``rrf`` means it came out of rank fusion.
RetrievalMethod = Literal["vector", "bm25", "rrf"]


@dataclass(frozen=True)
class Document:
    """A single source document loaded from the corpus."""

    doc_id: str
    title: str
    text: str
    source: str
    source_url: str | None = None
    published_at: str | None = None
    retrieved_at: str | None = None
    revision: str | None = None


@dataclass(frozen=True)
class Chunk:
    """A contiguous slice of a document, with its provenance."""

    chunk_id: str
    doc_id: str
    title: str
    text: str
    position: int
    source_url: str | None = None
    published_at: str | None = None
    retrieved_at: str | None = None
    revision: str | None = None


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


def _iso_date(value: str) -> str:
    if len(value) != 10 or date.fromisoformat(value).isoformat() != value:
        raise ValueError("Expected an ISO date (YYYY-MM-DD)")
    return value


def _source_url(value: str) -> str:
    parts = urlsplit(value)
    if (
        parts.scheme not in {"https", "http"}
        or not parts.hostname
        or any(c.isspace() for c in value)
    ):
        raise ValueError("source_url must be an absolute HTTP(S) URL")
    if parts.username is not None or parts.password is not None:
        raise ValueError("source_url must not contain credentials")
    return value


class SourceMetadata(BaseModel):
    """Optional .metadata.json provenance; absent dates are never inferred.

    published_at uses YYYY-MM-DD. retrieved_at accepts an ISO date or date-time.
    revision defaults at ingestion to sha256:<digest of the Markdown content>.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    source_url: str | None = None
    published_at: str | None = None
    retrieved_at: str | None = None
    revision: str | None = None

    @field_validator("source_url")
    @classmethod
    def valid_url(cls, value: str | None) -> str | None:
        return _source_url(value) if value is not None else value

    @field_validator("published_at")
    @classmethod
    def valid_date(cls, value: str | None) -> str | None:
        return _iso_date(value) if value is not None else value

    @field_validator("retrieved_at")
    @classmethod
    def valid_retrieval_date(cls, value: str | None) -> str | None:
        if value is not None:
            if len(value) == 10:
                _iso_date(value)
            else:
                if len(value) < 19 or value[10] != "T":
                    raise ValueError("retrieved_at must be an ISO date or date-time")
                datetime.fromisoformat(value)
        return value

    @field_validator("revision")
    @classmethod
    def valid_revision(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("revision must not be blank")
        return value


class RetrievalFilter(BaseModel):
    """AND across fields, OR within lists; date bounds are inclusive.

    A date filter excludes documents without a published_at date. Empty lists
    intentionally match nothing. Source URLs match exactly as recorded.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    doc_ids: list[str] | None = None
    source_urls: list[str] | None = None
    published_after: str | None = None
    published_before: str | None = None

    @field_validator("doc_ids", "source_urls")
    @classmethod
    def nonblank_values(cls, values: list[str] | None) -> list[str] | None:
        if values is not None and any(not value.strip() for value in values):
            raise ValueError("Filter values must not be blank")
        return values

    @field_validator("source_urls")
    @classmethod
    def valid_urls(cls, values: list[str] | None) -> list[str] | None:
        if values is not None:
            for value in values:
                _source_url(value)
        return values

    @field_validator("published_after", "published_before")
    @classmethod
    def valid_date(cls, value: str | None) -> str | None:
        return _iso_date(value) if value is not None else value

    @model_validator(mode="after")
    def valid_range(self) -> "RetrievalFilter":
        if (
            self.published_after
            and self.published_before
            and self.published_after > self.published_before
        ):
            raise ValueError("published_after must be on or before published_before")
        return self

    def matches(self, chunk: Chunk) -> bool:
        if self.doc_ids is not None and chunk.doc_id not in self.doc_ids:
            return False
        if self.source_urls is not None and chunk.source_url not in self.source_urls:
            return False
        if self.published_after is not None and (
            chunk.published_at is None or chunk.published_at < self.published_after
        ):
            return False
        if self.published_before is not None and (
            chunk.published_at is None or chunk.published_at > self.published_before
        ):
            return False
        return True
