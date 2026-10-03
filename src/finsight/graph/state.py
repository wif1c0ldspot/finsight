"""Typed state shared across agent-graph nodes."""

from typing import Any, TypedDict

from finsight.rag.models import RetrievedChunk


class AgentState(TypedDict, total=False):
    """The mutable state threaded through the LangGraph.

    ``total=False`` means nodes may return partial updates; unset keys are simply
    absent until a node writes them.

    Key groups:
      * request      — ``question``, ``current_query``
      * retrieval    — ``retrieved``, ``attempts``
      * verification — ``sufficient``, ``verification_note``
      * answer       — ``answer``, ``citations``
      * grounding    — ``grounded``, ``dangling_citations``, ``grounding_note``
    """

    # request
    question: str
    current_query: str

    # retrieval
    retrieved: list[RetrievedChunk]
    attempts: int
    attempted_queries: list[str]
    reformulation_exhausted: bool

    # verification
    sufficient: bool
    verification_note: str

    # answer
    answer: str
    citations: list[str]
    context_citations: dict[int, RetrievedChunk]
    context_text: str
    no_evidence: bool
    rerank_applied: bool
    rerank_note: str
    semantic_supported: bool | None
    semantic_checked_segments: int
    semantic_unsupported_segments: list[int]

    # grounding
    grounded: bool
    dangling_citations: list[int]
    grounding_note: str

    # Sanitized per-invocation controls and telemetry, supplied by AgentRunner.
    runtime: dict[str, Any]
