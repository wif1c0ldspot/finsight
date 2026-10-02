"""Assemble the agent graph and expose a single entrypoint.

Topology:

    START → retrieve → verify ─┬─ sufficient ──────────────────→ answer
                               └─ insufficient (attempts left) → reformulate → retrieve

    answer → grade ─┬─ grounded ───────────────────────────────→ finalize → END
                    ├─ ungrounded (attempts left, enforced) ───→ reformulate → retrieve
                    └─ budget exhausted ───────────────────────→ finalize → END

An empty or previously attempted reformulation skips another retrieval and
finishes against the current evidence without further retrying.

The second decision point is what makes the grounding guardrail an actual
guardrail: an answer citing nothing — or citing sources that do not exist —
triggers another retrieval attempt instead of being returned as-is.

``build_agent`` takes injectable ``retriever`` and ``llm`` so the topology can be
exercised with fakes. Previously the graph could only be constructed against live
services, which is why the router and the loop bound went untested.
"""

from collections.abc import Callable
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from finsight.config import Settings
from finsight.graph.nodes import (
    RetrieveFn,
    make_answer_node,
    make_finalize_node,
    make_reformulate_node,
    make_retrieve_node,
    make_verify_node,
)
from finsight.graph.state import AgentState
from finsight.llm import build_llm
from finsight.rag.retrieve import HybridRetriever


def _noop_node(_state: AgentState) -> AgentState:
    """Pass-through node so the conditional edge from ``answer`` has a named source.

    All of the grading logic lives in :func:`_grade_router`; this exists only
    because LangGraph conditional edges must originate from a node.
    """
    return {}


def _verify_router(max_attempts: int) -> Callable[[AgentState], str]:
    def route(state: AgentState) -> str:
        if state.get("sufficient"):
            return "answer"
        if state.get("attempts", 0) >= max_attempts:
            return "answer"
        return "reformulate"

    return route


def _grade_router(max_attempts: int, enforce_grounding: bool) -> Callable[[AgentState], str]:
    def route(state: AgentState) -> str:
        if state.get("grounded"):
            return "finalize"
        if (
            not enforce_grounding
            or state.get("attempts", 0) >= max_attempts
            or state.get("reformulation_exhausted", False)
        ):
            return "finalize"
        return "reformulate"

    return route


def _resolve_retrieve_fn(
    settings: Settings, retriever: HybridRetriever
) -> tuple[RetrieveFn, str]:
    """Return the retrieval callable plus a label for which path is in use."""
    if settings.use_mcp_tools:
        from finsight.mcp.tools import make_retrieve_tool

        return make_retrieve_tool(retriever), "mcp"
    return retriever.retrieve, "direct"


def build_agent(
    settings: Settings,
    *,
    retriever: HybridRetriever | None = None,
    llm: BaseChatModel | None = None,
) -> Any:
    """Build the compiled LangGraph for the research agent.

    ``retriever`` and ``llm`` are injectable for testing; otherwise they are
    constructed from ``settings``.
    """
    active_retriever = retriever if retriever is not None else HybridRetriever(settings)
    active_llm = llm if llm is not None else build_llm(settings)
    retrieve_fn, _path = _resolve_retrieve_fn(settings, active_retriever)

    graph: StateGraph[AgentState] = StateGraph(AgentState)

    # LangGraph types nodes via a contravariant-TypeVar Protocol that mypy strict
    # cannot resolve against plain callables; the registrations below are correct.
    graph.add_node("retrieve", make_retrieve_node(retrieve_fn))  # type: ignore[call-overload]
    graph.add_node(  # type: ignore[call-overload]
        "verify", make_verify_node(active_llm, context_max_chars=settings.context_max_chars)
    )
    graph.add_node(  # type: ignore[call-overload]
        "reformulate", make_reformulate_node(active_llm)
    )
    graph.add_node(  # type: ignore[call-overload]
        "answer",
        make_answer_node(active_llm, context_max_chars=settings.context_max_chars),
    )
    graph.add_node("grade", _noop_node)  # type: ignore[call-overload]
    graph.add_node(  # type: ignore[call-overload]
        "finalize", make_finalize_node(enforce_grounding=settings.enforce_grounding)
    )

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "verify")
    graph.add_conditional_edges(
        "verify",
        _verify_router(settings.max_retrieval_attempts),
        {"answer": "answer", "reformulate": "reformulate"},
    )
    graph.add_conditional_edges(
        "reformulate",
        lambda state: "answer" if state.get("reformulation_exhausted") else "retrieve",
        {"answer": "answer", "retrieve": "retrieve"},
    )
    graph.add_edge("answer", "grade")
    graph.add_conditional_edges(
        "grade",
        _grade_router(settings.max_retrieval_attempts, settings.enforce_grounding),
        {"finalize": "finalize", "reformulate": "reformulate"},
    )
    graph.add_edge("finalize", END)

    return graph.compile()
