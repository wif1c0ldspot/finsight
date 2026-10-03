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
from typing import Any, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from finsight.config import Settings
from finsight.graph.nodes import (
    RetrieveFn,
    make_answer_node,
    make_finalize_node,
    make_grade_node,
    make_reformulate_node,
    make_rerank_node,
    make_retrieve_node,
    make_verify_node,
)
from finsight.graph.state import AgentState
from finsight.llm import build_llm
from finsight.rag.models import RetrievalFilter
from finsight.rag.retrieve import HybridRetriever
from finsight.runtime import AgentRunner, BudgetedModel, RunLimits, timed_node


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
        if state.get("grounded") or state.get("no_evidence"):
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
    settings: Settings, retriever: HybridRetriever, filters: RetrievalFilter | None = None,
) -> tuple[RetrieveFn, str]:
    """Return the retrieval callable plus a label for which path is in use."""
    top_k = (
        max(settings.retrieval_candidates, settings.retrieval_top_k)
        if settings.rerank_enabled else None
    )
    if settings.use_mcp_tools:
        from finsight.mcp.tools import make_retrieve_tool

        return make_retrieve_tool(retriever, filters=filters, top_k=top_k), "mcp"
    if filters is not None:
        return lambda query: retriever.retrieve(query, top_k=top_k, filters=filters), "direct"
    if top_k is not None:
        return lambda query: retriever.retrieve(query, top_k=top_k), "direct"
    return retriever.retrieve, "direct"


def build_agent(
    settings: Settings,
    *,
    retriever: HybridRetriever | None = None,
    llm: BaseChatModel | None = None,
    filters: RetrievalFilter | None = None,
) -> Any:
    """Build the compiled LangGraph for the research agent.

    ``retriever`` and ``llm`` are injectable for testing; otherwise they are
    constructed from ``settings``.
    """
    active_retriever = retriever if retriever is not None else HybridRetriever(settings)
    active_llm = cast(BaseChatModel, BudgetedModel(
        llm if llm is not None else build_llm(settings)
    ))
    retrieve_fn, _path = _resolve_retrieve_fn(settings, active_retriever, filters)

    graph: StateGraph[AgentState] = StateGraph(AgentState)

    # Wrap every node so custom extensions share timing and control boundaries.
    graph.add_node("retrieve", timed_node("retrieve", make_retrieve_node(retrieve_fn)))
    graph.add_node(
        "verify", timed_node("verify", make_verify_node(
            active_llm, context_max_chars=settings.context_max_chars,
            context_max_tokens=settings.context_max_tokens,
        ))
    )
    graph.add_node(
        "reformulate", timed_node("reformulate", make_reformulate_node(active_llm))
    )
    graph.add_node(
        "answer",
        timed_node("answer", make_answer_node(
            active_llm, context_max_chars=settings.context_max_chars,
            context_max_tokens=settings.context_max_tokens,
        )),
    )
    graph.add_node("grade", timed_node("grade", make_grade_node(
        active_llm, enabled=settings.semantic_verification,
        max_segments=settings.semantic_max_segments,
    )))
    graph.add_node(
        "finalize", timed_node("finalize", make_finalize_node(
            enforce_grounding=settings.enforce_grounding
        ))
    )

    graph.add_edge(START, "retrieve")
    if settings.rerank_enabled:
        graph.add_node("rerank", timed_node("rerank", make_rerank_node(
            active_llm, top_k=settings.retrieval_top_k, max_chars=settings.context_max_chars,
            max_tokens=settings.context_max_tokens,
        )))
        graph.add_edge("retrieve", "rerank")
        graph.add_edge("rerank", "verify")
    else:
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

    # Each full retry can execute retrieve/verify/answer/grade/reformulate.
    # Keep LangGraph's safety limit above the configured finite loop budget.
    compiled = graph.compile().with_config(
        {"recursion_limit": 6 * (settings.max_retrieval_attempts + 1) + 4}
    )
    return AgentRunner(compiled, RunLimits.from_settings(settings))
