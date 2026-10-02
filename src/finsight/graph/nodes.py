"""Agent-graph nodes: retrieve, verify, reformulate, answer, grade, finalize.

Each node is produced by a factory that closes over its dependencies (a retrieval
function or an LLM), keeping the nodes pure and trivially testable with fakes.

Two behaviours changed from the original implementation:

* **Verification is typed.** The old node read the model's free text with
  ``text.upper().startswith("YES")``, which misreads *"… sufficient — YES"* and
  silently treats an empty reply as *insufficient*. It now asks for a JSON object
  and validates it against a pydantic model.
* **Grounding is enforced.** The answer node assesses its own citations (including
  dangling ones) and the graph routes on the result, so an unsupported answer is
  retried or explicitly annotated instead of shipped silently.
"""

from collections.abc import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from finsight.graph.state import AgentState
from finsight.guardrails.validation import assess_grounding, redact_pii
from finsight.rag.format import render_context
from finsight.rag.models import RetrievedChunk
from finsight.structured import StructuredOutputError, invoke_structured

NodeFn = Callable[[AgentState], AgentState]
RetrieveFn = Callable[[str], list[RetrievedChunk]]


def _invoke_text(llm: BaseChatModel, prompt: str) -> str:
    raw = llm.invoke(prompt)
    content = getattr(raw, "content", None)
    if isinstance(content, list):
        return "".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part)
            for part in content
        ).strip()
    return str(content).strip() if content is not None else ""


class VerifyVerdict(BaseModel):
    """Typed verdict from the context-sufficiency check."""

    sufficient: bool = Field(description="Whether the context answers the question")
    reason: str = Field(default="", description="One short sentence of justification")


VERIFY_PROMPT = """You are verifying whether retrieved context is sufficient to answer a question.

Question:
{question}

Retrieved context:
{context}

Is this context sufficient to answer the question accurately?
Reply with ONLY a JSON object: {{"sufficient": true|false, "reason": "<short reason>"}}"""


def make_retrieve_node(retrieve: RetrieveFn) -> NodeFn:
    """Build the retrieve node over any retrieval callable.

    ``retrieve`` is a function rather than a ``HybridRetriever`` so the same node
    can be driven by the in-process retriever or by the MCP tool layer — one
    retrieval contract, two transports.
    """

    def retrieve_node(state: AgentState) -> AgentState:
        query = state.get("current_query") or state["question"]
        return {
            "retrieved": retrieve(query),
            "current_query": query,
            "attempted_queries": [*state.get("attempted_queries", []), query],
            "grounding_note": "",
        }

    return retrieve_node


def make_verify_node(llm: BaseChatModel, *, context_max_chars: int | None = None) -> NodeFn:
    def verify_node(state: AgentState) -> AgentState:
        chunks = state.get("retrieved", [])
        context = render_context(chunks, context_max_chars)
        if not context.citations:
            return {"sufficient": False, "verification_note": "No context retrieved."}

        prompt = VERIFY_PROMPT.format(
            question=state["question"],
            context=context.text,
        )
        try:
            verdict = invoke_structured(llm, prompt, VerifyVerdict)
        except StructuredOutputError as exc:
            # Previously an unparseable reply silently meant "insufficient".
            return {
                "sufficient": False,
                "verification_note": f"Verification could not be parsed ({exc}); "
                "treating context as insufficient.",
            }
        return {
            "sufficient": verdict.sufficient,
            "verification_note": (verdict.reason or "")[:200],
        }

    return verify_node


REFORMULATE_PROMPT = """The following question could not be answered with the retrieved context.
Rewrite it as a clearer, more specific search query (a few words, not a full sentence).

Original question:
{question}

Previous query:
{current_query}

Queries already attempted (do not repeat these):
{attempted_queries}

Retrieval verification feedback:
{verification_note}

Answer grounding feedback:
{grounding_note}

Target the missing evidence described above. Return only a new query.
New query:"""


def make_reformulate_node(llm: BaseChatModel) -> NodeFn:
    def reformulate_node(state: AgentState) -> AgentState:
        current_query = state.get("current_query") or state["question"]
        attempted = state.get("attempted_queries", []) or [current_query]
        prompt = REFORMULATE_PROMPT.format(
            question=state["question"],
            current_query=current_query,
            attempted_queries="\n".join(attempted),
            verification_note=state.get("verification_note", "No verification feedback."),
            grounding_note=state.get("grounding_note") or "No grounding failure.",
        )
        new_query = _invoke_text(llm, prompt)
        normalized = " ".join(new_query.casefold().split())
        duplicate = normalized in {" ".join(query.casefold().split()) for query in attempted}
        exhausted = not normalized or duplicate
        return {
            "current_query": current_query if exhausted else new_query,
            "attempts": state.get("attempts", 0) + 1,
            "reformulation_exhausted": exhausted,
        }

    return reformulate_node


ANSWER_PROMPT = """Answer the question using ONLY the retrieved context below.
Cite every factual claim with its bracketed reference number (e.g. [1], [2]).
Only cite reference numbers that appear in the context above.
If the context does not contain the answer, say so explicitly rather than guessing.

Question:
{question}

Context:
{context}

Answer (with citations):"""


def make_answer_node(
    llm: BaseChatModel, *, context_max_chars: int | None = None
) -> NodeFn:
    def answer_node(state: AgentState) -> AgentState:
        chunks = state.get("retrieved", [])
        context = render_context(chunks, context_max_chars)
        prompt = ANSWER_PROMPT.format(
            question=state["question"],
            context=context.text,
        )
        answer = redact_pii(_invoke_text(llm, prompt))
        grounding = assess_grounding(answer, context.citations)
        return {
            "context_citations": context.citations,
            "context_text": context.text,
            "answer": answer,
            "citations": grounding.cited_doc_ids,
            "grounded": grounding.is_grounded,
            "dangling_citations": grounding.dangling_citations,
            "grounding_note": grounding.reason,
        }

    return answer_node


def make_finalize_node(*, enforce_grounding: bool) -> NodeFn:
    """Terminal node: annotate an ungrounded answer rather than hiding it.

    The run has already exhausted its re-query budget by the time this executes,
    so the honest options are to ship the answer with a visible warning or to
    discard it. A visible warning is more useful than silence.
    """

    def finalize_node(state: AgentState) -> AgentState:
        if state.get("grounded", False) or not enforce_grounding:
            return {}
        note = state.get("grounding_note", "Answer is not grounded in the retrieved context.")
        answer = state.get("answer", "")
        return {"answer": f"{answer}\n\n[citation warning] {note}"}

    return finalize_node
