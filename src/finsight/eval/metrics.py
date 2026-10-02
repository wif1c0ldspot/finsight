"""Evaluation metrics.

Two things were wrong with the original metrics:

1. **Recall was rank-blind set-overlap**, so a document at rank 20 scored the same
   as one at rank 1 — even though ranking is the whole point of a retriever.
   ``recall_at_k``, ``reciprocal_rank`` and ``ndcg_at_k`` are now first-class;
   ``retrieval_recall`` is kept only for backwards compatibility.
2. **Faithfulness used a regex** (``re.search(r"\\d+", text)``) against free-text
   judge output, which reads *"I'd rate this 4/5, though 3 is arguable"* as **4** and
   clamps an out-of-range *"10"* to 5. It now requests typed JSON.

``faithfulness`` also returns ``None`` — not ``0.0`` — when there is nothing to
judge, so a retrieval failure is no longer averaged in as an unfaithful answer.
"""

from __future__ import annotations

import math
import re

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from finsight.rag.format import format_context
from finsight.rag.models import RetrievedChunk
from finsight.structured import StructuredOutputError, invoke_structured

_ABSTENTION = re.compile(
    r"("
    r"i (?:do not|don't|cannot|can't) (?:know|answer|find|determine)"
    r"|not (?:enough|sufficient) (?:information|context|detail)"
    r"|no (?:information|context|details?) (?:is |are )?(?:available|provided|given)"
    r"|cannot be (?:answered|determined|found)"
    r"|the context does not (?:contain|include|mention|provide)"
    r"|unable to (?:answer|determine|find)"
    r")",
    re.IGNORECASE,
)


# --- Retrieval quality ----------------------------------------------------


def retrieval_recall(retrieved_doc_ids: list[str], expected_doc_ids: list[str]) -> float:
    """Fraction of expected documents present anywhere in the retrieved set.

    Retained for backwards compatibility. Prefer :func:`recall_at_k`; this
    function is rank-blind.
    """
    if not expected_doc_ids:
        return 1.0
    return len(set(expected_doc_ids) & set(retrieved_doc_ids)) / len(expected_doc_ids)


def recall_at_k(retrieved_doc_ids: list[str], expected_doc_ids: list[str], k: int) -> float:
    """Fraction of expected documents present in the top ``k`` results."""
    if not expected_doc_ids:
        return 1.0
    return len(set(expected_doc_ids) & set(retrieved_doc_ids[:k])) / len(expected_doc_ids)


def reciprocal_rank(retrieved_doc_ids: list[str], expected_doc_ids: list[str]) -> float:
    """1 / rank of the first relevant document; 0.0 if none appears."""
    if not expected_doc_ids:
        return 1.0
    expected = set(expected_doc_ids)
    for position, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in expected:
            return 1.0 / position
    return 0.0


def ndcg_at_k(retrieved_doc_ids: list[str], expected_doc_ids: list[str], k: int) -> float:
    """Binary-relevance nDCG@k: rewards finding relevant documents early."""
    if not expected_doc_ids:
        return 1.0
    expected = set(expected_doc_ids)
    # Chunks from one document earn relevance once; duplicates still consume
    # result positions, because they displace other relevant documents.
    seen: set[str] = set()
    dcg = 0.0
    for position, doc_id in enumerate(retrieved_doc_ids[:max(k, 0)], start=1):
        if doc_id in expected and doc_id not in seen:
            dcg += 1.0 / math.log2(position + 1)
            seen.add(doc_id)
    ideal_hits = min(len(expected), k)
    idcg = sum(1.0 / math.log2(position + 1) for position in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


# --- Answer quality -------------------------------------------------------


class FaithfulnessScore(BaseModel):
    """Typed judge verdict."""

    score: int = Field(ge=1, le=5, description="1 = fabricates, 5 = fully supported")
    rationale: str = Field(default="", description="One short sentence")


class CorrectnessScore(BaseModel):
    """Typed correctness verdict."""

    correct: bool = Field(description="Whether the answer matches the reference")
    rationale: str = Field(default="")


_FAITHFULNESS_PROMPT = """You are evaluating whether an answer is faithful to the provided context.

Question:
{question}

Context:
{context}

Answer:
{answer}

Rate how factually supported the answer is by the context on a scale of 1-5,
where 1 = contradicts or fabricates and 5 = fully supported.
Reply with ONLY a JSON object: {{"score": <1-5>, "rationale": "<short reason>"}}"""

_CORRECTNESS_PROMPT = """Does the candidate answer convey the same factual content as the reference?

Minor wording differences are acceptable; missing or contradictory facts are not.

Question:
{question}

Reference answer:
{reference}

Candidate answer:
{answer}

Reply with ONLY a JSON object: {{"correct": true|false, "rationale": "<short reason>"}}"""


def faithfulness(
    llm: BaseChatModel,
    question: str,
    answer: str,
    chunks: list[RetrievedChunk],
    *,
    context_text: str | None = None,
) -> float | None:
    """LLM-as-judge support score in 0-1, or ``None`` when there is nothing to judge.

    ``None`` — rather than ``0.0`` — distinguishes *"retrieval returned nothing, so
    faithfulness is undefined"* from *"the answer was judged unfaithful"*. The
    original code collapsed both into 0.0.
    """
    if not answer or not chunks or context_text == "":
        return None
    prompt = _FAITHFULNESS_PROMPT.format(
        question=question,
        context=format_context(chunks) if context_text is None else context_text,
        answer=answer,
    )
    try:
        verdict = invoke_structured(llm, prompt, FaithfulnessScore)
    except StructuredOutputError:
        return None
    return (verdict.score - 1) / 4.0


def answer_matches_reference(
    llm: BaseChatModel, question: str, answer: str, reference: str
) -> float | None:
    """LLM-as-judge correctness against a reference answer, in 0-1.

    The original harness had no correctness metric at all: it could report whether
    an answer was *grounded*, never whether it was *right*.
    """
    if not reference or not answer:
        return None
    prompt = _CORRECTNESS_PROMPT.format(question=question, reference=reference, answer=answer)
    try:
        verdict = invoke_structured(llm, prompt, CorrectnessScore)
    except StructuredOutputError:
        return None
    return 1.0 if verdict.correct else 0.0


def abstained(answer: str) -> bool:
    """True if the answer declines to answer rather than guessing.

    Used for negative cases. When the corpus does not contain the answer the
    correct behaviour is an explicit abstention — the answer prompt asks for it,
    but nothing previously tested that the model complies.
    """
    return bool(_ABSTENTION.search(answer))
