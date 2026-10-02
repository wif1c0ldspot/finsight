# Learning & Refinement Plan

**Goal:** build the skillset to land — and justify — a senior/lead agentic-AI
engineering role at market rate, backed by a portfolio artifact that proves it.

## The core idea

Frameworks are a checkbox; **engineering depth is the moat.** In a hot AI
market everyone lists "LangChain / RAG / MCP" on their resume, so those alone
won't get you to the top of a band. What separates a market-rate engineer from
a "prompt engineer" is the ability to ship production systems: retrieval that
can be measured, agents that can be verified, and code that can be tested.

Finsight is built so that **each concept has a home in the codebase.** Learn
the concept, then *refine the corresponding module* to prove you understand it
below the API surface. Learning by extension, not by tutorial.

## Skill map

| Concept | What it actually is | Where in finsight | How to deepen it |
|---|---|---|---|
| **SWE fundamentals** | typing, testing, packaging, CI | `pyproject`, mypy strict, pytest, GitHub Actions | property-based tests (`hypothesis`), Docker, async runtime |
| **Agentic orchestration** | stateful graphs, conditional edges, reflection loops | `graph/` (retrieve→verify→reformulate→answer) | checkpointing, streaming, human-in-the-loop |
| **RAG** | chunking, embeddings, hybrid retrieval, fusion | `rag/` (Chroma + BM25 + RRF) | reranking, query expansion/HyDE, multimodal |
| **Evaluation** | recall, faithfulness, LLM-as-judge | `eval/` | answer-correctness vs ground truth, Ragas metrics |
| **MCP** | a protocol for exposing tools to agents | `mcp/` (server + client) | wire tools *into* the graph (`langchain-mcp-adapters`) |
| **Guardrails** | PII, citation enforcement, input validation | `guardrails/` | prompt-injection detection, structured-output validation |
| **Observability** | traces, latency, token cost | `observability/` | OpenTelemetry/LangSmith, token-cost tracking |
| **Vector search** | embeddings, ANN indexes, distance metrics | `rag/` (Chroma cosine) | benchmark Chroma vs Qdrant/pgvector, tune HNSW |
| **LLM engineering** | prompting, temperature, structured output | `graph/nodes.py` | JSON-mode output, tool/function calling |
| **A2A** | agent-to-agent interoperability | *(not yet)* | a two-agent delegation demo |

## Phased roadmap

*Each phase is a PR-sized slice: build it, measure it, then move on.*

### Phase 1 — Retrieval quality (this is where most RAG value is)
- Add a **reranker** (cross-encoder or LLM rerank) after RRF; measure
  `recall@k` before/after on the golden set.
- Add **query expansion / HyDE** and see if it helps recall on the "hard"
  questions.
- Expand the golden set with questions that *require* hybrid (lexical-heavy)
  vs dense (semantic) retrieval, and report recall per category.

### Phase 2 — Productionisation
- **Async + streaming**: stream tokens from the answer node; run verify in
  parallel where possible.
- **Checkpointing + human-in-the-loop**: persist graph state (SQLite/Postgres
  checkpointer) and add an `approve` node before answering.
- **Containerise**: `docker-compose` with Chroma + Ollama + the agent.
- **OpenTelemetry**: replace the stdlib collector with real spans/traces.

### Phase 3 — Agent tool-use
- Wire the MCP `search_documents` / `get_document` tools into the graph as a
  `ToolNode` (via `langchain-mcp-adapters`), so the LLM decides *when* to
  retrieve, not just *what*.
- Add a **calculator / formula** tool to demonstrate mixed tool-use and
  structured tool schemas.

### Phase 4 — Multi-agent & A2A
- Split into a **researcher agent** + **writer agent** that pass a structured
  handoff (draft → critique → revise).
- Prototype an **A2A** exchange between two agents.

## Refinement exercises (pick one per session)

1. Add a reranker and report the recall delta.
2. Compare the custom chunker against `RecursiveCharacterTextSplitter` and a
   semantic splitter — benchmark chunk count, retrieval recall, faithfulness.
3. Add a checkpointer + an `approve` (HITL) node.
4. Add answer-correctness evaluation (LLM judge vs `ground_truth`) and a
   combined quality score.
5. Wire MCP tools into the graph.
6. Add prompt-injection classification to the input guardrail.
7. Track token cost + per-node latency and print a budget report.
8. Swap the LLM to a different Ollama model and re-run eval — measure the
   quality/latency tradeoff.

## Resources (high-signal, stable)

- LangGraph concepts: <https://langchain-ai.github.io/langgraph/>
- Model Context Protocol spec + SDK: <https://modelcontextprotocol.io>
- Ragas metrics (faithfulness, context precision/recall): <https://docs.ragas.io>
- Ollama: <https://ollama.com>
- Your own agent and retrieval setups are living reference
  implementations — read their agent loops critically against what you build here.

## How to talk about it in interviews

> "I built an agentic RAG system with a self-reflective retrieval loop — the
> agent verifies whether its context is sufficient and reformulates the query
> rather than blindly answering. Hybrid retrieval (dense + BM25, RRF fusion),
> citation enforcement and PII guardrails on the output, and an eval harness
> that reports recall and LLM-judged faithfulness. It runs fully offline on a
> local Ollama stack."

For each competency an interviewer probes, you can point at a module *and* at a
measurement you ran against it. That — a system you built, measured, and
refined — is what a 150K+ agentic-AI engineer looks like on paper.
