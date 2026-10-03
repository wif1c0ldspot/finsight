# Finsight

A financial research agent demonstrating an **adaptive RAG loop** — hybrid retrieval
(dense vectors + BM25, fused by reciprocal-rank fusion), a self-verifying
retrieve/reformulate cycle, and citation-checked answers with a small evaluation
harness. The same retrieval capability is exposed as an **MCP tool server**, so other
agents can drive it, and the agent itself consumes those tools.

Runs fully offline on a local Ollama stack by default, and against OpenAI, Anthropic
or any OpenAI-compatible endpoint by configuration.

## What it does

Given a question, the agent:

1. **Retrieves** the top context via hybrid search (dense vector + sparse BM25,
   fused with reciprocal-rank fusion).
2. **Verifies** the context is sufficient — a typed JSON verdict, not a string match.
3. **Reformulates** the query and re-retrieves if the context is insufficient
   (bounded loop).
4. **Answers** with inline citations.
5. **Grades its own grounding** — the answer must cite a real chunk and no
   non-existent ones. An ungrounded answer triggers another retrieval attempt, or is
   returned with a visible citation warning if the budget is spent.

Two decision points, both bounded by `max_retrieval_attempts`.

## Stack

| Layer | Technology |
|-------|-----------|
| Agent graph | LangGraph (stateful graph, conditional edges) |
| LLM | Ollama `qwen3:8b` by default; OpenAI / Anthropic / OpenAI-compatible by config |
| Embeddings | Ollama `nomic-embed-text` by default; OpenAI by config |
| Vector store | ChromaDB (cosine) |
| Sparse search | BM25 (`rank-bm25`), rebuilt in memory from the chunk registry |
| Fusion | Reciprocal-rank fusion (RRF) |
| Tool protocol | MCP server + client, shared with the agent's own retrieval path |
| Evaluation | Rank-aware retrieval metrics + typed LLM-as-judge |
| CLI | Typer + Rich |

## Quickstart

```bash
# 1. Install (creates .venv with a pinned Python 3.12)
uv sync

# 2. Ensure Ollama is running, then pull each model separately
ollama pull qwen3:8b
ollama pull nomic-embed-text

# 3. Check configuration, paths and service reachability
uv run finsight doctor

# 4. Index the sample corpus
uv run finsight ingest

# 5. Ask a question
uv run finsight ask "Who founded Airwallex and when?"

# 6. Run the evaluation
uv run finsight evaluate

# 7. Demo the MCP server
uv run finsight mcp-demo "What is Stripe's core product?"
```

## Using a hosted provider

Hosted backends are optional dependencies, so the base install stays offline.

```bash
# The example below uses Anthropic generation and OpenAI embeddings.
uv sync --extra all
```

```bash
# Anthropic for generation, OpenAI for embeddings
export FINSIGHT_LLM_PROVIDER=anthropic
export FINSIGHT_LLM_MODEL=claude-sonnet-4-5
export FINSIGHT_EMBED_PROVIDER=openai
export FINSIGHT_EMBED_MODEL=text-embedding-3-small
export ANTHROPIC_API_KEY=... OPENAI_API_KEY=...
```

For a local OpenAI-compatible gateway with Ollama embeddings, use this
alternative configuration in a fresh shell:

```bash
uv sync --extra openai
# Any OpenAI-compatible gateway (vLLM, LM Studio, OpenRouter, a proxy, ...)
export FINSIGHT_LLM_PROVIDER=openai_compatible
export FINSIGHT_LLM_BASE_URL=http://localhost:8000/v1
export FINSIGHT_LLM_MODEL=Qwen/Qwen3-8B
# Use your gateway's key; a nonempty placeholder works for a gateway without auth.
export FINSIGHT_LLM_API_KEY=local
export FINSIGHT_EMBED_PROVIDER=ollama
export FINSIGHT_EMBED_MODEL=nomic-embed-text
```

`openai_compatible` requires an explicit base URL — the settings model rejects it
without one rather than silently talking to the wrong host.

> **Changing `embed_model` invalidates the index.** The manifest records which model
> produced the stored vectors and refuses to serve them if the setting no longer
> matches. Re-run `finsight ingest`.

Index rebuilds publish a new immutable generation atomically, and running MCP
servers refresh on the next search. Existing version 1 indexes need a one-time
`finsight ingest`. Previous generations remain on disk to protect active readers;
automatic cleanup is not yet implemented.

## The MCP surface

`finsight/mcp/tools.py` is the single implementation of the retrieval tools.
`mcp/server.py` registers those functions for external MCP hosts; the agent graph
calls the same functions in-process when `FINSIGHT_USE_MCP_TOOLS` is set (default).
One contract, two transports — so the agent and an external host cannot drift apart.

Tools: `search_documents`, `list_documents`, `get_document`.

```bash
uv run python -m finsight.mcp.server          # stdio
uv run python -m finsight.mcp.server --http   # streamable-http on :8000
```

## Evaluation

`uv run finsight evaluate` runs the golden set in `data/golden/golden.json`.

Retrieval metrics are **rank-aware** and computed over answerable cases only:

- `recall_at_k` — fraction of expected documents in the top k.
- `mrr` — reciprocal rank of the first relevant document.
- `ndcg_at_k` — position-weighted ranking quality.

Answer metrics:

- `grounded_rate` — answers citing a real chunk with no dangling markers.
- `dangling_citation_cases` — answers citing sources that do not exist.
- `mean_faithfulness` — LLM-as-judge support score, or `null` when retrieval
  returned nothing (a retrieval failure is not an unfaithful answer).
- `mean_correctness` — LLM-as-judge agreement with the reference answer.
- `abstention_rate_on_negative` — the judge-assessed fraction of scored negative
  cases that decline to answer without also supplying a substantive or guessed
  answer. An empty answer scores as a failure.
- `n_abstention_scored` / `n_abstention_errors` — how many of the `n_negative`
  cases received a usable abstention verdict versus an unparseable verdict.
  Unscored cases are excluded from the rate; it is `null` when none were scored.

> **Judge bias.** By default the judge is the answering model, which is self-grading.
> The summary prints a caveat when that is the case. Set `FINSIGHT_JUDGE_MODEL` (and
> optionally `FINSIGHT_JUDGE_PROVIDER`) to a different family for an honest score.

## Project layout

```
src/finsight/
  graph/         LangGraph agent (state, nodes, builder)
  rag/           ingestion, hybrid retrieval, RRF fusion, index manifest
  mcp/           shared tool layer + MCP server and client
  eval/          rank-aware metrics + golden-set harness
  guardrails/    PII redaction, query validation, grounding assessment
  observability/ step timing + metrics collector
  structured.py  provider-agnostic typed model output
  cli.py         Typer entrypoint
```

## Configuration

All settings are overridable via `FINSIGHT_<FIELD>` environment variables or `.env`
(see `src/finsight/config.py`). The settings model validates cross-field constraints —
for example, `openai_compatible` without a base URL is a load-time error, not a
runtime surprise.

## Known limitations

- **Single-turn.** No conversation memory or session state.
- **No reranking.** RRF fusion feeds top-k directly; a cross-encoder rerank is the
  obvious next step.
- **No metadata filtering.** Retrieval cannot be scoped by document, source or date.
- **ASCII sparse tokenizer.** Corpora without any ASCII word/number tokens use
  dense retrieval only; other scripts do not receive BM25 lexical matching.
- **Small demo corpus.** Five short documents and nine golden cases — enough to prove
  the machinery, not enough for statistical claims.
- **Heuristic chunking.** Paragraph-packing at a character budget, not token-aware.
- **Character-bounded context.** Only whole chunks that fit are shown; citations
  are checked against that exact subset. Character limits are not token limits.
- **Deterministic grounding check.** Citation markers are validated mechanically; that
  a claim is *supported* by the chunk it cites is not verified.

## Docs

- [`docs/architecture.md`](docs/architecture.md) — design decisions and rationale.
- [`docs/learning-plan.md`](docs/learning-plan.md) — learning & refinement roadmap.
