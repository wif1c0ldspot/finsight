# Finsight

A financial research agent demonstrating an **adaptive RAG loop** — hybrid retrieval
(dense vectors + BM25, fused by reciprocal-rank fusion), a self-verifying
retrieve/reformulate cycle, and citation-checked answers with a small evaluation
harness. The same retrieval capability is exposed as an **MCP tool server**, so other
agents can drive it, and the agent itself consumes those tools.

Runs locally on Ollama by default, and against OpenAI, Anthropic or an
OpenAI-compatible endpoint by configuration. Local inference can run offline
after dependencies and models have been downloaded.

## Release status

Finsight is a **public proof of concept for local experimentation and evaluation**.
Automated tests cover orchestration, real local Chroma persistence, MCP stdio,
provider construction and failure handling using deterministic model doubles.
Live-model answer quality, latency and hardware requirements have **not yet been
validated**. The bundled company summaries are demonstration fixtures, not a
maintained financial dataset.

Start with the sample corpus, inspect cited evidence, and save an evaluation report
before comparing models. See the [readiness and development roadmap](docs/learning-plan.md)
for the remaining work and the criteria for a stronger release claim.

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

When no usable source text reaches the answer stage, it returns a deterministic
abstention. Citation checks validate reference existence; they do not prove a
claim is supported by the referenced text.

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
# 1. Clone and install (requires uv; creates .venv with a pinned Python 3.12)
git clone https://github.com/wif1c0ldspot/finsight.git
cd finsight
uv sync --frozen

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
uv run finsight evaluate --output reports/baseline.json

# 7. Demo the MCP server
uv run finsight mcp-demo "What is Stripe's core product?"
```

Run commands from the repository root: the default corpus and golden-set paths
are relative to it. A wheel installation alone does not include the sample data.
For a code-only check without Ollama, credentials or model downloads:

```bash
uv run --frozen --no-sync ruff check .
uv run --frozen --no-sync mypy src
uv run --frozen --no-sync pytest -q
```

These checks verify software behavior, not live model accuracy. `doctor` checks
local service reachability; it does not certify that a configured model is loaded.

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

> **Changing the embedding provider, model or endpoint requires reindexing.**
> The manifest binds those settings to the stored vectors and refuses to serve
> them if they no longer match. Re-run `finsight ingest`. Also rebuild if the model
> behind an unchanged alias is replaced; aliases are not immutable model revisions.

Index rebuilds publish a new immutable generation atomically, and running MCP
servers refresh on the next search. Existing version 1 or 2 indexes need a one-time
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

The HTTP transport is a local demo without application authentication, per-user
isolation or rate limits. Public repository availability does not make this a
publicly deployable API; remote service deployment needs those controls first.

## Evaluation

`uv run finsight evaluate --output reports/baseline.json` runs the golden set in
`data/golden/golden.json` and saves a versioned JSON report. `--output` is optional;
the summary is always printed. Reports include per-case answers, scores, sanitized
error types and scoring counts, plus model/configuration, golden-set and index
provenance. Credentials and raw endpoint URLs are excluded; questions and answers
still contain corpus material. The `reports/` directory is ignored by Git.

Malformed golden cases are rejected before model setup. Agent and judge failures
are isolated so later cases still run; operational failures produce exit code 1
after the report is saved. Malformed judge verdicts are unscored and counted as
metric errors. A zero exit code alone is not a quality pass: inspect scoring
coverage and errors as well as the averages. Observed index or golden-set changes
during a run are flagged in the report; rerun against unchanged inputs for a
comparison.

Each golden case requires a nonempty `question`. Optional `id` values must be unique
(otherwise `case-N` is assigned), `answerable` must be a JSON boolean, and
`expected_doc_ids` must be a list of unique document IDs. Negative cases use
`"answerable": false` with no expected documents. `ground_truth` supplies the
reference answer for correctness scoring.

Retrieval metrics are **rank-aware** and computed over successful answerable cases
with nonempty expected-document labels. Cases without labels remain unscored:

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

`n_agent_errors` and `n_operational_failures` expose infrastructure failures;
`n_retrieval_scored`, `n_grounding_scored`, `n_faithfulness_scored`, and
`n_correctness_scored` expose the denominators behind each metric. Failed agent
cases are unscored, so high averages with poor coverage do not demonstrate quality.

> **Judge bias.** By default the judge is the answering model, which is self-grading.
> The summary prints a caveat when that is the case. Set `FINSIGHT_JUDGE_MODEL` (and
> optionally `FINSIGHT_JUDGE_PROVIDER`) to a different family to reduce that bias.
> Independent judges can still be wrong; manually inspect a sample of verdicts.

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
- **Heuristic PII redaction.** Phone labels and explicit `+65` prefixes are recognized;
  bare eight-digit amounts are preserved. This is not a complete privacy filter.
- **Local storage and full rebuilds.** No incremental updates or generation cleanup;
  the registry and BM25 index are loaded into memory. Failed builds may leave
  unpublished collections on disk while preserving the working index.
- **Local service boundary.** No authentication, tenant isolation or resource quotas.
- **Limited telemetry.** Command timing is available; per-node traces, token usage
  and cost budgets are not implemented.

## Docs

- [`docs/architecture.md`](docs/architecture.md) — design decisions and rationale.
- [`docs/learning-plan.md`](docs/learning-plan.md) — readiness, evaluation procedure and roadmap.
