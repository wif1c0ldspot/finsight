# Architecture

Finsight is a single-turn research agent over a local Markdown corpus. The CLI
coordinates ingestion, the agent graph, evaluation, and an MCP client demo.
Factories in `llm.py` isolate provider construction; graph nodes accept injected
models and retrieval functions for offline testing.

## Index lifecycle

Ingestion splits paragraphs into overlapping, character-bounded chunks and
embeds them. Every build writes a new Chroma collection and a generation-specific
JSON registry. After validating the new collection, it atomically replaces the
version 3 manifest that identifies the published generation. The manifest binds
the logical collection, embedding provider/model/endpoint hash, chunk IDs, and a
content hash. The raw embedding endpoint is not stored. An endpoint change
requires rebuilding even if the model alias stays the same; changes behind an
unchanged alias still require an operator-initiated rebuild.

Embedding and Chroma insertion run in batches bounded by both a 128-chunk cap and
the Chroma client's maximum batch size. This bounds each request by document
count, not tokens. The registry and BM25 index still reside in memory, so this is
a small-corpus design rather than a distributed ingestion system.

A failed build leaves the previous generation available. Changing embedding
dimensions creates a fresh collection rather than deleting live data. Existing
readers remain bound to their generation; the MCP server checks the manifest on
search and reloads when a new generation is published. Old generations are kept
for active readers; automatic garbage collection is not implemented. Interrupted
builds may also retain unpublished collections. Version 1 and 2 indexes require
rebuilding with `finsight ingest`.

Retrieval combines dense vector ranks with BM25 ranks using reciprocal-rank
fusion. Documents without lexical matches do not receive a sparse-rank bonus.
Original component scores remain attached to results for inspection.
When the ASCII sparse tokenizer finds no terms in the entire corpus, retrieval
uses dense vectors alone. Mixed corpora still use BM25 for chunks with matching
ASCII terms, without giving tokenless chunks a sparse-rank bonus.
Queries are validated before embedding calls. Candidate pools expand to cover a
requested `top_k` and are capped by corpus size. Registry decoding/schema failures
raise an integrity error with rebuild guidance.

## Agent graph and evidence

The graph retrieves, verifies context sufficiency, and answers. Failed verification
or citation validation can trigger reformulation within the configured retry
budget. Reformulation receives prior queries and verification/grounding feedback.
An empty, overlong or repeated rewrite stops further retrieval and completes the
answer path. The graph's step limit is derived from the configured retry budget.
Ungrounded final answers carry a citation warning when enforcement is on. If no
source chunk fits the answer context, the graph returns a deterministic abstention
without an answer-generation call or a misleading citation warning.

Context rendering enforces its character limit across the entire block, including
any omission marker. Whole chunks that do not fit are excluded, including an
oversized first chunk. References keep their original retrieval numbers, so gaps
are possible. The renderer returns both text and an explicit citation map.
Generation, grounding validation, CLI source display, and evaluation use that
same evidence map; omitted chunks cannot validate a citation.

Citation validation checks reference existence, not whether every claim follows
from its source. PII output redaction is heuristic. Neither constitutes a general
prompt-injection defense or a guarantee of factual correctness.
Bare eight-digit values remain intact because they can represent financial
amounts; phone redaction requires a recognizable label or an explicit `+65` prefix.

## Provider and MCP boundaries

The answering model, embeddings, and evaluation judge use centralized factories.
The default judge inherits the answering endpoint when using the same provider;
an OpenAI-compatible judge must have a resolved endpoint. Chat and embedding
requests have finite timeout settings. Hosted embeddings use a bounded retry
budget; Ollama embedding requests make one attempt.
Settings reject nonfinite timeouts, invalid counts and overlap that cannot progress
through a chunk. Total service time and cost remain operator-controlled: there is
no run-wide deadline, cancellation API, quota or token budget.

The graph uses the shared MCP tool functions in-process by default. External
clients can use the same tools through stdio or HTTP. The included stdio client
explicitly forwards Finsight settings and supported provider credentials. Tool
failures raise `MCPToolError`, rather than masquerading as successful text.
Finsight environment names are forwarded case-insensitively, matching Settings.
MCP inputs are strict: invalid queries or nonpositive/noninteger search limits fail
before lazy index initialization. Omitting `top_k` uses the configured default.
`doctor` checks local answering and embedding endpoints independently, reports
HTTP failures, and labels hosted providers as unprobed rather than claiming they
are healthy without an authenticated request.

The HTTP transport has no authentication or user isolation. Keep this local unless
an authenticated boundary and resource limits are added. Graph retrieval uses the
shared Python tool implementation, not a network round trip or model-selected tool
call; the standalone stdio demo exercises the actual MCP transport separately.

## Evaluation and validation

Golden-set evaluation measures retrieval recall, reciprocal rank, document-level
nDCG, citation validity, and model-judged faithfulness/correctness. Repeated chunks
from one relevant document receive relevance credit only once in nDCG. The judge
sees the same bounded evidence used for the answer. Negative cases receive a typed
judge verdict checking both refusal and the absence of a substantive or guessed
answer. Empty answers fail; malformed verdicts remain unscored. The summary shows
the eligible, scored, and error counts alongside the abstention rate, so exclusions
are visible. The legacy phrase-match helper is not used for headline metrics.
These small golden sets demonstrate behavior, not statistical quality; the judge
can still make mistakes even when its output satisfies the schema.

The harness validates golden-set types and identities before constructing models.
It isolates agent failures per case and judge failures per metric, preserving the
remaining run. Failures retain sanitized exception types, never exception messages
that may contain credentials. Missing retrieval labels and unavailable metric
verdicts remain unscored with visible denominators. Empty reference-bearing answers
score zero correctness; missing references do not receive correctness credit.

`evaluate --output PATH` writes a versioned JSON report with every result and
summary, allowlisted settings, model identities, endpoint hashes, and before/after
golden/index provenance. It flags observed mutations but does not freeze external
data or model revisions. Reports are written at the end of a run, not incrementally;
process termination can still lose an unfinished run. Operational errors produce
a nonzero CLI exit after saving; malformed judge verdicts are metric errors visible
in the summary. Both coverage and scores must be checked before claiming quality.

Regression tests exercise graph branches, context budgets, citation mappings,
interrupted index publication, embedding dimension changes, live retriever reload,
provider configuration, and real MCP stdio calls using temporary corpora. CI runs
both the base installation and hosted-provider extras on supported Python
3.11–3.13, with Ruff and strict mypy. Models are replaced with deterministic doubles;
live-model quality and runtime characteristics remain to be evaluated.

See [the readiness roadmap](learning-plan.md) for the remaining scope and acceptance
criteria. Production service operation is outside the current POC boundary.
