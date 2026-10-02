# Architecture

Finsight is a single-turn research agent over a local Markdown corpus. The CLI
coordinates ingestion, the agent graph, evaluation, and an MCP client demo.
Factories in `llm.py` isolate provider construction; graph nodes accept injected
models and retrieval functions for offline testing.

## Index lifecycle

Ingestion splits paragraphs into overlapping, character-bounded chunks and
embeds them. Every build writes a new Chroma collection and a generation-specific
JSON registry. After validating the new collection, it atomically replaces the
version 2 manifest that identifies the published generation. The manifest binds
the logical collection, embedding provider/model, chunk IDs, and a content hash.

A failed build leaves the previous generation available. Changing embedding
dimensions creates a fresh collection rather than deleting live data. Existing
readers remain bound to their generation; the MCP server checks the manifest on
search and reloads when a new generation is published. Old generations are kept
for active readers; automatic garbage collection is not implemented. Version 1
indexes require rebuilding with `finsight ingest`.

Retrieval combines dense vector ranks with BM25 ranks using reciprocal-rank
fusion. Documents without lexical matches do not receive a sparse-rank bonus.
Original component scores remain attached to results for inspection.

## Agent graph and evidence

The graph retrieves, verifies context sufficiency, and answers. Failed verification
or citation validation can trigger reformulation within the configured retry
budget. Reformulation receives prior queries and verification/grounding feedback.
An empty or repeated rewrite stops further retrieval and completes the answer
path. Ungrounded final answers carry a citation warning when enforcement is on.

Context rendering enforces its character limit across the entire block, including
any omission marker. Whole chunks that do not fit are excluded, including an
oversized first chunk. References keep their original retrieval numbers, so gaps
are possible. The renderer returns both text and an explicit citation map.
Generation, grounding validation, CLI source display, and evaluation use that
same evidence map; omitted chunks cannot validate a citation.

Citation validation checks reference existence, not whether every claim follows
from its source. PII output redaction is heuristic. Neither constitutes a general
prompt-injection defense or a guarantee of factual correctness.

## Provider and MCP boundaries

The answering model, embeddings, and evaluation judge use centralized factories.
The default judge inherits the answering endpoint when using the same provider;
an OpenAI-compatible judge must have a resolved endpoint. Chat and embedding
requests have finite timeout settings. Hosted embeddings use a bounded retry
budget; Ollama embedding requests make one attempt.

The graph uses the shared MCP tool functions in-process by default. External
clients can use the same tools through stdio or HTTP. The included stdio client
explicitly forwards Finsight settings and supported provider credentials. Tool
failures raise `MCPToolError`, rather than masquerading as successful text.

## Evaluation and validation

Golden-set evaluation measures retrieval recall, reciprocal rank, document-level
nDCG, citation validity, and model-judged faithfulness/correctness. Repeated chunks
from one relevant document receive relevance credit only once in nDCG. The judge
sees the same bounded evidence used for the answer. Negative cases are scored for
abstention. These small golden sets demonstrate behavior, not statistical quality.

Regression tests exercise graph branches, context budgets, citation mappings,
interrupted index publication, embedding dimension changes, live retriever reload,
provider configuration, and real MCP stdio calls using temporary corpora. CI runs
both the base installation and hosted-provider extras, with Ruff and strict mypy.

Remaining scope includes semantic claim verification, larger quality datasets,
per-node/token telemetry, token-aware context budgets, and old-generation cleanup.
