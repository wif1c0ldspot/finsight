# Architecture

Finsight is a single-turn local research agent. LangGraph coordinates retrieval,
verification, reformulation, answer generation and grading. Provider factories
separate chat, embeddings and evaluation judges. Dependencies accept injected
models/retrievers so tests can verify behavior without live providers.

The supported storage boundary is macOS/Linux on a local POSIX filesystem. The
chunk registry, BM25 index and reuse cache are memory-resident. This is a small-corpus
POC, not a distributed ingestion or multi-tenant service.

## Source records and retrieval

Markdown files may have adjacent `.metadata.json` sidecars containing `source_url`,
`published_at`, `retrieved_at` and `revision`. Missing dates are not inferred;
publication dates use ISO calendar dates. Without an explicit revision, ingestion
hashes the local Markdown text. This identifies the summary, not an archived copy
of the publisher's complete page. The bundled 11 historical source summaries and
40 authored cases are documented in [dataset.md](dataset.md).

Metadata flows through chunks, search payloads, prompts and CLI source display.
`RetrievalFilter` combines fields with AND and list values with OR. Source URLs are
exact matches; publication bounds are inclusive and exclude undated chunks. Empty
filter lists match nothing. Dense candidate selection and sparse candidate scoring
use the same filter scope before reciprocal-rank fusion.

Unicode-aware BM25 tokenization covers letters/numbers across scripts and supports
Han-character matching in unsegmented CJK text. It is not a language-specific
morphological tokenizer. Corpora with no usable tokens fall back to dense search;
queries without lexical matches receive no arbitrary BM25 rank bonus. RRF combines
ranks while retaining original component scores for inspection.

Optional LLM reranking retrieves a larger bounded pool, requires an exact permutation
of displayed reference IDs, and falls back to original RRF ordering on malformed
rankings. It is off by default pending measured benefit. No cross-encoder or
empirical improvement is claimed.

## Index generations and reclamation

Ingestion creates a fresh Chroma collection and generation-specific registry,
validates their contents, then atomically publishes a v4 manifest. It binds logical
collection, embedding provider/model, endpoint hash, optional embedding revision,
chunk IDs and content hash. Raw endpoint URLs and credentials are not stored.
Versions 1–3 require an `ingest` rebuild.

Compatible previous generations provide a chunk-text-to-vector cache. Unchanged
text can reuse embeddings even when its provenance changes; the new registry
records current metadata. Changed text is embedded in bounded batches, capped by
128 chunks and Chroma's batch limit. Removed chunks disappear from the new generation.
A dimension change or incompatible manifest produces a fresh collection, preserving
the published generation if rebuilding fails.

`embed_revision` is operator-supplied identity metadata, not automatic verification
of provider weights. An unchanged alias can change meaning. Update the revision or
use `ingest --full` to bypass reuse after such a change. This caveat applies even
when endpoint and model strings stay the same.

POSIX `flock` leases protect active readers and builders. Lifecycle locking
coordinates lease acquisition, publication and cleanup. Readers remain attached
to their immutable generation; MCP search observes a new manifest on the next
request. Explicit `index-cleanup` is dry-run by default. `--apply` deletes only
owned, inactive generations older than the requested age floor and outside the
retained newest set. The published generation is always protected; unmarked
artifacts are never guessed to be safe. Failed managed builds become eligible
only after their lease is released. This is not a network-filesystem/distributed
lease protocol or an automatic storage quota.

## Graph and evidence contract

The graph retrieves, optionally reranks, verifies context sufficiency, answers and
grades. Failed sufficiency or grounding can trigger bounded reformulation. Rewrite
prompts include previous queries and failure feedback; empty, repeated or overlong
queries terminate retries. Graph step limits account for the configured retry
budget and optional nodes.

A context renderer returns both bounded text and its reference-to-chunk map.
Whole chunks that cannot fit are omitted, including an oversized first chunk.
Provenance and omission markers count against the budget. Reference gaps are valid.
Generation, citation checks, source display and faithfulness evaluation share this
map; an omitted chunk cannot validate a citation. With no usable evidence, the
agent returns a deterministic abstention without answer generation.

Character bounds always apply. Optional chunk/context token bounds use an injected
exact content counter when supplied, otherwise UTF-8 content bytes. That fallback
conservatively bounds byte-based tokenization of the content, excluding provider
framing/tool overhead; it is not a universal model-window calculation. The chunker
clips overlap when needed to preserve forward progress under a smaller budget.

Default grading checks citation existence. Optional semantic grading segments the
answer deterministically and requests one typed support verdict for every segment,
using only that segment's cited sources, including provenance. Missing citations,
invalid/partial verdicts or excess segment coverage fail the check. Persistent
semantic failure ends in abstention after the retry policy, rather than releasing
the rejected answer. This model judgment remains fallible; adversarial fixtures
and instructions to ignore embedded commands are not a general injection defense.
PII output redaction is also heuristic and preserves unlabelled financial amounts.

## Runtime and provider boundaries

`AgentRunner` creates fresh accounting for each invocation. `RunContext` supports
explicit cancellation and an injectable content counter. Supplied contexts inherit
configured limits and may tighten them; conflicting cost rates are rejected before
execution. Node and logical model
boundaries check deadlines, cancellation, call counts, input/output budgets and
configured cost limits. Native structured-output calls, fallback completions and
repair attempts consume logical calls when invoked; hidden SDK retries are not
separate logical calls. Per-request timeouts and retry settings remain necessary.

Cancellation and timeouts are cooperative: arbitrary synchronous provider or
retrieval work cannot be preempted. A run can exceed its wall-clock target while a
call completes, then stop at the next boundary. Output caps are passed to supported
provider clients, but actual accounting is reconciled after responses and does not
guarantee that a provider cannot exceed a requested budget.

Budget counts and provider-reported tokens are tracked separately. Tool arguments
are included in content accounting. Cost fields require explicitly configured
input/output rates; complete reported-usage cost estimates remain unavailable when
usage is missing. These are agent-chat estimates, excluding embeddings, separate
evaluation judges and hidden provider retries, not account-level billing quotas.

Sanitized traces retain fixed statuses, node/model elapsed times, counts and cost
estimates without prompts, source text, response bodies, endpoint URLs or credentials.
Failure summaries survive runtime exceptions. Evaluation stores this agent trace
alongside case results; its runtime figures do not measure separate judge work.

Provider factories preserve configured endpoints, including the inherited judge
endpoint, and enforce finite request timeouts. Compatible backends require explicit
URLs. `doctor` checks local endpoint reachability; hosted services are not declared
healthy without an authenticated model request.

## MCP boundary

The graph calls shared retrieval functions in-process by default. FastMCP exposes
the same functions through stdio or a local HTTP demonstration. Search validates
queries, limits and filters before lazy index creation. `get_document_record`
returns document text and provenance; legacy `get_document` returns text. Listing
and getting reflect current corpus files, while search reflects the published index.

The included stdio client forwards supported environment configuration and preserves
MCP error semantics. HTTP does not implement authentication, authorization, tenant
isolation, TLS termination or quotas. None of the local runtime controls establishes
a secure remote service boundary.

## Evaluation persistence and comparison

The harness validates the complete golden set before constructing models. Category
and `development`/`held_out` filters select a cohort. Agent failures are isolated per
case, judge failures per metric; sanitized exception types and scored denominators
remain visible. Malformed verdicts are unscored. Empty answers fail abstention and
reference correctness. nDCG grants each relevant document credit once despite
multiple returned chunks. Citation validity and semantic factual support are distinct.

Atomic checkpoints are saved before setup and after each fully processed case using
a flushed temporary file and replacement. An interrupted case is absent from the
checkpoint and reruns. Resume requires matching golden/model/config/index and runtime
version fingerprints and the same selection flags. Completed records are retained;
`--retry-failures` explicitly reruns operational or schema errors, not merely low
scores. Reports expose selected/completed/scored counts overall and by category.

Reports allowlist configuration, hash endpoint URLs, and record before/after input
provenance. A disappeared golden file does not discard completed results. Reports
contain questions and answers despite sanitized operational fields and should be
treated as potentially private artifacts.

Comparison requires matching golden/index-content hashes and selected IDs. Deltas
are computed separately for each metric on the intersection of scored case IDs;
paired means/counts/IDs accompany independent aggregates, coverage and errors.
No matched cases means no delta. Model/config differences are labelled, and neither
statistical significance nor an acceptance threshold is inferred.

The public 20/20 split is an authored fixture partition, not independent benchmark
data. Automated checks establish software behavior; live quality, latency, cost and
hardware evaluation are explicitly deferred. Human adjudication and independently
collected labels remain gates for stronger claims.
