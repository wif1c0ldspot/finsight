# Readiness & Development Roadmap

## Current objective and release boundary

Finsight demonstrates a single-turn research agent over a local Markdown corpus:
hybrid retrieval, bounded reformulation, citation checks, a shared MCP tool surface,
and measurable evaluation. The implementation supports that POC objective.

The public repository is intended for local experimentation and evaluation.
Automated verification covers code paths and failure handling with model doubles,
real local Chroma collections, and MCP stdio subprocesses. It does not establish
answer accuracy, live provider compatibility, latency or hardware requirements.
Live-model testing has been deferred; no live benchmark score is claimed.

## Stability work implemented

- Atomic index generations keep the working index available when a later batch
  fails. Bounded embedding/write batches respect Chroma limits. Manifests bind
  provider, model, endpoint, content and generation; malformed registries fail
  with rebuild guidance.
- Configuration and tool boundaries reject invalid counts, queries and nonfinite
  timeouts. Search limits expand the candidate pool consistently. MCP uses strict
  request types and reloads the published generation.
- Graph retries respect the configured budget. Empty, repeated or overlong
  rewrites terminate cleanly; an answer without usable evidence is a deterministic
  abstention. All citation consumers share the bounded evidence map.
- Evaluation validates inputs, isolates failures, exposes scoring denominators,
  saves versioned JSON reports, and reports operational failures with a nonzero
  exit after saving results.
- Public setup instructions distinguish code verification from model evaluation.
  CI covers base and hosted extras across the declared Python versions.

## Next release gate: live evaluation evidence

This is the first priority before claiming demonstrated answer quality.

1. On a machine with a configured model service, run `uv run finsight doctor`, then
   `uv run finsight ingest`. Record the model revisions and hardware outside the report
   if the provider only exposes mutable model aliases.
2. Run representative `uv run finsight ask` questions and `uv run finsight mcp-demo`
   through the installed CLI. Verify citations against source text, including an
   unanswerable question and a question that needs reformulation.
3. Choose a separately configured judge where possible. Run
   `uv run finsight evaluate --output reports/baseline.json` on unchanged input data.
4. Inspect operational and metric errors and the scored denominators before the
   averages. Repeat failed cases after fixing their cause. Manually review model
   and judge outputs; a successful command is not an accuracy threshold.
5. Record observed latency and memory use, preserve the configuration and data
   hashes, and compare repeated runs before publishing benchmark claims. Choose
   quality thresholds based on the intended use case rather than inventing them
   from this nine-case fixture.

The bundled five company summaries and nine golden cases are a smoke-test dataset.
They have no maintained source/date provenance and are insufficient for statistical
claims or current financial research.

## Architecture work still needed

### Evidence quality and retrieval

- **Source provenance and freshness:** introduce source URLs, publication and
  retrieval dates, document revisions and metadata filters. Acceptance: every
  presented fact can be traced to a dated source and filtered by intended scope.
- **Larger evaluation set:** add held-out, multi-document, numerical, misleading,
  unanswerable and adversarial cases with reviewed labels. Acceptance: report
  category scores, error coverage and repeatability with a documented dataset.
- **Semantic claim support:** assess whether cited passages support each claim.
  Existing citation validation only checks reference existence. Acceptance:
  deliberately misattributed claims are detected on a reviewed test set.
- **Token budgets and reranking:** measure tokenizer-aware chunk/context limits and
  a reranker against the saved baseline. Acceptance: improved retrieval/answer
  quality with documented latency and cost; adoption depends on measured benefit.
- **Multilingual sparse retrieval:** replace or supplement the ASCII tokenizer
  when the corpus requires it. Acceptance: lexical matches work for the supported
  languages without degrading the current corpus.

### Storage and resource management

- **Generation cleanup and incremental indexing:** safely reclaim old and failed
  generations after readers release them; update changed documents without full
  re-embedding. Acceptance: reader safety, bounded disk growth and recovery tests.
- **Model identity:** endpoint/model names do not identify immutable weights.
  Record provider model revisions where available and define a rebuild policy.
- **Resource controls:** add run-wide deadlines, cancellation, token/cost budgets
  and capacity limits. Current call timeouts and retry counts do not impose a total
  time or cost ceiling. Acceptance: cancellation and exhaustion terminate cleanly
  with preserved results under injected delays and failures.
- **Observability and resumable evaluation:** add per-node timings, token usage,
  structured traces and incremental report checkpoints. Current reports survive
  case errors but are only written at run completion.

### Required only before a remotely hosted service

- Authentication, authorization, per-user data isolation, rate limits and a
  configured TLS boundary. The included HTTP MCP server is a local demo.
- A threat model for untrusted documents/tools, prompt-injection tests and a
  privacy policy appropriate to the data. Regex redaction is illustrative and
  cannot guarantee removal of sensitive information.
- Deployment health/readiness checks, backup/restore, concurrent-load testing,
  operational monitoring and a documented dependency/security update process.

## Optional product extensions

Conversation memory, checkpoints, streaming, a web UI, numerical/calculator tools,
model-selected tool calls and multi-agent delegation are not implemented. They
are independent extensions, not prerequisites for this single-turn local POC.
Add them only when an evaluation case or user workflow demonstrates a need.
