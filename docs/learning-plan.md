# POC roadmap and evidence gates

## Current boundary

The core local POC roadmap is implemented. Finsight supports single-turn research
over a small, memory-resident corpus on macOS/Linux with a local POSIX filesystem.
Automated verification uses deterministic model doubles, local Chroma persistence,
MCP subprocesses and injected failures. This establishes software behavior, not
live answer accuracy or production service readiness.

**Live model quality, latency, cost and hardware testing are explicitly deferred
by request.** This document records those remaining evidence gates; it does not
claim they have been completed or authorize running them.

## Completed core work

### Evidence and retrieval

- Eleven historical primary-source summaries carry URLs, publication/retrieval
  dates and local revisions. Metadata reaches chunks, prompts, CLI output and
  `get_document_record`; filters scope document IDs, URLs and publication dates.
- Unicode BM25 with Han character tokens complements dense retrieval through RRF.
  Nonmatching lexical results do not gain an arbitrary rank bonus.
- Optional LLM reranking validates complete candidate rankings and falls back to
  RRF on invalid output. Optional semantic support checks require a verdict for
  every answer segment against its own cited sources, ending in abstention when
  checks remain unsuccessful. Both features default off pending empirical evidence.
- Character and optional token-content bounds constrain chunking/context. Exact
  counters can be injected; UTF-8 content bytes are the fallback, excluding
  provider framing overhead. All citation consumers share the rendered evidence map.

### Storage and execution

- Manifest v4 publishes immutable index generations atomically and binds provider,
  model, endpoint hash, optional embedding revision and content. Versions 1–3
  rebuild via `ingest`.
- Compatible unchanged text reuses vectors while changed/deleted content and
  provenance appear in the new generation. `ingest --full` bypasses reuse when a
  mutable model alias changes behind an unchanged identity.
- POSIX reader/builder leases and explicit dry-run/apply cleanup protect active
  and published generations. Retention/age controls leave unmanaged artifacts alone.
- Per-invocation logical call, content-token, output and configured cost controls
  share fresh runtime accounting. Deadlines/cancellation are cooperative and do not
  preempt synchronous calls. Sanitized traces preserve timings, usage and terminal
  statuses; cost figures exclude embeddings and evaluation judges.

### Evaluation and usability

- Forty authored cases span single-fact, numerical, multi-document, temporal,
  misleading, unavailable and adversarial questions. The 20 development / 20 held-out
  partitions are public fixtures from one corpus, **not independent benchmark data**.
- Golden validation precedes model construction. Agent/judge failures are isolated;
  unscored results and denominators remain visible, including category coverage.
- Atomic per-case checkpoints survive interruptions. Resume checks input/model/config/
  index identity and selection; recorded failures can be explicitly retried.
- Report comparison retains independent aggregates but computes each metric delta
  only on paired scored IDs. Pair counts, coverage and errors accompany deltas;
  no significance or quality threshold is inferred.
- CLI examples, `.env.example`, strict MCP inputs, provider construction tests and
  CI for base/hosted extras support reproducible setup.

## Deferred evidence gates

When live testing is explicitly resumed, use saved artifacts rather than relying
on a few convincing answers:

1. Record hardware, provider/model revisions and any mutable alias limitations.
   Check `doctor`, ingest the documented corpus, and manually inspect representative
   answerable and unanswerable CLI/MCP responses against source text.
2. Save development-cohort baseline reports with fixed corpus/golden identities and
   budgets. Inspect operational errors, incomplete cases and scored denominators
   before reading averages. Resume with the same selection flags; retry errors
   explicitly after addressing their cause.
3. Test reranking, semantic verification and exact token counters against that
   baseline. Record additional logical calls, wall time, provider usage and memory.
   Agent traces exclude evaluation-judge and embedding work; measure those costs
   separately when making end-to-end comparisons.
4. Review disputed answers and judge verdicts with humans. Check dates, currencies,
   qualifications and numerical reasoning. A correctly formatted judge verdict can
   still be wrong, and a citation-valid answer can still misstate its source.
5. Use the public held-out partition only with its authorship/tuning limitations
   disclosed. Obtain broader independently collected cases and human-adjudicated
   labels before generalizing results. Define acceptance thresholds from the intended
   workflow; neither the fixture size nor a successful command establishes them.

Dataset sourcing, category definitions and limitations are in [dataset.md](dataset.md).
No live benchmark score or measured improvement is bundled.

## Operator responsibilities and remaining limitations

An embedding revision is operator-supplied metadata, not a provider weight hash
verified by Finsight. Maintain revision discipline and force a full rebuild when
an alias changes without an identity update. Source dates describe historical
snapshots; there is no automatic freshness monitor or guarantee about current facts.

Fallback token accounting is content-only. Cost estimates use explicitly supplied
rates and reported usage where available; they are not an account billing ceiling.
Logical call counts do not enumerate hidden SDK retries. Cooperative cancellation
cannot terminate an arbitrary already-running synchronous call.

The current in-memory registry/BM25/cache and local POSIX leases constrain scale
and deployment. Unicode lexical handling is a baseline, not comprehensive linguistic
segmentation. PII heuristics, semantic checks and adversarial fixtures do not form
a complete privacy or prompt-injection security boundary.

## Optional deferred service and product work

Before exposing a remote service, design authentication/authorization, tenant/data
isolation, TLS termination, request quotas, backup/restore, deployment health checks,
load testing and operational monitoring. These are service prerequisites, not
completed properties of the included HTTP demo. Extend the threat model and privacy
policy for the actual data and users.

A web UI, streaming responses, conversation memory, calculator/numerical tools,
model-selected tool calls and multi-agent research workflows remain optional
extensions. Evaluation checkpoints are implemented; conversational persistence is
not. Add product features when a concrete workflow or evaluation case warrants them.
